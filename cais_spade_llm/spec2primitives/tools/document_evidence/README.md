# Document evidence

Every source must be an exact approved PDF in `references/products/approved_sources.json` with matching bytes/hash. Retrieval prepares or snapshots a deterministic `DocumentSourceIndexRecord` with ordered page text, rendered images and provenance. It makes no task-specific interpretation or ontology assertion.

PA can call `query_document` with its own question after retrieval. The bounded document VLM receives that question and the approved pages and persists a cited `DocumentQueryRecord` with uncertainty. No controller-authored question prescribes a component, destination or expected answer.

Existing overview preparation/cache commands and readers remain available: `poetry run python -m cais_spade_llm.spec2primitives.tools.document_evidence.prepare --all` or `--context-ref <exact-ref>`. Generic overview records are distinct from the active source-index/question path. Cache identity pins source bytes and applicable configuration/schema. Startup/cache inspection does not ask PA to reason.

The independent target review reads page text/images directly. Its first assessment excludes PA document questions and generated answers; its second review may consider cited interpretations alongside their sources. A document can establish an intended relation, but cannot establish current physical attachment without observation evidence. Source caveats persist after acceptance and UI reload.

No vector database, learned ranking or independent-page retrieval is claimed. Document tools do not label RGB-D candidates, select an arm or create coordinates. Approved source content is allowed; simulator/evaluator answers are forbidden. See [bias validation](../../BIAS_VALIDATION.md).
