# References

`products/approved_sources.json` is the exact document/CAD retrieval authority. Sources and hashes are configuration-driven; the runtime cannot add arbitrary files. `resources/` documents configured robot capabilities and selected-RA catalog ownership.

Approved CAD filenames and source document part names are allowed evidence before grounding. They do not establish which observed candidate has a task role. Source retrieval supplies document text/images, CAD measurements and neutral observation evidence. PA interprets these sources; deterministic validation checks proposal structure and evidence lineage. Product recognition does not receive resource-selection results or evaluator answers.

This Markdown documentation, scene setup notes, simulator names, spawn poses and evaluator labels are not the approved recognition corpus. Never pass scene manifests or this directory's setup prose to PA as source evidence. Keep calibration authority internal and model-facing sensor metadata opaque.

Current Phase 4 uses proposal/evidence and required live MoveIt reachability before completion. Historical source/proposal records remain read-only history. See [bias validation](../BIAS_VALIDATION.md) for corpus-label and distractor experiments.

In Phase 5, retrieved product/CAD records remain evidence, not automatically resolved robot geometry. A descriptive destination cannot establish placement coordinates. New composition interfaces exclude `model_name`; the future execution adapter must establish that simulator binding for the recognized physical instance, rather than infer it from a CAD filename. See [composition bindings](../agents/ra/README.md) and [bounded evidence resolution](../VALIDATION_AND_REVISION.md).
