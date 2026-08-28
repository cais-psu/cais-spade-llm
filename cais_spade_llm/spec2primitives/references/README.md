# References

Reusable static Spec2Primitives inputs are separated by ownership:

- `products/` contains approved product documents and the manifest that
  references the candidate CAD corpus.
- `resources/` is reserved for approved resource references, including
  resource-owned primitive catalogs for each exact RA identifier.

Runtime observations, retrieved snapshots, plans, `primitive_steps`, messages,
and validation results do not belong here. They belong to one interaction under
`../contexts/`.

Manuals enter only through explicit `products/approved_sources.json`
registration. Upload and directory auto-discovery are intentionally out of
scope. Registering a PDF does not infer its purpose; preparation creates a
generic cache later under `../contexts/source_cache/`.

The lifecycle is: register source → prepare generic cache → start the system →
run generalized PA grounding → create a late semantic projection → expose the
validated ontology projection and typed grounding contract to future RA. A
registered source is authorized input, not a guaranteed answer.
