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
run generalized PA grounding → PA authors and semantically reviews one cited
two-state `target_feature` → PA chooses and validates a provisional allocation
→ commit its ontology projection and v6 completion → reconstruct the target
feature for the selected RA. CAD sources are presented to PA through opaque
handles; canonical identity is revealed only after acceptance. A registered
source is authorized input, not a guaranteed answer; numeric geometry is derived
only from PA-selected evidence when a verifier requires it.
