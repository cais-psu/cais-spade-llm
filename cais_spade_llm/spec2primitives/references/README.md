# References

Reusable static Spec2Primitives inputs are separated by ownership:

- `products/` contains the approved product document and the inventory that
  references the candidate CAD corpus.
- `resources/` is reserved for approved resource references, including
  resource-owned primitive catalogs for each exact RA identifier.

Runtime observations, retrieved snapshots, plans, `primitive_steps`, messages,
and validation results do not belong here. They belong to one interaction under
`../contexts/`.
