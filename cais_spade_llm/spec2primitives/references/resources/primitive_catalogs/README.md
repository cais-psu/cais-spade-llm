# Primitive catalogs

Reserved for approved references to resource-owned primitive catalogs, divided
by exact RA identifier. Do not copy a shared RobotAgent primitive catalog here
or create a second authority.

At runtime the selected RA remains authoritative for its complete current
catalog snapshot and fingerprint. Catalog cardinality is runtime-determined;
there is no eight-entry invariant. Preserve every primitive symbol exactly and
never normalize, rename, replace, or silently filter an entry.

Phase 5.2A gives the selected RobotAgent LLM this exact snapshot with the
reconstructed PA `target_feature`. The LLM alone proposes the symbol order;
deterministic code checks exact membership and lineage without turning this
reference directory into a recipe or second catalog authority.
