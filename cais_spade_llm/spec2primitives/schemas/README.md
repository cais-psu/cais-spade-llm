# Schemas

This directory remains reserved for shared data templates and contracts. Schemas
define the shape and required fields of stage inputs and outputs; they are not
experiment results.

Phase 4.0 ontology code is not stored as a schema in this directory. Shared
immutable PPR TBox loading and validation live in `ontology/ppr_tbox.py`, while
PA-owned writable interaction-ABox behavior lives in
`agents/pa/product_context.py`. Together they represent `specification defines
required feature` and `requested process realizes the same feature`, while
rejecting RA, primitive-offering, and `capableOf` assertions from the PA ABox.
A future RA resource ABox remains separately owned; only the TBox is shared.
Phase 4.0 is not yet wired into the Phase 3 retrieval loop and adds no shared
schema file here.

Planned contracts cover:

- `target_feature`, target pose, insertion axis, and tolerances
- fresh resource state and resource-owned primitive catalog
- `primitive_steps`
- state checks and IK/collision/trajectory validation feedback
- rejected candidate revision and accepted candidate handoff

Phase 0 intentionally defines no JSON, YAML, or Python schema.
