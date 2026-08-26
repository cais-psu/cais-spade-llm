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
Phase 4.0 is wired into Phase 3 through package-local Python contracts and adds
no shared schema file here. Retrieval, interpretation, ontology delta, and
Phase 4.3-style decision records share an aligned operation number.

Phase 4.1 uses a strict structured-output schema owned by
`tools/document_evidence/interpreter.py`. It records ordered page evidence,
entity keys, relations, literal facts, uncertainty, and unresolved evidence
needs before compiling a generic delta for the shared validator.

Phase 4.2A typed geometry records are owned by
`tools/rgb_d_cad_grounding/preprocessor.py`. `CADMeshRecord` references complete
triangle and facet-normal arrays; `ColoredPointCloudSetRecord` references one
calibrated colored point cloud per camera optical frame. Their generic deltas
contain no RDF assertions and leave correspondence and pose unresolved.

Phase 4.2B1 records are owned by
`tools/rgb_d_cad_grounding/segmenter.py` and `diagnostic.py`.
`RGBDSegmentationRecord` references one camera-local `uint16` label mask per
camera and stores automatic role, candidate count, bounds, centroid, hashes, and
fixed parameters. `RGBDSegmentationStatus` is the compact status-only UI
contract with `idle`, `running`, `ready`, or `failed`, source and assembly
candidate counts, and unevaluated identity, CAD correspondence, and pose. These
records contain no ontology assertions and express neither matching, context
completion, nor assembly readiness.

Phase 4.2B2A adds `CADSizeCorrespondenceRecord`, owned by
`tools/rgb_d_cad_grounding/size_correspondence.py`. It binds one exact validated
CAD record to one validated segmentation record and preserves CAD and artifact
hashes, two-dimensional principal-size comparisons, relative errors,
deterministic ranking, and any unique candidate's median center and optical
frame. Its `CAD_correspondence` state is `accepted`, `ambiguous`, or `rejected`;
its `location` state is `available`, `ambiguous`, or `unavailable`; pose remains
`not_evaluated`. The compact `RGBDSegmentationStatus` can show these two states
without exposing coordinates, scores, masks, or thresholds. The record is not
an ontology assertion, complete pose, context-completion assessment, or
assembly-readiness claim.

Planned contracts cover:

- `target_feature`, complete target pose, insertion axis, and tolerances
- fresh resource state and resource-owned primitive catalog
- `primitive_steps`
- state checks and IK/collision/trajectory validation feedback
- rejected candidate revision and accepted candidate handoff

Phase 0 intentionally defines no JSON, YAML, or Python schema.
