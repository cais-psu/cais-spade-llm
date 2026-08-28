# Schemas

This directory remains reserved for shared data templates and contracts. Schemas
define the shape and required fields of stage inputs and outputs; they are not
experiment results.

Phase 4.0 ontology code is not stored as a schema in this directory. Shared
immutable PPR TBox loading and validation live in `ontology/ppr_tbox.py`, while
the project-authoritative schema-only vocabulary lives in
`ontology/spec2primitives_ppr_tbox.owl`, and
PA-owned writable interaction-ABox behavior lives in
`agents/pa/product_context.py`. The TBox supplies allowed classes and property
signatures; it does not require any particular relation for a requirement.
The validator rejects wrong-domain/range relations and RA,
primitive-offering, and `capableOf` assertions from the PA ABox.
A future RA resource ABox remains separately owned; only the TBox is shared.
Phase 4.0 is wired into Phase 3 through package-local Python contracts and adds
no ontology file to this directory. Retrieval, interpretation, ontology delta, and
Phase 4.3-style decision records share an aligned operation number.

Document handling uses separate strict contracts. `DocumentOverviewRecord`, owned by
`tools/document_evidence/interpreter.py`, contains only a summary,
surface-form observations, uncertainty, and exact page refs.
`DocumentEvidenceRecord` has the same ontology-neutral evidence shape for a
deterministically selected targeted page subset. `OntologyGroundingProposal`,
owned by `agents/pa/ontology_grounding.py`, is a late untrusted mapping from
directly supported `GroundingStatement` IDs. It has numbered new individuals,
not model-authored entity keys; `specification` is supplied separately and
controller-owned. Only the existing triple-delta validator can accept compiled
assertions.

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

The simple remaining Phase 4.2B2 pose increment adds
`CADPoseEstimationRecord`, owned by
`tools/rgb_d_cad_grounding/pose_estimation.py`. It binds one validated size
correspondence to ranked camera-frame registration hypotheses and stores a
translation, rotation matrix, quaternion, and camera-from-CAD transform only
for a clear fit. `pose` is `accepted`, `ambiguous`, or `rejected`; qualified
competing rotations remain in the typed record. The compact status may display
the pose state but never exposes coordinates or transforms.

The simple frame-conversion increment adds `CameraToRobotCalibrationRecord` and
`RobotFramePoseRecord`, owned by
`tools/rgb_d_cad_grounding/frame_conversion.py`. The calibration record binds
one caller-approved rigid transform to exact source and target frames, a
validity window, provenance, and a deterministic payload hash. The robot-frame
record hashes both inputs and stores the composed translation, rotation matrix,
quaternion, and transform only when the camera pose is accepted. Its
`robot_frame_conversion` state is `accepted`, `ambiguous`, or `rejected`; the
compact status exposes only that state.

Implemented pre-RA grounding contracts in
`agents/pa/grounding_contracts.py` cover:

- `GroundingStatement`, with plain text, `directly_stated` or `inferred`
  status, source refs, and a reason;
- `InformationNeed`, with a question, required flag, accepted record types,
  source refs, state, and answer statement IDs;
- `GroundingActionAttempt`, uniquely tracking
  `need_id + provider_id + source_revision`;
- `GroundingDecision` and versioned `GroundingSession` state;
- `GroundingProducerDescriptor`, describing one provider's accepted evidence,
  produced records, prerequisites, availability, and estimated cost;
- `TypedContextBinding` and `ProductContextView`, joining compact ABox meaning
  with validated record refs, hashes, status, frames, validity, and source
  details;
- `TypedGroundingContract`, preserving supported statements, missing
  information, and current-TBox gaps; and
- append-only `PAClarification` records plus `PAContextGroundingCompletion`
  version 2, which pins the session, ontology, typed records, sources, and
  clarification hashes.

Earlier PA draft, need, and completion formats are not loaded or migrated. New
production interactions write only the generalized session and completion
version 2.

Planned downstream contracts cover:

- robot-independent `TaskTransitionContract`
- versioned `CompositionContextBundle` records with task, ABox, binding,
  selected-resource, and catalog fingerprints
- a complete selected-RA-authoritative primitive-only catalog snapshot with
  runtime-determined cardinality and exact symbols
- RA-authored `PrimitiveProgramDraft`, deduplicated `MissingContextBatch`, PA
  batch response, and progress or no-progress decision records
- fully bound `primitive_steps`, state checks, IK/collision/trajectory feedback,
  rejected candidate revision, and accepted candidate handoff

The downstream names are planned boundaries, not current schema
implementations. No RA resource ABox, task-transition contract, composition
bundle, missing-context batch, or primitive candidate is produced by the
current runtime.

Phase 0 intentionally defines no JSON, YAML, or Python schema.
