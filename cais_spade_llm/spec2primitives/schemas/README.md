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

Document handling uses a strict `DocumentOverviewRecord` version 2 contract,
owned by `tools/document_evidence/interpreter.py`. One retrieval contains every
ordered page's extracted text, rendered-page hash, neutral visual observations,
uncertainty, and exact page citations. New runs do not produce targeted
`DocumentEvidenceRecord` values. Older overview and targeted-evidence records
remain read-only recovery inputs where a loader still supports them.

`OntologyGroundingProposal`, owned by
`agents/pa/ontology_grounding.py`, is built from the requirement, ontology, and
retrieved typed records together. New proposals use schema version 5 and carry
per-individual and per-relation citations. Validation first produces a
transient candidate and provisional ABox. It neither writes an accepted
proposal nor merges RDF assertions at that boundary. Only after the active
consumer's typed prerequisite chain passes does the system persist the accepted
version-5 proposal and merge the compiled assertions. Versions 3 and 4 remain
read-only recovery contracts.

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

The frame-conversion increment adds `CameraToRobotCalibrationRecord`,
`RobotFrameLocationRecord`, and `RobotFramePoseRecord`, owned by
`tools/rgb_d_cad_grounding/frame_conversion.py`. The calibration record binds
one caller-approved rigid transform to exact source and target frames, a
validity window, provenance, and a deterministic payload hash.
`RobotFrameLocationRecord` version 1 hashes the accepted correspondence and
calibration inputs and stores the translated candidate center for the current
location-based resource consumer. `RobotFramePoseRecord` stores the composed
translation and orientation only for an orientation-sensitive consumer. Pose
records remain available, but they are not an active prerequisite for current
coarse reachability.

Implemented pre-RA grounding contracts in
`agents/pa/grounding_contracts.py` cover:

- `GroundingProducerDescriptor`, describing one provider's accepted evidence,
  produced records, prerequisites, availability, and estimated cost;
- `TypedContextBinding` and `ProductContextView`, joining compact ABox meaning
  with validated record refs, hashes, status, frames, validity, and source
  details;
- `ResourceAssignmentNeed`, whose current coarse-reach consumer declares
  `RobotFrameLocationRecord` and its required target frame;
- `TypedGroundingContract` schema version 3, preserving the final cited context
  summary, accepted ontology projection, and hash-pinned records; and
- append-only native PA tool audits, `PAClarification` records, and
  `PAContextGroundingCompletion` version 3.

The active system expands `ResourceAssignmentNeed` through producer descriptor
prerequisites and compares that closure with current, accepted, hash-valid
typed bindings. Missing source-produced types are mapped to eligible approved
evidence handles and returned to PA as prompt-only validation feedback. This
feedback and the provisional candidate are not persisted contracts.

`GroundingNextAction`, `GroundingActionAttempt`, `GroundingSession` version 2,
targeted document evidence, pose-linked `ResourceSelectionRecord` version 1,
`TypedGroundingContract` version 2, and completion version 2 describe historical
interactions only. Supported loaders may verify them for read-only recovery;
new production runs do not write or migrate them.

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
