# Contexts

Each interaction stores its runtime inputs, retrieved snapshots, messages,
outputs, and validation evidence under its caller-owned root. Product-side PA,
geometry, Phase 5.1 assignment/state/catalog, and Phase 5.2A structural-draft
paths exist today. Later composition paths remain planned:

```text
contexts/<interaction_identifier>/
├── products/
│   ├── user_requirement/
│   ├── observations/
│   ├── served_references/
│   ├── grounding/
│   │   ├── document_evidence/overview_<number>.json
│   │   ├── document_evidence/evidence_<number>.json  # historical recovery only
│   │   ├── session/revision_<number>.json            # historical recovery only
│   │   ├── ontology_grounding/proposal_<number>.json
│   │   ├── product_context/view_<number>.json
│   │   ├── completion/typed_grounding_contract_v3.json
│   │   ├── completion/pa_context_grounding_completion_v3.json
│   │   └── <typed producer records>
├── resources/                              # Phase 5.1 RA-owned snapshots
│   └── <exact_RA_identifier>/
│       ├── robot_state/
│       ├── primitive_catalog_snapshot/
│       ├── primitive_steps/                # planned
│       └── validation/                     # planned
├── composition/
│   ├── selected_ra_assignments/            # Phase 5.1
│   ├── primitive_program_drafts/            # Phase 5.2A
│   ├── context_bundles/                    # planned
│   ├── missing_context_batches/            # planned
│   └── progress_decisions/                 # planned
└── interaction_record/
    ├── clarification_<turn>.json
    └── context_completion_0001.json
```

`contexts/source_cache/document/<source_sha256>/<cache_fingerprint>/` is a
generated, cross-interaction overview cache outside individual interaction
roots. The fingerprint includes the document bytes, configured VLM behavior,
and overview-schema version. F5 reads this index without inference. Interaction
records snapshot the validated overview and never treat a cache file as an ABox
assertion.

New runs keep native PA turns and tool audits rather than an active
`GroundingSession`. A proposal candidate and its validation gap remain
transient while source evidence and failures stay audited. `ProductContextView`
is written only from the accepted ABox and typed-record view after the evidence
gate passes; it is not PA reasoning state. Clarification answers or
cancellation and version-3 completion records are append-only. Supported older
session and completion records remain read-only recovery inputs. Planned
`CompositionContextBundle` versions
will preserve task, ABox, typed-binding, catalog, and selected-resource
fingerprints. Each deduplicated `MissingContextBatch`, PA response, progress or
no-progress decision, fully bound candidate, and validation trace will be
append-only and reviewable. Phase 5.1 records the minimum selected-RA assignment
and the complete catalog returned by its injected runtime without assuming a
fixed number of primitives. Phase 5.2A appends one RA-authored structural
`PrimitiveProgramDraft` per captured pair. It produces no composition bundle,
missing-context batch, bound candidate, validation trace, or execution record.

Product RGB-D observation bundles use an `observation_ref` and retain the
existing `manifest.json`, lossless RGB PNG, and original metric `float32` depth
`.npy` contract. Every record remains labeled `fixture`, `replay`, or `live`.
Phase 1.1 writes and tests `fixture` and `replay`. Phase 1.2 writes `live` only
after an explicit `capture_gazebo_observation(...)` request. It does not capture
in the background or automatically attach observations to PA or RA. The four
`<camera>_rgb.png` files in a successful bundle are directly inspectable.

When the Phase 4.2B1 automatic observation entrypoint is invoked, it creates a
unique `rgbd_segmentation_<identifier>/` interaction and performs capture,
Phase 4.2A observation preprocessing, and minimal segmentation without operator
parameters. Its geometry products contain the colored point-cloud record, a
compact segmentation record, and four camera-local label masks. The contexts
root also holds one atomically replaced `rgbd_segmentation_status.json` for the
read-only UI card. It contains only processing status, candidate counts,
CAD-correspondence, camera-frame location, pose, and robot-frame conversion
states plus failure information; it is not a context-completion or
assembly-readiness record.

When a controlled caller supplies one exact preprocessed CAD record and one
segmentation record, the implemented Phase 4.2B2A path writes
`products/grounding/rgb_d_cad_grounding/correspondence_<number>/correspondence_record.json`
atomically in that same interaction. It records deterministic size rankings and
may preserve one candidate center in its camera optical frame. The status-only
UI never exposes that coordinate. No cross-camera or robot-frame transform,
rotation, complete pose, PA decision, planning, or execution is stored by this
operation.

When a controlled caller supplies that intact size-correspondence record, the
pose step writes
`products/grounding/rgb_d_cad_grounding/pose_<number>/pose_record.json`
atomically. A clear asymmetric fit stores the camera-from-CAD transform. When
qualified symmetric rotations agree on the transformed physical CAD centroid,
the version 2 record retains `location: available`, `pose: ambiguous`, and only
`CAD_centroid_translation_m`; it does not reinterpret the off-center STL origin
as the object location or fabricate a rotation. Genuinely different centroids
remain `location: ambiguous`. No cross-camera or robot-frame transform, PA
decision, planning, or execution is stored by this operation, and the
status-only UI never exposes its coordinates or rotation.

The simulation-only UI runtime loads the bundled, versioned
`config/gazebo_camera_to_world_calibration.json` by default. Its four transforms
were composed once from the fixed camera poses in the hash-pinned
`table_spec2primitives.world` and the ROS optical-frame convention authorized
for this Gazebo case. Runtime recognition never reads the world file and
continues to select the active exact frame dynamically: `cam_mk3_link`,
`cam_mk4_1_link`, `cam_mk4_2_link`, or `cam_assembly_link`. A deployment may
replace the bundled simulation manifest through
`SPEC2PRIMITIVES_CAMERA_TO_WORLD_CALIBRATION_PATH`. The PA panel displays
calibration readiness and any exact actionable configuration failure.

When a controlled caller supplies the matching approved extrinsic calibration,
`calibration_<number>/calibration_record.json` stores its exact source and
target frames, rigid transform, validity window, source details, and payload hash.
The active frame-conversion step writes a version-1
`RobotFrameLocationRecord`, checking calibration against the originating
observation timestamp and translating the accepted camera-frame centroid into
the candidate resources' required frame. Missing, invalid, stale, or
selected-frame-missing calibration leaves the typed prerequisite unsatisfied,
so the provisional semantic proposal is not committed. A separate pose record
is written only when an orientation-sensitive consumer requires one. Rejected
or ambiguous locations retain no robot-frame coordinate, and the status-only UI
never exposes a transform.

Generated interaction directories are ignored by Git. Ground-truth evaluation
is excluded from runtime context and belongs under `../evaluations/` so it
cannot leak to PA or RA.
