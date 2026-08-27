# Contexts

Each interaction stores its runtime inputs, retrieved snapshots, messages,
outputs, and validation evidence under its caller-owned root. Product-side PA
and geometry paths exist today; the resource and composition paths labeled
`planned` below are not produced by the current runtime:

```text
contexts/<interaction_identifier>/
├── products/
│   ├── user_requirement/
│   ├── observations/
│   ├── served_references/
│   ├── grounding/
│   └── assembly_plan/                     # planned Phase 5
├── resources/                              # planned RA-owned context
│   └── <exact_RA_identifier>/
│       ├── robot_state/
│       ├── primitive_catalog_snapshot/
│       ├── primitive_program_drafts/
│       ├── primitive_steps/
│       └── validation/
├── composition/                            # planned PA/RA exchange
│   ├── context_bundles/
│   ├── missing_context_batches/
│   └── progress_decisions/
└── interaction_record/
```

The planned `ProductContextView` will aggregate the PA ABox with validated
`TypedContextBinding` summaries. Planned `CompositionContextBundle` versions
will preserve task, ABox, typed-binding, catalog, and selected-resource
fingerprints. Each RA `PrimitiveProgramDraft`, deduplicated
`MissingContextBatch`, PA response, progress or no-progress decision, fully
bound candidate, and validation trace will be append-only and reviewable. A
catalog snapshot records the complete catalog returned by the selected RA and
does not assume a fixed number of primitives. No RA resource subtree,
composition bundle, missing-context batch, or primitive candidate is generated
today.

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

When a controlled future caller supplies one exact preprocessed CAD record and
one segmentation record, Phase 4.2B2A writes
`products/grounding/rgb_d_cad_grounding/correspondence_<number>/correspondence_record.json`
atomically in that same interaction. It records deterministic size rankings and
may preserve one candidate center in its camera optical frame. The status-only
UI never exposes that coordinate. No cross-camera or robot-frame transform,
rotation, complete pose, PA decision, planning, or execution is stored by this
operation.

When a controlled caller supplies that intact size-correspondence record, the
pose step writes
`products/grounding/rgb_d_cad_grounding/pose_<number>/pose_record.json`
atomically. A clear fit stores the camera-from-CAD transform; weak or competing
fits remain rejected or ambiguous. No cross-camera or robot-frame transform,
PA decision, planning, or execution is stored by this operation, and the
status-only UI never exposes its coordinates or rotation.

When a controlled caller supplies one approved extrinsic calibration,
`calibration_<number>/calibration_record.json` stores its exact source and
target frames, rigid transform, validity window, provenance, and payload hash.
The frame-conversion step then writes
`robot_pose_<number>/robot_frame_pose_record.json` atomically. It checks the
calibration against the originating observation timestamp and composes a
robot-frame pose only when the camera-frame pose is accepted. Ambiguous and
rejected poses retain no robot-frame coordinates, and the status-only UI never
exposes an accepted transform.

Generated interaction directories are ignored by Git. Ground-truth evaluation
is excluded from runtime context and belongs under `../evaluations/` so it
cannot leak to PA or RA.
