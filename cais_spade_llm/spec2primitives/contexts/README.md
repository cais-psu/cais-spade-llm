# Contexts

Each future interaction stores its runtime inputs, retrieved snapshots,
messages, outputs, and validation evidence under:

```text
contexts/<interaction_identifier>/
├── products/
│   ├── user_requirement/
│   ├── observations/
│   ├── served_references/
│   ├── grounding/
│   └── assembly_plan/
├── resources/
│   └── <exact_RA_identifier>/
│       ├── robot_state/
│       ├── primitive_catalog_snapshot/
│       ├── primitive_steps/
│       └── validation/
└── interaction_record/
```

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
CAD-correspondence and camera-frame location states, `pose: not_evaluated`, and
failure information; it is not a context-completion or assembly-readiness
record.

When a controlled future caller supplies one exact preprocessed CAD record and
one segmentation record, Phase 4.2B2A writes
`products/grounding/rgb_d_cad_grounding/correspondence_<number>/correspondence_record.json`
atomically in that same interaction. It records deterministic size rankings and
may preserve one candidate center in its camera optical frame. The status-only
UI never exposes that coordinate. No cross-camera or robot-frame transform,
rotation, complete pose, PA decision, planning, or execution is stored by this
operation.

Generated interaction directories are ignored by Git. Ground-truth evaluation
is excluded from runtime context and belongs under `../evaluations/` so it
cannot leak to PA or RA.
