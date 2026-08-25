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

Generated interaction directories are ignored by Git. Ground-truth evaluation
is excluded from runtime context and belongs under `../evaluations/` so it
cannot leak to PA or RA.
