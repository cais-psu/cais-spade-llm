# Tests

This directory contains coherent Spec2Primitives tests. The scene-only milestone
covers the narrow dual Gazebo adapter, placeholder reply, NIST scene structure,
launch pass-through, installed-world prerequisite, do-not-leak rule, and the
no-`bridge.py`-import boundary.

Phase 1 tests cover the approved 35-source product inventory and exact-ref
resolver under `../tools/`. Phase 1.1 tests cover fixture/replay RGB-D bundle
validation, lossless storage, reloading, and the absence of live or agent
dependencies. Phase 1.2 tests cover explicit request-only subscriptions,
four-camera live capture, freshness, synchronization, calibration and message
rejections, atomic storage, ROS cleanup, and the absence of agent, UI, detector,
Gazebo-state, and ground-truth dependencies. The scene suite also checks the
current directory boundaries. Phase 2 UI tests cover the PA-only workspace,
Phase 2.1 connection, and preserved dual-Gazebo controls. Phase 3.1 tests cover
exact product-requirement preservation, TBox/ABox initialization before PA,
document or live-observation first requests, first-turn clarification rejection,
structured failure recording, and exclusive interaction records. Phase 3.2 tests
cover exact document, CAD, and controlled live observation serving, preserved
`provenance` and retrieval failures, exclusive
records, no fallback evidence, and the absence of PA turns, VLM, UI, grounding,
planning, RA, forbidden Gazebo-state inputs, or execution. Phase 3.3 tests
inject only the schema-only fixture and controlled producer doubles; they cover
configurable limits, output-capable producer-descriptor routing, numbered source serving,
cumulative atomic ABox deltas, aligned interpretation and decision records,
alternative source order, duplicate prevention, limits, clarification, and
completion only from persisted assessment, without concrete Phase 4 producer,
planning, primitive-catalog, RA, evaluator, or execution dependencies. Phase
4.1 tests cover strict model configuration, a six-page non-stored OpenAI request,
offline vision doubles, page-level provenance, compiled deltas, atomic ABox
acceptance and rejection, and the separate fail-closed diagnostic. Phase 4.2A
tests cover exact approved CAD paths and hashes, complete binary STL loading,
millimetre-to-metre conversion, analytic four-camera deprojection, RGB/pixel
association, invalid-depth removal, calibration rejection, atomic typed records,
assertion-free delta merging, injected fresh capture, and controlled
preprocessing failure. Phase 4.2B1 tests cover deterministic
source-plane removal, loose depth-connected regions, assembly-plate retention,
fixed camera roles and parameters, filtering, stable label masks and hashes,
zero-candidate unresolved records, tamper rejection, atomic cleanup and
no-overwrite behavior, automatic capture → preprocessing → segmentation,
unique roots, compact status, and the absence of operator processing controls.
Phase 4.2B2A tests use analytic 22 mm, 42 mm, and 62 mm candidates to verify
that `Gear_Medium.STL` uniquely selects the 42 mm candidate and reports its
camera optical frame and median center. They also cover accepted measurement
noise, duplicate-size ambiguity, no-match and zero-candidate rejection,
image-boundary partial visibility, deterministic ranking, record hashes,
tamper rejection, atomic cleanup, no-overwrite persistence, compact status, and
the absence of coordinates or controls from the UI surface.
The simple remaining Phase 4.2B2 pose tests cover known asymmetric camera-frame
translations and rotations, same-size shape-fit resolution, candidate and
rotation ambiguity, zero-candidate rejection, complete provenance, tamper
rejection, deterministic atomic persistence, no-overwrite behavior, compact
pose status, and the absence of coordinates or transforms from the UI surface.
Camera-to-robot frame-conversion tests cover rotated-camera composition,
changing camera-frame poses without code changes, calibration and frame
validation, observation-time validity, payload and input-hash tampering,
ambiguous and rejected propagation, deterministic reruns, atomic cleanup,
no-overwrite behavior, and status-only UI output.
Phase 4.3 contract tests cover deterministic need computation, exact fixed
symbols, authorized descriptor selection, PA-only authority, typed-binding
status/frame/freshness validation, and artifact tamper rejection. Production
grounding tests cover document-only completion; dynamically ordered CAD,
observation, segmentation, size-correspondence, and camera-frame-pose paths;
unused-modality exclusion; and ambiguous pose failure without user
clarification. The PA UI tests cover the turn control, live turn counter,
compact served-context summaries, full audit records, configured production
grounding, fail-closed unconfigured state, read-only ontology assertions,
provenance, typed bindings, unresolved needs, producer decisions, and rejection
of an unbacked completion turn.
Phase 3.4 tests cover same-interaction answers, repeated questions, new evidence
after a reply, exact history, cancellation, interruption recovery, and rejection
of system-evidence clarification. Phase 3.5 tests cover referenced completion,
empty current draft needs, document and typed-geometry paths, UI readiness, and
completion, draft, clarification, and typed-record tampering.

Later phases will add focused tests for the Phase 5 task handoff,
cross-camera transformation, robot-frame grounding after resource
selection, later agent adapters, validation feedback, revision behavior, and
accepted candidate handoff. Composition coverage will exercise multiple catalog cardinalities
without an eight-entry invariant, exact symbol membership, pinned catalog and
bundle fingerprints, ownership routing, draft-derived batch aggregation and
stable deduplication, replay rejection, stale versions, catalog changes,
ambiguity, multiple productive PA/RA batch rounds, and fail-closed repeated or
non-progressing requests. Until those phases are implemented, no test fixture
or controlled double establishes a production RA composition path.
