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
configurable limits, evidence-type routing, numbered source serving,
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
The PA UI tests cover the turn
control, live turn counter, compact served-context summaries, full audit
records, production `grounding_unavailable` state, and rejection of an unbacked
completion turn.

Later phases will add focused tests for user clarification replies, formal
handoff, rotation and complete pose estimation, Phase 4.3
assessment, schemas, later agent adapters, validation feedback, revision
behavior, and accepted candidate handoff.
