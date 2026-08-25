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
Phase 2.1 connection, and preserved dual-Gazebo controls. Phase 3.1 tests cover exact
product-requirement preservation, the three valid first `needed_context`
decisions, structured failure recording, exclusive interaction records, and the
absence of shared agent lifecycle, retrieval, capture, UI, RA, CCA, planning, or
execution calls. Phase 3.2 tests cover exact document, CAD, and controlled live
observation serving, preserved `provenance` and retrieval failures, exclusive
records, no fallback evidence, and the absence of PA turns, VLM, UI, grounding,
planning, RA, CCA, forbidden Gazebo-state inputs, or execution. Phase 3.3 tests
cover configurable limits, cumulative evidence, PA-selected ordering, numbered
document/CAD/live serving, clarification, completion, failures, and the absence
of grounding, planning, primitive-catalog, RA, CCA, evaluator, or execution
dependencies. The PA UI tests cover the turn control, live turn counter, compact
served-context summaries, full audit records, and completion boundary.

Later phases will add focused tests for user clarification replies, formal
handoff, grounding, schemas, later agent adapters, validation feedback, revision
behavior, and accepted candidate handoff.
