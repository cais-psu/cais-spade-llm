# Tests

This directory contains coherent Spec2Primitives tests. The scene-only milestone
covers the narrow dual Gazebo adapter, placeholder reply, NIST scene structure,
launch pass-through, installed-world prerequisite, do-not-leak rule, and the
no-`bridge.py`-import boundary.

Current PA production-grounding coverage verifies the native `retrieve` tool,
single `target_feature` proposal validation, the separate semantic-review pass,
and the evidence-gated retry loop. Tests prove
that an early proposal does not write accepted RDF, a descriptor-derived typed
gap re-enters PA with the previous evidence state, and the proposal is committed
only after its active consumer's prerequisite chain is accepted. They also
cover CAD-first, observation-first, document-first, repeated live observation,
unauthorized or stale evidence, target-feature-only CAD correspondence,
ambiguous results, missing calibration, inconsistent frames, unreachable
resources, emergency exhaustion, and synthetic descriptor routing without
product or modality branches.
The same suite verifies zero, one, and multiple state values; multiple paths in
one record; unchanged PA-authored text, names, refs, paths, and evidence; and
rejection of empty statements, duplicate names, unsupported processes,
unauthorized evidence, missing or unaccepted records, invalid JSON Pointers,
empty resolved values, and changed hashes. It verifies that deterministic state-
evidence validation runs before semantic review, ambiguity stops before review
or allocation, and a scaffold-free semantic consistency gap returns to PA before
ontology commit.
It also verifies that PA cannot turn a required-record or approved-evidence
choice into a user clarification while genuine requirement ambiguity remains
eligible for clarification.

Phase 1 tests cover the approved 35-source product inventory and exact-ref
resolver under `../tools/`. Phase 1.1 tests cover fixture/replay RGB-D bundle
validation, lossless storage, reloading, and the absence of live or agent
dependencies. Phase 1.2 tests cover explicit request-only subscriptions,
four-camera live capture, freshness, synchronization, calibration and message
rejections, atomic storage, ROS cleanup, and the absence of agent, UI, detector,
Gazebo-state, and ground-truth dependencies. The scene suite also checks the
current directory boundaries. Phase 2 UI tests cover the PA-only workspace,
Phase 2.1 connection, and preserved dual-Gazebo controls. Phase 3.1 tests cover
exact product-requirement preservation, ABox initialization, authorized
inference-free previews, terminal first-turn states, structured failure
recording, and exclusive interaction records. Phase 3.2 tests
cover exact document, CAD, and controlled live observation serving, preserved
`provenance` and retrieval failures, exclusive
records, no fallback evidence, and the absence of PA turns, VLM, UI, grounding,
planning, RA, forbidden Gazebo-state inputs, or execution. Phase 3.3 tests
inject only the schema-only fixture and controlled provider doubles; they cover
configurable limits, capability-based action discovery, numbered source
serving, cumulative atomic ABox deltas, aligned interpretation and decision
records, alternative source order, duplicate prevention, clarification, and
terminal incomplete states without planning, primitive-catalog, RA, evaluator,
or execution dependencies. Phase
4.1 tests cover ontology-neutral non-stored overview requests, cache hits,
document/model/schema invalidation, atomic persistence, a second registered PDF,
the `--all`/`--context-ref` preparation command, page-level provenance,
assertion-free document deltas, and the separate stage-oriented diagnostic.
Production grounding tests reproduce the original `specification` entity-key
output and prove it cannot mutate the ABox. They also check one schema for
`assemble medium gear`, drilling, welding, and inspection; answer-leak prompt
boundaries; native document, CAD, and observation retrieval; dynamic selection
independent of registration order; synthetic providers; source-revision replay
protection; evidence-gap retries and emergency ceilings; directly-supported-only
late mapping; wrong-domain/range rejection; completion-v6 proposal, review,
opaque presentation, process and physical-state choices, two-state reachability,
endpoint-motion RobotAgent validation, registry/workcell lineage, evidence, and
typed-value hashes; F5
zero-inference startup; and zero document-VLM calls on CAD/RGB-D paths. Phase 4.2A
tests cover exact approved CAD paths and hashes, complete binary STL loading,
millimetre-to-metre conversion, analytic four-camera deprojection, RGB/pixel
association, invalid-depth removal, calibration rejection, atomic typed records,
assertion-free delta merging, injected fresh capture, and controlled
preprocessing failure. Phase 4.2B1 tests cover deterministic
neutral support-plane recording, loose depth-connected regions, uniform
candidate retention, fixed parameters, stable label masks and hashes,
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
The implemented Phase 4.2B2 pose tests cover known asymmetric camera-frame
translations and rotations, same-size shape-fit resolution, candidate and
rotation ambiguity, zero-candidate rejection, complete provenance, tamper
rejection, deterministic atomic persistence, no-overwrite behavior, compact
pose status, and the absence of coordinates or transforms from the UI surface.
Camera-to-robot frame-conversion tests cover rotated-camera composition,
changing camera-frame poses without code changes, calibration and frame
validation, observation-time validity, payload and input-hash tampering,
ambiguous and rejected propagation, deterministic reruns, atomic cleanup,
no-overwrite behavior, and status-only UI output.
Phase 4.3 contract tests cover provider capabilities, typed context bindings,
`TargetFeatureSemanticReview` version 2, proposal version 8, typed-contract and
completion version 6, both state IRIs, presentation lineage, nested evidence and
referenced state-value hashes, two-state reachability, endpoint-motion RobotAgent
validation, native tool audits, and read-only validation of supported historical
proposal, selection, completion, envelope, and session/action contracts. They
also cover exact fixed symbols, PA-only authority, typed-binding
status/frame/freshness validation, rejected obsolete formats, and artifact
tamper rejection. Production grounding tests cover document-only semantic
consumers; dynamically discovered CAD, observation, segmentation,
size-correspondence, calibration, and robot-frame-location paths;
unused-modality exclusion; and derived incomplete diagnostics without turning a
system evidence gap into user clarification. The PA UI tests cover the turn control, live turn counter,
compact served-context summaries, full audit records, configured production
grounding, fail-closed unconfigured state, read-only ontology assertions,
source records, typed bindings, native tool audits, complete/waiting/incomplete
states, and rejection of an unbacked completion turn.
Phase 3.4 tests cover same-interaction answers, repeated questions, new evidence
after a reply, exact history, cancellation, interruption recovery, and rejection
of system-evidence clarification. Phase 3.5 tests cover referenced completion
versions 2, 3, and 4, document and typed-geometry paths, UI readiness, and session,
ontology, contract, clarification, source, and typed-record tampering.

Phase 5.1 tests cover the contract-first selected-RA handoff. A controlled
production-profile fixture proves that a PA-selected `xarm6@localhost` assignment
reaches only that exact RobotAgent before state retrieval; this is fixture
evidence, not a default-selection rule. Separate arbitrary-process/resource and
reversed-presentation tests prove active code does not enumerate `assembly`,
`xarm6`, or `ur5e` and presentation order cannot choose the committed resource.
Changed Phase 4 evidence blocks dispatch, a differently addressed RA rejects the
v3 envelope, and response JID/fingerprint mismatches fail closed. The tests also
cover complete ordered catalogs, invalid or composite catalog entries,
non-finite state, paired append-only revisions, unpaired-history rejection,
failure-only assignment audits, and the absence of binding, primitive-level
validation, and execution artifacts.
The read-only diagnostic coverage checks its transition from
`waiting_for_phase_4` to `ready_for_assignment`, `waiting_for_ra`,
`context_captured`, or `blocked`; exact `xarm6@localhost` assignment display;
latest paired revision counts and refs; full ordered primitive symbols; current
state; and catalog fingerprint. The PA UI test proves a valid Phase 4 completion
surfaces the selected assignment without pretending that an RA response exists.

Phase 5.2A tests prove that the exact selected RobotAgent receives a bounded
structural-authoring request with a reconstructed `target_feature`, no
top-level `task`, and no tools; returns only exact catalog symbols; and may
repeat them or report unsupported. They verify host-generated feature lookup,
unchanged desired-state meaning/evidence, bounded resolved-value projections,
and fail-closed behavior before the RA call for mismatched feature/process,
unpinned or changed records, invalid paths, and empty values. They cover one immutable
`PrimitiveProgramDraft` per state/catalog pair, append-only revision after a
Phase 5.1 restart, pinned evidence preservation, invented-symbol rejection, and
the fail-closed UI action gate. They also verify that the unchanged persisted
draft does not copy the target feature and diagnostics reconstruct the same
input from pinned records.

Later phases will add focused tests for live selected-RA delivery,
`MissingContextBatch`, `CompositionContextBundle`, validation feedback, bound
candidate revision behavior, and accepted candidate handoff.
Composition coverage will exercise multiple catalog cardinalities
without an eight-entry invariant, exact symbol membership, pinned catalog and
bundle fingerprints, ownership routing, draft-derived batch aggregation and
stable deduplication, replay rejection, stale versions, catalog changes,
ambiguity, multiple productive PA/RA batch rounds, and fail-closed repeated or
non-progressing requests. The Phase 5.1 controlled double establishes only the
assignment and snapshot contract, not live SPADE delivery. Predetermined target
geometry records, cross-camera fusion, multi-feature grounding, parameter
binding, grasp/contact/orientation/tolerance/insertion validation, execution,
and observed state updates remain intentionally unimplemented. Endpoint reach,
IK, collision-aware endpoint/path validation, and no-motion assignment commit
are implemented and covered.
