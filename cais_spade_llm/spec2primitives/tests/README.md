# Tests

This directory contains coherent Spec2Primitives tests. The scene-only milestone
covers the narrow dual Gazebo adapter, placeholder reply, NIST scene structure,
launch pass-through, installed-world prerequisite, do-not-leak rule, and the
no-`bridge.py`-import boundary.

Current PA production-grounding coverage verifies the native `retrieve` tool,
transient proposal validation, and the evidence-gated retry loop. Tests prove
that an early proposal does not write accepted RDF, a descriptor-derived typed
gap re-enters PA with the previous evidence state, and the proposal is committed
only after its active consumer's prerequisite chain is accepted. They also
cover CAD-first, observation-first, document-first, repeated live observation,
unauthorized or stale evidence, target-feature-only CAD correspondence,
ambiguous results, missing calibration, inconsistent frames, unreachable
resources, emergency exhaustion, and synthetic descriptor routing without
product or modality branches.
The same suite verifies that a proposal omitting a required feature relation is
not committed, receives deterministic correction feedback, and can be replaced
by a valid proposal within the bounded PA investigation.
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
late mapping; wrong-domain/range rejection; completion-v3 hashes; F5
zero-inference startup; and zero document-VLM calls on CAD/RGB-D paths. Phase 4.2A
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
typed-contract version 3, completion version 3, native tool audits, and
read-only validation of supported historical session/action contracts. They
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
versions 2 and 3, document and typed-geometry paths, UI readiness, and session,
ontology, contract, clarification, source, and typed-record tampering.

Phase 5.1 tests cover the contract-first selected-RA handoff. They prove that
the exact `assemble medium gear` assignment reaches `xarm6@localhost` before
state retrieval, changed Phase 4 evidence blocks dispatch, a differently
addressed RA rejects the envelope, and response JID/fingerprint mismatches fail
closed. They also cover complete ordered catalogs, invalid or composite catalog
entries, non-finite state, paired append-only revisions, unpaired-history
rejection, failure-only assignment audits, and the absence of all Phase 5.2,
validation, and execution artifacts.
The read-only diagnostic coverage checks its transition from
`waiting_for_phase_4` to `ready_for_assignment`, `waiting_for_ra`,
`context_captured`, or `blocked`; exact `xarm6@localhost` assignment display;
latest paired revision counts and refs; full ordered primitive symbols; current
state; and catalog fingerprint. The PA UI test proves a valid Phase 4 completion
surfaces the selected assignment without pretending that an RA response exists.

Later phases will add focused tests for live selected-RA delivery,
`PrimitiveProgramDraft`, `MissingContextBatch`, `CompositionContextBundle`,
validation feedback, revision behavior, and accepted candidate handoff.
Composition coverage will exercise multiple catalog cardinalities
without an eight-entry invariant, exact symbol membership, pinned catalog and
bundle fingerprints, ownership routing, draft-derived batch aggregation and
stable deduplication, replay rejection, stale versions, catalog changes,
ambiguity, multiple productive PA/RA batch rounds, and fail-closed repeated or
non-progressing requests. The Phase 5.1 controlled double establishes only the
assignment and snapshot contract, not a live or production RA composition path.
