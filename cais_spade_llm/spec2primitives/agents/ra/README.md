# RA

RA means RobotAgent. This directory owns Phase 5.1 selected-RA context handoff and Phase 5.2A structural primitive authoring.

## Phase 4 authority

Current arm assignment uses configured capability plus live MoveIt position planning for all grounded current/destination references. The owned RA adapter checks the exact provisional robot and invokes the plan-only backend. The grounded proposal, live reachability records, selection and completion define current authority. These position checks do not establish assembly outcomes.

## Phase 5.1

`activate_selected_ra_context` revalidates current completion, evidence, evidence, capability and assignment lineage before writing `SelectedRAAssignmentEnvelope`. The envelope pins the exact selected process/resource/states and reports `motion_validation_performed: true` for position planning only.

The injected `RobotAgentCompositionRuntime` must confirm that exact assignment before returning state and the complete primitive-only catalog. The owned UI adapter reuses the selected JID/mode or, under existing simulation readiness conditions, starts only that context-only RobotAgent. It does not launch the full Agent System, substitute another arm or execute primitives.

Paired `RobotStateSnapshot` and `PrimitiveCatalogSnapshot` revisions are append-only. Mismatched JIDs, unpaired history, malformed catalogs or altered evidence fail closed. Restart appends a new pair under the same immutable assignment. Refresh and `read_phase_5_1_diagnostic` only read records.

Incompatible saved records cannot activate an RA or bypass the gate through recovered snapshots. They remain untouched and require “Start a fresh interaction”; there are no compatibility readers.

## Phase 5.2A

`author_primitive_program_draft` requires current completion and the latest valid state/catalog pair. It reconstructs the grounded target, relationship state memberships and bounded typed state values. The input also contains the exact assignment, post-assignment ontology projection, RA state and complete catalog.

The same selected RA chooses/orders exact catalog symbols, may repeat symbols, or returns unsupported. The request has no tools, bindings or top-level `task` section. Host checks verify lineage, identities, values and symbol membership; they do not create or repair the sequence.

One immutable `PrimitiveProgramDraft` is stored per context pair. It pins authorities without copying the target feature. Diagnostics reconstruct the same transient input. Historical completions cannot create new drafts.

## Remaining work

Live SPADE delivery, `MissingContextBatch`, `CompositionContextBundle`, parameter binding, fully bound `primitive_steps`, executable validation, execution and observed outcomes remain future work. Phase 4's new relationship collection does not implement those steps.

See [handoff tests](../../tests/test_ra_context_handoff.py) and [bias validation](../../BIAS_VALIDATION.md). Shared RobotAgent and `SystemBridge` interfaces remain unchanged.
