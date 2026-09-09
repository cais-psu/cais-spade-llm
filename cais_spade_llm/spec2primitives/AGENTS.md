# Spec2Primitives instructions

## Scope

- This directory owns the ICRA 2027 case study
  `Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly`.
- Put all new Spec2Primitives implementation, schemas, cases, tests, documentation,
  UI implementation, and generated runtime records under this directory.
- Search this directory by default. Read-only inspection and search outside this
  directory are permitted without asking when they are needed for a
  Spec2Primitives task.
- Ask the user before modifying any file outside this directory. Read-only
  inspection outside this directory never requires confirmation.

## Incremental workflow and handoff

- Take one small, explicitly authorized step at a time so the user can recognize
  each change.
- Do not advance into a later implementation-plan phase or combine adjacent
  steps unless the user explicitly requests it.
- Before editing, provide the code-reading preview defined below, including the
  exact files and behavior that the step will change.
- After editing, provide the code-reading handoff defined below, including
  clickable links to every changed file, what changed, what intentionally
  remained unchanged, and the verification commands and results.
- Preserve unrelated local changes and call them out separately from the current
  step.

## Scoped refactoring

- You may automatically refactor only functions directly touched by the
  authorized change or nearby duplicated logic when the refactor is small,
  behavior-preserving, and makes that code path easier to read.
- Before editing, announce the exact refactor and explain why it helps the
  current change.
- Preserve public interfaces, fixed symbols, runtime authority, persistence
  formats, safety checks, fail-closed behavior, and observable behavior.
- Treat module extraction, broad renaming, public-interface changes, and
  unrelated cleanup as separate tasks that require user authorization.
- In the handoff, report feature or bug-fix changes separately from refactoring
  changes. If no refactoring was performed, say so explicitly.

## Error-driven replacement and legacy removal

- When an error requires changing an authorized plan or implementation, remove
  the superseded code path that the replacement makes invalid or nonfunctional.
- Remove the obsolete implementation together with its unused helpers,
  fallback branches, deprecated aliases, compatibility wrappers, tests, and
  documentation when they exist only to support the failed behavior.
- Do not retain two active implementations of the same behavior unless the user
  explicitly requires backward compatibility or a staged migration.
- Before removing legacy code, confirm that the replacement covers the current
  runtime call path and that no authorized persistence or public-interface
  requirement still depends on it.
- If complete removal would exceed the authorized scope, change a public
  interface, or require a data migration, stop at that boundary, report the
  remaining legacy dependency, and ask the user before expanding the change.

## Simplicity and minimum contracts

- Implement the user-approved plain-language process directly.
- Do not add an intermediate field, state transition, duplicate representation,
  or compatibility layer unless an active runtime consumer requires it.
- The model returns only the next semantic action or the final ontology/context
  proposal. The host derives IDs, provider metadata, revisions, hashes, and
  persistence fields.
- Reuse retrieved typed records as evidence state instead of copying them into
  model-authored reasoning records.
- Before adding a field, identify its runtime consumer and deterministic
  validator. If neither exists, do not add the field.
- When two designs provide the same behavior, choose fewer model-controlled
  fields and fewer persisted objects.

## Before-edit code-reading preview

Before every implementation change, provide a concise preview that:

- states the plain-language objective and identifies the exact entry-point
  function;
- shows the ordered call path containing only the relevant functions;
- explains each function's input, output, and responsibility in one sentence;
- identifies the nearest relevant regression test and runtime record, trace, or
  UI state;
- states where the path stops and what is intentionally out of scope.

For instruction-only or documentation-only changes with no runtime entry point,
say that directly and show the relevant document-reading path instead of
inventing a runtime flow.

## After-change code-reading handoff

Keep the reading guide short enough to follow in approximately two to five
minutes. End every Spec2Primitives result with these headings in this order:

1. **Outcome**
2. **Process flow**
3. **Read these locations in order**
4. **Read this test**
5. **Runtime evidence**
6. **You can ignore**
7. **Refactoring performed**
8. **Verification and intentionally unchanged behavior**

The handoff must:

- link to the exact functions in recommended reading order and explain what to
  inspect at each location without asking the user to read the entire file;
- include a compact relevant flow such as
  `UI -> orchestration -> producer -> validation -> trace`;
- link the most useful regression test and explain what its assertion proves;
- point to the relevant UI state, trace, or persisted record when applicable;
- include an explicit **You can ignore** list naming unrelated modules.

When no function, regression test, or runtime evidence applies, state that
explicitly under the corresponding heading.

## Comments and rationale

- Add rationale comments for non-obvious API restrictions, safety checks,
  runtime authority boundaries, fail-closed behavior, provenance, and
  validation-stage decisions.
- Place each rationale comment at the decision or boundary it explains and
  describe why the constraint exists.
- Continue to avoid comments that merely restate the next line of code.

## Short, proportional Codex verification

- Codex must default to the smallest verification set that directly exercises
  the changed behavior. Do not automatically run the complete Spec2Primitives
  suite or repository-wide checks after every change.
- For a small Python change, normally run one focused selection from the nearest
  existing test file, a syntax or static check limited to the touched files, and
  `git diff --check`.
- For an instruction-only or documentation-only change, inspect the Markdown
  diff and run `git diff --check`; do not run Python tests, Ruff, compileall, or
  runtime checks.
- Run the complete Spec2Primitives suite, repository-wide compileall, broad Ruff,
  `poetry check`, UI rendering, external API calls, Gazebo, or hardware checks
  only when the changed boundary requires them, the user explicitly requests
  them, or a milestone handoff requires them. State the reason before starting.
- Extend the nearest existing test file when a behavior change needs regression
  coverage. Do not create a new test file unless the change introduces a
  coherent new subsystem with no suitable existing test location or the user
  explicitly requests one.
- Do not create repository-local throwaway tests, verification scripts, reports,
  snapshots, logs, or captured artifacts. Use temporary locations outside the
  repository when a disposable reproduction is necessary.
- Do not broaden verification merely because a focused check fails. Diagnose the
  focused failure first, and report any broader verification as not run.
- Keep routine verification short. Before starting a check likely to take more
  than approximately five minutes, explain why it is necessary and obtain user
  authorization.
- In the handoff, separate checks actually run from broader checks intentionally
  not run. Never imply that focused checks prove unrelated runtime behavior.

## MUST: Do not leak the answer

- Allowed recognition inputs are only the user requirement, approved NIST
  documents, approved candidate CAD files, RGB, depth, and camera calibration.
- Forbidden recognition inputs are Gazebo model names, Gazebo entity names,
  world or SDF contents, spawn manifests, configured spawn poses,
  `/gazebo/model_states`, `/get_entity_state`, current detector responses, and
  evaluator labels.
- Candidate CAD filenames and document part names are allowed because they are
  part of the supplied runtime corpus. Recognition must still determine which
  observed object matches which candidate and where it belongs.
- Ground truth may be read only by a separate evaluator after the prediction is
  finalized.
- Recognition code must not import, invoke, or share runtime objects with the
  ground-truth evaluator.
- Any experiment that violates this boundary is invalid and must not be
  reported.

## Shared runtime boundaries

- Do not inspect or modify `cais_spade_llm/ui/bridge.py` unless the user
  explicitly requests work on that file.
- Do not copy ProductAgent or RobotAgent into this directory.
- Treat ProductAgent and RobotAgent as shared, read-only runtime authorities.
- Keep the ProductAgent connection under `spec2primitives/agents/pa/` and the
  RobotAgent connection under `spec2primitives/agents/ra/`.
- Do not create folders for unused shared-agent subsystems; they are outside the
  current Spec2Primitives roadmap.
- When a future adapter requires an existing public interface, inspect the exact
  external interface without asking. Do not modify it without explicit
  permission.

## Current non-executing runtime boundary

- The UI may start, stop, and read status for the no-hardware
  `gazebo_dual_spec2primitives` process only through
  `spec2primitives/adapters/dual_gazebo.py`.
- Do not import `SystemBridge` or issue direct ROS2 shell commands from this
  package. The application passes a runtime object that satisfies the narrow
  Spec2Primitives adapter protocol.
- Do not add hardware launch or shared-agent changes. Preserve the owned
  exact-selected-RA context-only adapter and its readiness gates. The explicitly
  authorized Run in Gazebo action may execute an unchanged saved validated
  program through the owned simulation adapter after fresh checks.
- Keep ProductAgent access behind the Spec2Primitives-owned
  `ProductAgentContextRuntime` composition boundary. Do not start its lifecycle
  or expose other shared-agent operations.
- Keep User ↔ ProductAgent interaction non-executing. Phase 5.1 may contact only
  the selected RA after current completion and evidence validation. Primitive
  composition authors one program with available parameters; omitted required
  parameters remain unbound. Strict selected numerical calculations and private
  motion validation are permitted; composition never executes robot primitives.
- Preserve complete runtime contracts in snapshots. Project grasp/release formal
  conditions/effects to `held_part` only in composition inputs. Omit `model_name`
  from composition parameters, nested schemas, outputs and custody effects;
  retain it in authoritative runtime contracts for the separate execution adapter.
  Reject new submissions containing that execution-only argument. Filter recovery
  examples/decompositions from initial state and every model-facing record read.
- Preserve nested schemas and distinguish signature-required arguments from
  grounding requirements. Binding diagnostics inspect supplied sources without changing the proposal. The
  owned deterministic binder fills only checked PA answers at requested missing or
  incompatible paths in a separate PrimitiveProgramBinding. Only the strict
  calculation adapter evaluates RA-selected helper calls with grounded inputs.
- Read each saved attempt against the composition catalog in its own hash-checked
  request. Preserve old programs and reports; use the current projection only for
  new attempts. Historical catalogs never become new composition inputs.
- Capture configured planning frame and controlled-link/TCP names without
  treating configuration as measured feedback. Do not infer `model_name` from
  product labels or apply an automatic world-to-base conversion.
- Phase 4 supplies observed product evidence and desired relationships. Phase 5
  owns bounded supplemental evidence, measured robot context, strict selected
  target calculations and isolated Cartesian validation with execution disabled.
  RA alone chooses and revises primitive steps. PA input resolution and parameter
  binding are deterministic, using scoped SPADE messaging and pinned answers.
  Helper waypoints prescribe no order.
  Preserve Phase 4 ontology/views and historical programs. Follow
  `VALIDATION_AND_REVISION.md` for implemented scope and subsequent physical work,
  and `COMPOSITION_EVALUATION.md` for experiments and claim limits.
- Current Phase 4 implements PA-owned evidence investigation, deterministic validation,
  pairwise assembly relationships and required MoveIt position planning for arm
  assignment in simulation. It does not establish grasping or assembly outcome.
- Each owned record type has one current format without format-version markers or
  compatibility readers. Keep saved interactions untouched; incompatible records
  require “Start a fresh interaction” and cannot authorize RA work.
- Use `BIAS_VALIDATION.md` for audits and live counterfactual experiments. Offline
  fixtures establish contracts only; never claim they prove no bias.
- Keep `NIST_assembly_instructions.pdf` in `references/products/`. Reference the
  existing NIST STL files by repository path; do not copy the STL files into
  this directory.
- Keep runtime observations, retrieved snapshots, messages, plans,
  `primitive_steps`, validation traces, and execution logs under `contexts/`.
- Keep ground truth and post-prediction evaluation under `evaluations/`, where
  PA, RA, retrieval tools, and recognition code cannot access them.
- Preserve `target_feature`, target pose, insertion axis, tolerances,
  `primitive_steps`, and all other supplied terms exactly as written.
- Keep Run in Gazebo instance binding and command records under the interaction's
  `execution/` directory. Only that post-composition adapter may read simulator
  fixture definitions and live entity state to bind an already accepted CAD
  instance. Exclude those records from all PA/RA evidence readers. This does not
  authorize simulator input to recognition or execution during composition.
