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
- Before editing, state the exact files and behavior that the step will change.
- After editing, show clickable links to every changed file, summarize exactly
  what changed and what intentionally remained unchanged, and report the
  verification commands and results.
- Preserve unrelated local changes and call them out separately from the current
  step.

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
- Put the future ProductAgent connection under `spec2primitives/agents/pa/` and the
  future RobotAgent connection under `spec2primitives/agents/ra/`.
- Do not create ResourceAgent or CCA folders; they are outside the current
  Spec2Primitives roadmap.
- When a future adapter requires an existing public interface, inspect the exact
  external interface without asking. Do not modify it without explicit
  permission.

## Scene-only milestone boundary

- The UI may start, stop, and read status for the no-hardware
  `gazebo_dual_spec2primitives` process only through
  `spec2primitives/adapters/dual_gazebo.py`.
- Do not import `SystemBridge` or issue direct ROS2 shell commands from this
  package. The application passes a runtime object that satisfies the narrow
  Spec2Primitives adapter protocol.
- Do not add hardware launch, ResourceAgent, CCA, RobotAgent, or robot execution.
- Keep ProductAgent access behind the Spec2Primitives-owned
  `ProductAgentContextRuntime` composition boundary. Do not start its lifecycle
  or expose other shared-agent operations.
- Keep the User ↔ ProductAgent interaction non-executing. RA communication
  remains unavailable until its separately authorized phase.
- `table_spec2primitives.world` may display the approved NIST CAD corpus, but no
  recognition, VLM, planning, insertion physics, or robot execution is
  implemented in this milestone.
- Keep `NIST_assembly_instructions.pdf` in `references/products/`. Reference the
  existing NIST STL files by repository path; do not copy the STL files into
  this directory.
- Keep runtime observations, retrieved snapshots, messages, plans,
  `primitive_steps`, validation traces, and execution logs under `contexts/`.
- Keep ground truth and post-prediction evaluation under `evaluations/`, where
  PA, RA, retrieval tools, and recognition code cannot access them.
- Preserve `target_feature`, target pose, insertion axis, tolerances,
  `primitive_steps`, and all other supplied terms exactly as written.
