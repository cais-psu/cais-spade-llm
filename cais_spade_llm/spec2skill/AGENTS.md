# Spec2Skill instructions

## Scope

- This directory owns the ICRA 2027 case study
  `Spec2Skill: A Multi-Agent Framework for Primitive Composition in Robotic Assembly`.
- Put all new Spec2Skill implementation, schemas, cases, tests, documentation,
  UI implementation, and generated artifacts under this directory.
- Search this directory by default. Do not run repository-wide searches for a
  Spec2Skill task.
- Ask the user before inspecting or modifying a file outside this directory.

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
- Do not copy ProductAgent, ResourceAgent, CCA, or RobotAgent into this directory.
- Treat ProductAgent, ResourceAgent, CCA, and RobotAgent as shared, read-only
  runtime authorities.
- Put future connections to those authorities behind modules owned by
  `spec2skill/adapters/`.
- When a future adapter requires an existing public interface, ask before opening
  the exact external interface file. Inspect only that file and do not modify it
  without explicit permission.

## Scene-only milestone boundary

- The UI may start, stop, and read status for the no-hardware
  `gazebo_dual_spec2skill` process only through
  `spec2skill/adapters/dual_gazebo.py`.
- Do not import `SystemBridge` or issue direct ROS2 shell commands from this
  package. The application passes a runtime object that satisfies the narrow
  Spec2Skill adapter protocol.
- Do not add hardware launch, agent calls, ProductAgent, ResourceAgent, CCA,
  RobotAgent, or robot execution.
- Keep the User Interaction chat local and non-executing until a later task
  explicitly connects the PA/RA pipeline.
- `table_spec2skill.world` may display the approved NIST CAD corpus, but no
  recognition, VLM, planning, insertion physics, or robot execution is
  implemented in this milestone.
- Keep `NIST_assembly_instructions.pdf` in `references/`. Reference the existing
  NIST STL files by repository path; do not copy the STL files into this
  directory.
- Keep generated experiment results, validation traces, execution logs, and
  evaluation outputs under `artifacts/`.
- Preserve `target_feature`, target pose, insertion axis, tolerances,
  `primitive_steps`, and all other supplied terms exactly as written.
