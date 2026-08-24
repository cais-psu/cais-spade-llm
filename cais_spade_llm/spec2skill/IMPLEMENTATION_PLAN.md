# Spec2Skill Implementation Plan

Phase 0 and the Phase 0.1 operator shell are implemented. Every later phase is
future work and requires a separate, explicitly scoped implementation request.
This plan does not authorize an end-to-end implementation.

`PA` refers to ProductAgent and `RA` refers to RobotAgent throughout this plan.
Only PA and RA participate in the current Spec2Skill roadmap.

## MUST: Do not leak the answer

Allowed recognition inputs are only the user requirement, approved NIST
documents, approved candidate CAD files, RGB, depth, and camera calibration.
Forbidden recognition inputs are Gazebo model names, Gazebo entity names, world
or SDF contents, spawn manifests, configured spawn poses,
`/gazebo/model_states`, `/get_entity_state`, current detector responses, and
evaluator labels.

Candidate CAD filenames and document part names are allowed because they belong
to the supplied runtime corpus. The system must still determine which observed
object matches which candidate and where it belongs. Ground truth may be read
only by a separate evaluator after the prediction is finalized. Recognition
code must not import, invoke, or share runtime objects with the ground-truth
evaluator. Any experiment that violates this boundary is invalid and must not
be reported.

## Phase 0: isolated skeleton - implemented

- Establish the Spec2Skill package and directory boundary.
- Record the ICRA 2027 title, starting case, and proposed workflow.
- Keep `NIST_assembly_instructions.pdf` under `references/` and reference the
  existing NIST CAD files in place.
- Provide a static `/spec2skill` NiceGUI starting page.
- Keep all existing runtime authorities unchanged.

## Phase 0.1: NIST scene-only operator shell - implemented

- Start, stop, and show status for the no-hardware
  `gazebo_dual_spec2skill` simulation through a narrow Spec2Skill adapter
  protocol.
- Launch the dedicated `table_spec2skill.world` through the package-local
  `world_file` argument while leaving the default `table.world` scene unchanged.
- Show the actual NIST plate, KET pins, RGOCG pins, static `Gear_Plate` and three
  `Gear_Shaft` fixtures, and loose gears while preserving the four RGB-D cameras
  and forcing `run_perception:=false`.
- Use the plate mesh collision and exact-size box or cylinder pin collisions
  without making a physics-accuracy claim.
- Keep ROS2 process ownership and prerequisite validation with the existing UI
  runtime.
- Provide a local placeholder chat that clearly states no plan or robot action
  was executed.
- Keep PA, RA, and hardware disconnected.
- Do not implement recognition, VLM, detector calls, planning, insertion, or
  robot execution in this milestone.

## Phase 1: controlled Medium Gear case

This phase requires a separate implementation request.

- Define the controlled case input as
  `product requirement: assemble Medium Gear`.
- Inventory the allowed NIST references and candidate CAD files for the case.
- Define the permitted RGB, depth, and camera calibration observations.
- Require provenance for every supplied reference and observation.
- Record and enforce the forbidden recognition inputs.
- Do not implement retrieval or call PA or RA.

## Phase 2: cases and contracts

This phase requires a separate implementation request.

- Define reviewable contracts for PA and RA interaction records, PA retrieved
  `context ref`, PA grounding, PA clarification questions, user replies, the
  assembly plan, PA-to-RA communication, RA retrieved `context ref`, fresh
  robot state, resource-owned primitive catalog, `primitive_steps`, and
  robot-local validation feedback.
- Add controlled fixtures and contract validation tests.
- Keep the contracts local and non-executing; do not call PA or RA.

## Phase 3: offline PA-RA UI prototype

This phase requires a separate implementation request.

- Provide one PA card and one card for each exact RA instance.
- Label every interaction record as fixture, replay, or live.
- Show current activity, retrieved `context ref`, missing evidence, the last
  exchanged message, and whether PA is waiting for the user.
- Provide expandable ordered records for structured messages, provenance,
  assembly-plan updates, `primitive_steps`, revisions, and validation results.
- Use controlled fixtures and replay only; do not connect PA or RA.

## Phase 4: PA context retrieval, grounding, and clarification

This phase requires a separate implementation request.

- Add a Spec2Skill-owned adapter to the shared ProductAgent public interface.
- Retrieve only the controlled case references and permitted observations.
- Produce provenance-backed `target_feature`, target pose, insertion axis, and
  tolerances.
- Leave missing or contradictory evidence unresolved instead of inventing a
  value.
- When the product requirement is vague or required evidence is missing, pause
  the same interaction and have PA ask a focused question in the UI.
- Route every user reply through PA, preserve the original requirement,
  question, and reply as provenance, and resume the same interaction.
- Allow PA to repeat clarification until grounding is sufficient or the user
  cancels.

## Phase 5: PA assembly plan

This phase requires a separate implementation request.

- Have PA convert the grounded product requirement into an assembly plan.
- Show PA building the assembly plan from grounded evidence in the PA card and
  ordered interaction record.
- Keep the assembly plan separate from robot-specific `primitive_steps`.
- Do not have PA author or prescribe `primitive_steps`.

## Phase 6: live PA-to-RA communication

This phase requires a separate implementation request.

- Add a Spec2Skill-owned RobotAgent adapter for the structured exchange between
  PA and each RA.
- Send grounded assembly tasks and their required outcomes from PA to RA.
- Display exchanged messages in the corresponding live PA and RA cards and the
  ordered interaction record.
- Keep shared PA and RobotAgent implementations unchanged.
- Preserve exchanged inputs, outputs, and provenance as reviewable artifacts.

## Phase 7: RA context retrieval and primitive composition

This phase requires a separate implementation request.

- Retrieve fresh robot state and the resource-owned primitive catalog.
- Show RA building context from the retrieved `context ref` in the RA card and
  ordered interaction record.
- Have RA author `primitive_steps` against the grounded assembly task.
- Preserve every retrieved `context ref`, candidate `primitive_steps`, and
  revision as artifacts.

## Phase 8: RA validation and revision

This phase requires a separate implementation request.

- Apply PA syntax/schema/binding checks.
- Have RA perform robot-local feasibility, IK, collision, and trajectory
  validation.
- Return concrete validation feedback to RA when a candidate is rejected.
- Accept only a candidate that passes every required check.
- Show validation results and revisions in the RA card and ordered interaction
  record.
- Store validation traces and revision histories under `artifacts/`.

## Phase 9: simulation execution and evaluation

This phase requires a separate implementation request and explicit execution
authorization.

- Enable RA simulation execution only after the earlier contracts and
  validators are tested.
- Recheck fresh robot state before simulation dispatch.
- Keep robot safety and runtime authority with RA.
- Show RA as executing only while a live simulation dispatch is active.
- Keep the ground-truth evaluator separate from recognition and expose ground
  truth only after the prediction is finalized.
- Record simulation results, execution logs, and evaluation outputs under
  `artifacts/`.
- Report fixture, replay, contract validation, simulation, and physical
  execution as distinct evidence.

## UI and interface boundaries

- No public interface or UI code changes are authorized by this roadmap update.
- Future adapters provide read-only structured interaction records and a
  narrowly scoped user-reply operation routed to PA.
- Show auditable messages, retrieved sources, and outputs, not hidden model
  reasoning.
- Label fixture, replay, and live records clearly.
- Store the ordered interaction record under `artifacts/`.
- Apply the recognition do-not-leak boundary to every displayed record.

The ground-truth evaluator must remain separate from recognition. It may read
ground truth only after a prediction is finalized and must never share runtime
objects with recognition code.
