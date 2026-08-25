# Spec2Primitives Implementation Plan

Phase 0, the Phase 0.1 operator shell, Phase 1, Phase 1.1, and Phase 1.2 are
implemented. Every later phase is future work and requires a separate,
explicitly scoped implementation request. This plan does not authorize an
end-to-end implementation.

`PA` refers to ProductAgent and `RA` refers to RobotAgent throughout this plan.
Only PA and RA participate in the current Spec2Primitives roadmap.

## Directory architecture

```text
spec2primitives/
├── agents/
│   ├── pa/
│   └── ra/
├── adapters/
├── cases/
├── tools/
│   ├── exact_ref_resolver.py
│   ├── observation_context.py
│   ├── document_evidence/
│   └── rgb_d_cad_grounding/
├── references/
│   ├── products/
│   └── resources/
│       └── primitive_catalogs/
├── contexts/
│   └── <interaction_identifier>/
│       ├── products/
│       ├── resources/
│       └── interaction_record/
├── evaluations/
│   └── ground_truth/
├── schemas/
├── tests/
└── ui.py
```

- `agents/pa/` and `agents/ra/` contain only future Spec2Primitives-owned adapters
  and workflow code. Shared ProductAgent and RobotAgent implementations remain
  outside this package and read-only.
- `adapters/` retains non-agent runtime boundaries, including the implemented
  scene-only Gazebo adapter.
- `tools/` contains controlled components; tools are not agents.
- `references/` contains reusable static inputs separated into `products` and
  `resources`. A resource-owned primitive catalog is a static resource
  reference, not runtime output.
- `contexts/` contains everything retrieved or produced during one interaction.
  Product observations, served references, grounding, assembly plans, fresh
  robot state, retrieved primitive-catalog snapshots, `primitive_steps`,
  messages, and validation evidence are runtime context.
- `evaluations/` contains ground truth and post-prediction evaluation. PA, RA,
  retrieval tools, and recognition code must not read it.
- There is no `artifacts/` runtime category. Values produced during an
  interaction become context for later turns and remain under `contexts/`.
- Multiple RobotAgent instances use the same `agents/ra/` code but receive
  separate runtime resource directories using their exact RA identifiers.

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

- Establish the Spec2Primitives package and directory boundary.
- Record the ICRA 2027 title, starting case, and proposed workflow.
- Keep `NIST_assembly_instructions.pdf` under `references/products/` and
  reference the existing NIST CAD files in place.
- Provide a static `/spec2primitives` NiceGUI starting page.
- Keep all existing runtime authorities unchanged.

## Phase 0.1: NIST scene-only operator shell - implemented

- Start, stop, and show status for the no-hardware
  `gazebo_dual_spec2primitives` simulation through a narrow Spec2Primitives adapter
  protocol.
- Launch the dedicated `table_spec2primitives.world` through the package-local
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

## Phase 1: approved-source exact-ref retrieval - implemented

- Define the controlled case input as
  `product requirement: assemble Medium Gear`.
- Inventory `NIST_assembly_instructions.pdf` and all 34 current NIST STL files
  as 35 explicit approved sources.
- Define the request and served-context contracts for exact `context ref`
  retrieval.
- Implement a Spec2Primitives-owned local resolver restricted to the approved source
  inventory.
- Verify approved repository paths and provenance before serving the complete
  six-page document evidence or bounded CAD evidence.
- Reject unknown, forbidden, missing, unsupported, or malformed refs and
  sources.
- Do not store or verify SHA-256 in the Phase 1 inventory or resolver.
- Add controlled fixtures and focused resolver tests.
- Keep document evidence and CAD evidence distinct.
- Do not connect PA, modify the UI, perform grounding, or use vector retrieval.

## Phase 1.1: offline observation context contract - implemented

- Define a complete observation bundle containing the exact `cam_mk3`,
  `cam_mk4_1`, `cam_mk4_2`, and `cam_assembly` camera evidence.
- Support the `fixture`, `replay`, and reserved `live` evidence labels while
  writing and testing only `fixture` and `replay` evidence in this phase.
- Store lossless RGB PNG files, original metric `float32` depth arrays,
  calibration, frames, timestamps, synchronization measurements, and relative
  paths under
  `contexts/<interaction_identifier>/products/observations/<observation_ref>/`.
- Require registered RGB-D data, the fixed shapes and encodings, approved
  intrinsics, bounded per-camera and cross-camera timestamp skew, and at least
  one finite positive depth value per camera.
- Record `extrinsics_available: false`; do not infer extrinsics from world or
  SDF content, spawn poses, Gazebo entity state, or general TF state.
- Validate a complete bundle before an atomic, no-overwrite write and reject
  unsafe observation refs or relative observation paths.
- Keep observation refs separate from the static document/CAD source inventory.
- Do not import or call ROS2, Gazebo state, VLM, PA, RA, UI, segmentation,
  recognition, grounding, or execution behavior.

## Phase 1.2: demand-driven live Gazebo RGB-D observation tool - implemented

- Add `tools/rgb_d_cad_grounding/gazebo_observation_provider.py` with lazy ROS2
  imports and one explicit `capture_gazebo_observation(...)` operation.
- Create request-owned subscriptions only while one capture call is active; do
  not run a background provider, preload observations, or reuse a latest bundle.
- Subscribe only to `/<camera>/<camera>/image_raw`,
  `/<camera>/<camera>/depth/image_raw`, and
  `/<camera>/<camera>/camera_info` for `cam_mk3`, `cam_mk4_1`, `cam_mk4_2`,
  and `cam_assembly`, matching the live Gazebo publishers.
- Require fresh synchronized RGB, metric depth, and approved `CameraInfo` for
  all four cameras. Return `calibration_missing`, `capture_timeout`,
  `invalid_message`, or `ros_unavailable` without creating a partial context.
- Convert accepted messages into the existing Phase 1.1 `ObservationBundle`
  contract and atomically write evidence label `live` under the caller-provided
  `contexts/<interaction_identifier>/products/observations/` root.
- Preserve one lossless `<camera>_rgb.png` and one original metric
  `<camera>_depth_m.npy` per camera for direct inspection.
- Keep PA and RA disconnected. Phase 2 may let PA request this operation
  dynamically; no agent receives observations automatically.
- Cover request-only lifecycle, complete capture, freshness, synchronization,
  calibration, malformed messages, cleanup, and dependency boundaries with
  controlled tests. The live Gazebo smoke capture stored and reloaded one
  complete four-camera `live` bundle with finite positive metric depth for every
  camera.

### Phase 1.2 boundaries

- Never derive calibration or extrinsics from world or SDF content, spawn
  poses, Gazebo entity state, or general TF state.
- Preserve `run_perception:=false`, avoid existing detector services and all
  forbidden Gazebo state, and stop before segmentation or recognition.
- Do not connect PA or RA, modify the UI, render PDF pages, call a VLM, perform
  RGB segmentation, depth geometry, CAD registration, grounding, planning, or
  execution.

## Phase 2: PA context retrieval and clarification

Every Phase 2.x step requires a separate implementation request. None of these
steps is implemented.

PA retrieves permitted context before asking for clarification unless the user
intent itself is ambiguous. PA treats user expertise as unknown. The user is
authoritative about the requested goal, but user factual claims never override
contradictory approved evidence. Unsupported factual claims remain unresolved.

### Phase 2.1: product requirement intake and first `needed context` decision

- Add a Spec2Primitives-owned adapter under `agents/pa/` using composition with
  the shared ProductAgent. Do not subclass ProductAgent or LlmAgent.
- Define the planned package-local entrypoint as:

  ```python
  async def start_pa_context_interaction(
      product_agent: ProductAgentContextRuntime,
      interaction_root: Path,
      product_requirement: str,
  ) -> dict[str, object]:
      ...
  ```

- Expose only the inherited public `ask_llm_structured(...)` operation through
  `ProductAgentContextRuntime`. Do not call `ProductAgent.setup()`, start SPADE
  behaviours, build a plan, or contact CCA or RA.
- Reject an empty or whitespace-only `product_requirement`; otherwise preserve
  it exactly as supplied.
- Give PA the unchanged `product_requirement`, `approved_context_refs()`, the
  permitted request shapes, and the option to request one fresh live RGB-D
  observation. Do not preload document, CAD, or observation content.
- Run one structured PA turn with this response shape:

  ```json
  {
    "needed_context": {
      "context_ref": null,
      "request_live_observation": false,
      "clarification_question": null
    }
  }
  ```

- Require exactly one active value: an approved `context_ref`,
  `request_live_observation: true`, or a non-empty `clarification_question`.
  Reject mixed, malformed, unknown-ref, or unsupported decisions.
- Do not allow the first PA turn to return `context understanding complete`
  because no requested evidence has been served.
- Preserve the exact requirement, PA input, PA output or failure, and ordered
  interaction record under the caller-owned
  `contexts/<interaction_identifier>/` directory.
- Do not resolve document or CAD evidence, capture live RGB-D, call a VLM,
  perform grounding, modify the UI, build a plan, or execute robot behavior.

### Phase 2.2: requested context serving

- Resolve only the exact approved `context_ref` requested by PA, or perform one
  explicit fresh live RGB-D capture when PA requests it.
- Return the served context or structured failure on the next PA turn. Never
  substitute unrequested, failed, or missing evidence with guessed content.
- Preserve every request, served value, `context_ref`, `observation_ref`,
  evidence label, provenance record, and retrieval error.

### Phase 2.3: context understanding assessment

- Have PA assess each served context against the unchanged
  `product_requirement` and decide whether it needs another permitted context,
  a user clarification, or has reached `context understanding complete`.
- Preserve the auditable evidence summary and supporting refs, not hidden model
  reasoning.
- Let `assemble Medium Gear` proceed directly to context retrieval. Do not make
  clarification mandatory merely because execution details were not supplied.
- When a user factual claim is unsupported or contradicts approved evidence,
  preserve the exact claim and conflict. Do not use it as grounded evidence.

### Phase 2.4: user clarification

- Pause the same interaction only when user intent remains ambiguous or required
  evidence cannot resolve a necessary question.
- Ask one focused question, route the exact user reply back through PA, and
  preserve the original requirement, question, reply, and resulting decision.
- Resume the same interaction and allow PA to retrieve more context or repeat
  clarification until the requirement is sufficient or the user cancels.
- If the user insists on a factual claim contradicted by approved evidence, do
  not allow `context understanding complete`; let the user revise or cancel.

### Phase 2.5: `context understanding complete` handoff

- Allow `context understanding complete` only after every context item PA
  identified as required has been served, required ambiguities are resolved,
  and no required evidence remains missing or contradictory.
- Preserve the original requirement, retrieved refs, observations,
  clarification history, retrieval errors, and supporting provenance for the
  Phase 4 handoff.
- Treat `context understanding complete` as ready for Phase 4 PA grounding. It
  does not produce `target_feature`, target pose, insertion axis, tolerances, an
  assembly plan, or `primitive_steps`.

### Planned Phase 2.1 verification

- Test exact requirement preservation and the approved `context_ref`, live
  observation, and clarification decisions.
- Test rejection of malformed, mixed, unknown-ref, unsupported, and premature
  completion responses.
- Test PA-call failure recording without claiming completion.
- Verify that Phase 2.1 does not call `setup()`, planning, CCA, RA, the resolver,
  live capture, VLM, UI, or execution behavior and cannot access forbidden
  Gazebo or evaluator inputs.
- Run the focused Phase 2.1 tests, `poetry check`, repository compileall, and
  `git diff --check` when Phase 2.1 is separately implemented.

## Phase 3: PA interaction UI

This phase requires a separate implementation request.

- Provide a PA card that shows its current retrieval or clarification activity.
- Show PA requests, served context, provenance, retrieval errors, and user
  clarification in an expandable ordered interaction record.
- Label fixture, replay, and live records distinctly.
- Do not add RA interaction or robot execution.

## Phase 4: PA grounding

This phase requires a separate implementation request.

- Implement the VLM as a controlled tool under `tools/document_evidence/`, not
  as another agent.
- Let the VLM interpret approved document diagrams only. Do not give it RGB-D
  observations or use it to propose metric geometry.
- Preserve the exact document `context ref`, page-level provenance, structured
  VLM output, and uncertainty.
- Keep RGB segmentation, depth geometry, and CAD registration under
  `tools/rgb_d_cad_grounding/`.
- Do not let VLM output independently establish `target_feature`, target pose,
  insertion axis, or tolerances.
- Produce provenance-backed `target_feature`, target pose, insertion axis, and
  tolerances from retrieved evidence.
- Leave missing or contradictory evidence unresolved instead of inventing a
  value.
- Preserve the exact supporting `context ref` and provenance for each grounded
  value.
- Keep recognition inputs within the approved document, candidate CAD, RGB,
  depth, and camera calibration boundary.

## Phase 5: PA assembly plan

This phase requires a separate implementation request.

- Have PA convert the grounded product requirement into an assembly plan.
- Show PA building the assembly plan from grounded evidence in the PA card and
  ordered interaction record.
- Keep the assembly plan separate from robot-specific `primitive_steps`.
- Do not have PA author or prescribe `primitive_steps`.

## Phase 6: PA-to-RA communication

This phase requires a separate implementation request.

- Add a Spec2Primitives-owned RobotAgent adapter under `agents/ra/` for the structured
  exchange between PA and each RA.
- Send grounded assembly tasks and their required outcomes from PA to RA.
- Display exchanged messages in the corresponding live PA and RA cards and the
  ordered interaction record.
- Keep shared PA and RobotAgent implementations unchanged.
- Preserve exchanged inputs, outputs, and provenance as reviewable context
  records.

## Phase 7: RA context retrieval and primitive composition

This phase requires a separate implementation request.

- Retrieve fresh robot state and the resource-owned primitive catalog.
- Show RA building context from the retrieved `context ref` in the RA card and
  ordered interaction record.
- Have RA author `primitive_steps` against the grounded assembly task.
- Preserve every retrieved `context ref`, candidate `primitive_steps`, and
  revision in the active interaction under `contexts/`.

## Phase 8: RA validation and revision

This phase requires a separate implementation request.

- Apply PA syntax/schema/binding checks.
- Have RA perform robot-local feasibility, IK, collision, and trajectory
  validation.
- Return concrete validation feedback to RA when a candidate is rejected.
- Accept only a candidate that passes every required check.
- Show validation results and revisions in the RA card and ordered interaction
  record.
- Store validation traces and revision histories in the active interaction
  under `contexts/`.

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
- Record simulation results and execution logs in the active interaction under
  `contexts/`. Store ground truth and post-prediction evaluation under
  `evaluations/`.
- Report fixture, replay, contract validation, simulation, and physical
  execution as distinct evidence.

## UI and interface boundaries

- No public interface or UI code changes are authorized by this roadmap update.
- Phase 1 adds only a local exact-ref resolver interface. Phase 1.1 adds only an
  offline observation-bundle interface, and Phase 1.2 adds live Gazebo capture
  through that interface. ProductAgent integration begins in Phase 2.1.
- The planned Phase 2.1 adapter composes the shared ProductAgent through
  `ask_llm_structured(...)`; it does not own or invoke the ProductAgent SPADE
  lifecycle.
- Future adapters provide read-only structured interaction records and a
  narrowly scoped user-reply operation routed to PA.
- Show auditable messages, retrieved sources, and outputs, not hidden model
  reasoning.
- Label fixture, replay, and live records clearly.
- Store the ordered interaction record in the active interaction under
  `contexts/`.
- Apply the recognition do-not-leak boundary to every displayed record.
- Treat exact-ref retrieval as the initial baseline. Evaluate vector retrieval
  only as a later, separately authorized scalability extension.

The ground-truth evaluator must remain separate from recognition. It may read
ground truth only after a prediction is finalized and must never share runtime
objects with recognition code.
