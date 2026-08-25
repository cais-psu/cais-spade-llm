# Spec2Primitives Implementation Plan

Phase 0, the Phase 0.1 operator shell, Phase 1, Phase 1.1, Phase 1.2, the Phase
2 PA interaction UI, Phase 2.1, Phase 3.1, and Phase 3.2 are implemented. The
Phase 3.3 loop mechanics, persistence, numbered serving, turn limit, and UI are
implemented, but its retrieval-first clarification policy is not complete.
Phase 3.4 onward and all Phase 4 context understanding remain future work
requiring a separate, explicitly scoped implementation request. This plan does
not authorize an end-to-end implementation.

`PA` refers to ProductAgent and `RA` refers to RobotAgent throughout this plan.
Only PA and RA participate in the current Spec2Primitives roadmap.

## Current implementation status

| Phase | Status | Current boundary |
|---|---|---|
| Phase 0 and Phase 0.1 | implemented | Isolated package and no-hardware NIST operator scene. |
| Phase 1, Phase 1.1, and Phase 1.2 | implemented | Approved exact-ref retrieval plus stored and request-scoped live RGB-D observations. |
| Phase 2 and Phase 2.1 | implemented | PA UI, configurable turn limit, live PA turn counter, transcript, compact evidence summaries, and complete audit records. |
| Phase 3.1 | implemented; policy revision pending | Exact requirement intake and first request work, but first-turn clarification is still accepted by the current code. |
| Phase 3.2 | implemented | One persisted document, CAD, or live-observation request is served exactly and recorded without fallback evidence. |
| Phase 3.3 | partially implemented | The reassessment loop works, but PA can still request clarification before retrieving all relevant permitted context. |
| Phase 3.4 and Phase 3.5 | not implemented | User replies and the post-Phase 4 `context understanding complete` handoff are unavailable. |
| Phase 4 onward | not implemented | No document-diagram VLM interpretation, CAD/RGB-D grounding, assembly plan, RA communication, primitive composition, validation, or execution exists. |

The next implementation is the Phase 3.3 retrieval-first correction and its
transition into Phase 4.1. Controlled tests of retrieval and observation capture
do not constitute document-diagram understanding, metric grounding, Gazebo task
execution, or physical execution.

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
└── spec2primitives_ui.py
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
- Keep PA and RA disconnected. Phase 3 may let PA request this operation
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

## Phase 2: PA interaction UI - implemented

- Preserve the dual-Gazebo launcher controls.
- Provide a PA-only workspace with the exact `product_requirement` input, a
  `Start PA Context Interaction` control, PA activity, User ↔ ProductAgent Messages,
  `needed_context`, served context, Evidence Sources, retrieval error,
  clarification, and ordered interaction record areas. Evidence Sources is the
  operator-facing label for the unchanged internal `provenance` record.
- Show a disabled Phase 5 Assembly Plan layout with Step, Assembly Task,
  Required Outcome, and Evidence Sources columns, but no invented plan entries.
- Keep the disabled Phase 5 Assembly Plan preview separate from connected PA
  interaction results.
- Add the first RA UI card in Phase 6 when PA-to-RA messages exist, then extend
  that card during Phase 7 composition and Phase 8 validation/revision.

### Phase 2.1: UI connection through Phase 3.3 - implemented

- Compose one application-owned shared ProductAgent behind
  `ProductAgentContextRuntime`; do not call `setup()` or expose its SPADE
  lifecycle to the UI.
- Create one unique caller-owned interaction directory per submission and
  preserve the exact `product_requirement`.
- Let the operator select `Maximum PA turns` from 2 through 50, with 12 as the
  default and Phase 3.1 counted as turn 1. Keep the value fixed while the
  interaction is active.
- Run Phase 3.1 through Phase 3.3 when requested context is served. Display every
  PA decision, compact served result, Evidence Sources, retrieval error,
  clarification, completion, live turn counter, and complete ordered records.
- Stop on completion, clarification, failure, or the emergency turn limit. Do
  not perform grounding, planning, RA, CCA, or robot execution.

## Phase 3: PA context retrieval and clarification

Every Phase 3.x step requires a separate implementation request. Phase 3.1 and
Phase 3.2 are implemented. Phase 3.3 is partially implemented; its mechanical
loop is present but its retrieval-first clarification gate remains future work.
Phase 3.4 and Phase 3.5 are not implemented.

PA retrieves permitted context and Phase 4 attempts context understanding before
PA asks for clarification. PA treats user expertise as unknown. The user is
authoritative about the requested goal, but user factual claims never override
contradictory approved evidence. Unsupported factual claims remain unresolved.
An unresolved component location, receiving feature, pose, diagram association,
CAD association, or current arrangement is a system evidence problem, not a
user clarification question.

### Phase 3.1: product requirement intake and first `needed context` decision - implemented; policy revision pending

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
- The current implementation accepts first-turn `clarification_question`. The
  retrieval-first revision must stop accepting it as a terminal shortcut and
  must proceed through permitted evidence retrieval and Phase 4 understanding
  before Phase 3.4 can ask the user.
- Do not allow the first PA turn to return `context understanding complete`
  because no requested evidence has been served.
- Preserve the exact requirement, PA input, PA output or failure, and ordered
  interaction record under the caller-owned
  `contexts/<interaction_identifier>/` directory.
- Do not resolve document or CAD evidence, capture live RGB-D, call a VLM,
  perform grounding, modify the UI, build a plan, or execute robot behavior.

### Phase 3.2: requested context serving - implemented

- Read and strictly validate the persisted Phase 3.1 requirement and successful
  `needed_context` decision; do not accept a replacement decision from a caller.
- Resolve only the exact approved `context_ref` requested by PA, or perform one
  explicit fresh live RGB-D capture as `observation_0001` when PA requests it.
- Store static results under `products/served_references/`, live artifacts under
  `products/observations/`, and the ordered request, result, or failure under
  `interaction_record/retrieval_0001.json` using exclusive writes.
- Return the exact served context or a structured failure. Never substitute
  unrequested, failed, or missing evidence with guessed content.
- Preserve every request, served value, `context_ref`, `observation_ref`,
  evidence label, provenance record, and retrieval error.
- Stop before the next PA turn. Do not call ProductAgent, an LLM, VLM, UI,
  grounding, planning, RA, CCA, or robot execution.

### Phase 3.3: configurable ReAct-style context understanding assessment - partially implemented

- Record `max_pa_turns` and `live_observation_timeout_sec` exclusively in
  `interaction_record/pa_context_settings.json`; count `turn_0001` in the limit.
- Have PA assess all accumulated served context against the unchanged
  `product_requirement` and decide whether it needs one more approved
  `context_ref`, a fresh live observation, or is ready to transition into Phase
  4 context understanding.
- Automatically serve each valid request and reassess. Number PA and retrieval
  records together and assign live captures the next `observation_000N`.
- Require sufficient context for the requested component, assembly destination,
  receiving feature, and current arrangement when relevant, with no unresolved
  ambiguity, contradiction, outstanding request, or failed required retrieval.
- Tell PA that Phase 4 owns grounding, Phase 5 owns robot-independent assembly
  planning, Phase 6 introduces PA-to-RA communication, and Phase 7 lets RA
  retrieve robot state and the resource-owned primitive catalog. Do not contact
  RA or retrieve that catalog in Phase 3.
- Preserve exact prompts, response formats, raw outputs, served results,
  Evidence Sources through internal `provenance`, and failures. Do not preserve
  hidden model reasoning or RGB/depth arrays in JSON.
- Let `assemble Medium Gear` proceed directly to context retrieval. PA must not
  ask the user to choose a gear-shaft location while relevant approved document,
  `Gear_Medium.STL`, `Gear_Plate.STL`, `Gear_Shaft.STL`,
  `GMC_Laser_Plate_Virtual.STL`, or live-observation evidence can still address
  the unresolved system question. Do not hard-code their request order.
- Add this Phase 3.3 transition response without changing the existing field
  names:

  ```json
  {
    "needed_context": null,
    "context understanding complete": false
  }
  ```

  It means PA selected no further permitted retrieval and the recorded evidence
  must proceed to Phase 4. It does not mean the context has been understood.
- Do not accept `clarification_question` from the Phase 3 retrieval loop. Only a
  persisted Phase 4.3 result may enter Phase 3.4.
- When a user factual claim is unsupported or contradicts approved evidence,
  preserve the exact claim and conflict. Do not use it as grounded evidence.
- Stop on the Phase 4 transition, PA or retrieval failure, invalid response,
  existing-record conflict, or `pa_turn_limit_reached`. Limit exhaustion never
  implies completion.
- The current implementation still permits early clarification and early
  `context understanding complete`; that observed behavior is the remaining
  Phase 3.3 correction, not completed context understanding.

### Phase 3.4: user clarification

- Enter Phase 3.4 only from a persisted Phase 4.3 result showing that the whole
  available context was processed and only unresolved user intent prevents
  completion.
- Do not ask the user to determine a component identity, receiving feature,
  location, pose, geometric association, or current arrangement that approved
  evidence and Phase 4 tools are responsible for establishing.
- Ask one focused question, route the exact user reply back through PA, and
  preserve the original requirement, question, reply, and resulting decision.
- Resume the same interaction through Phase 4.3 and allow another `needed_context`
  request, one further clarification, or completion until the requirement is
  sufficient or the user cancels.
- If the user insists on a factual claim contradicted by approved evidence, do
  not allow `context understanding complete`; let the user revise or cancel.

### Phase 3.5: `context understanding complete` handoff

- Allow `context understanding complete` only after Phase 4.1 document-diagram
  interpretation, Phase 4.2 CAD/RGB-D grounding, and the Phase 4.3 decision have
  succeeded with no required ambiguity, contradiction, missing evidence, or
  failed required retrieval.
- Preserve the original requirement, retrieved refs, observations,
  clarification history, retrieval errors, Phase 4 outputs, and supporting
  provenance for the Phase 5 handoff.
- Treat `context understanding complete` as ready for Phase 5 PA assembly
  planning. It does not produce an assembly plan or `primitive_steps`.

### Phase 3.1 verification

- Preserve the existing exact requirement, approved `context_ref`, and live
  observation coverage. Revise first-turn clarification coverage so that it is
  rejected instead of treated as a valid terminal decision.
- Test rejection of clarification, malformed, mixed, unknown-ref, unsupported,
  and premature completion responses.
- Test PA-call failure recording without claiming completion.
- Verify that Phase 3.1 does not call `setup()`, planning, CCA, RA, the resolver,
  live capture, VLM, UI, or execution behavior and cannot access forbidden
  Gazebo or evaluator inputs.
- Run the focused Phase 3.1 tests, `poetry check`, repository compileall, and
  `git diff --check` for the implemented Phase 3.1 boundary.

### Phase 3.2 verification

- Test exact document, CAD, and controlled live-observation serving, including
  provenance, deterministic artifact references, and one requested tool call.
- Test clarification decisions, invalid Phase 3.1 records, resolver and capture
  failures, malformed results, and existing-record protection without fallback
  evidence.
- Verify that Phase 3.2 does not call ProductAgent, an LLM, VLM, UI, grounding,
  planning, RA, CCA, forbidden Gazebo-state inputs, or robot execution.
- Run the focused Phase 3.2 tests, the complete Spec2Primitives suite, Ruff,
  `poetry check`, repository compileall, and `git diff --check`.

### Phase 3.3 verification

- Test limits 2 and 12 plus the UI maximum 50, invalid values, exclusive settings,
  cumulative evidence, exact requirement preservation, numbered records, and
  one PA call plus one resolver or capture call per request.
- Revise the Medium Gear document, CAD, and live-observation sequence to finish
  with the Phase 4 transition rather than direct completion or clarification.
  Preserve alternative PA-selected ordering, malformed or duplicate refs, PA
  and retrieval failure, record conflicts, and `pa_turn_limit_reached` coverage.
- Verify every decision and compact served result appears in the PA UI while the
  expandable audit view preserves full records and internal `provenance`.
- Verify Phase 3.3 has no VLM, grounding, planning, primitive-catalog, RA, CCA,
  forbidden Gazebo-state, evaluator, or execution dependency.
- Add revised tests proving that Phase 3.3 requests relevant remaining evidence
  instead of asking which gear-shaft location receives the Medium Gear.
- Test the exact `needed_context: null` and
  `context understanding complete: false` transition into Phase 4 and rejection
  of Phase 3 clarification or completion before a Phase 4.3 record exists.

## Phase 4: PA grounding

This is the context-understanding stage and requires separate, bounded
implementation requests. It consumes the complete persisted Phase 3 evidence;
it does not assume that retrieving a PDF or STL means its diagrams or geometry
were understood.

### Phase 4.1: document-diagram VLM interpretation

- Implement the VLM as a controlled tool under `tools/document_evidence/`, not
  as another agent.
- Let the VLM interpret approved document diagrams only. Do not give it RGB-D
  observations or use it to propose metric geometry.
- Preserve the exact document `context ref`, page-level provenance, structured
  VLM output, and uncertainty.

### Phase 4.2: CAD and RGB-D grounding

- Keep RGB segmentation, depth geometry, and CAD registration under
  `tools/rgb_d_cad_grounding/`.
- Load approved CAD geometry for grounding; the existing filename, units,
  triangle count, and bounds summary alone does not establish a component match,
  receiving feature, or pose.
- Use only approved RGB, depth, camera calibration, document interpretation, and
  CAD evidence. Never use Gazebo model names, entity state, world contents,
  configured spawn poses, or evaluator data.
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

### Phase 4.3: post-understanding decision

- Assess the combined Phase 4.1 and Phase 4.2 outputs against the exact
  `product_requirement` and every unresolved requirement recorded by Phase 3.
- Use one existing structured decision:
  - an active `needed_context` with `context understanding complete: false`
    returns to Phase 3.2 for exactly that approved ref or fresh observation;
  - an active `clarification_question` with
    `context understanding complete: false` enters Phase 3.4 only when the
    unresolved value is user intent that system evidence cannot determine;
  - `needed_context: null` with `context understanding complete: true` enters
    Phase 3.5;
  - a VLM, retrieval, CAD, RGB-D, contradiction, or grounding failure stops
    fail-closed and must not be converted into a user clarification question.
- Preserve the Phase 4 input, evidence refs, output, uncertainty, unresolved
  values, and decision in the ordered interaction record without hidden model
  reasoning.
- Allow the runtime to loop Phase 4.3 → Phase 3.2 → Phase 4 when more permitted
  evidence can resolve the missing context. Phase numbering does not require a
  one-way runtime sequence.

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

- Phase 2 preserves the public `render(runtime: DualGazeboRuntime)` interface
  while adding only a disconnected PA presentation scaffold.
- Phase 1 adds only a local exact-ref resolver interface. Phase 1.1 adds only an
  offline observation-bundle interface, and Phase 1.2 adds live Gazebo capture
  through that interface. ProductAgent integration begins in Phase 3.1.
- The Phase 3.1 adapter composes the shared ProductAgent through
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
