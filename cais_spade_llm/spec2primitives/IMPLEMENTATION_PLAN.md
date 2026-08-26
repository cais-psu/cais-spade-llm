# Spec2Primitives Implementation Plan

Phase 0, the Phase 0.1 operator shell, Phase 1, Phase 1.1, Phase 1.2, the Phase
2 PA interaction UI, Phase 2.1, Phase 3.1, and Phase 3.2 are implemented. The
Phase 3.3 loop mechanics, persistence, numbered serving, turn limit, and UI are
implemented, but the loop still assesses raw retrieved summaries and can finish
before ontology-backed grounding. The standalone Phase 4.0 PPR context
foundation is implemented, but it is not wired into Phase 3. Phase 3.4 onward
and Phase 4.1 onward remain future work requiring separate, explicitly scoped
implementation requests. The target runtime initializes Phase 4.0 before PA
selects its first source, then alternates Phase 3 retrieval with Phase 4
interpretation and ABox updates. Phase numbers identify capabilities, not a
required one-way runtime order. This plan does not authorize an end-to-end
implementation.

`PA` refers to ProductAgent and `RA` refers to RobotAgent throughout this plan.
Only PA and RA participate in the current Spec2Primitives roadmap.

## Current implementation status

| Phase | Status | Current boundary |
|---|---|---|
| Phase 0 and Phase 0.1 | implemented | Isolated package and no-hardware NIST operator scene. |
| Phase 1, Phase 1.1, and Phase 1.2 | implemented | Approved exact-ref retrieval plus stored and request-scoped live RGB-D observations. |
| Phase 2 and Phase 2.1 | implemented | PA UI, configurable turn limit, live PA turn counter, transcript, compact evidence summaries, and complete audit records. |
| Phase 3.1 | implemented; ontology integration pending | Exact requirement intake and first request work exist, but no TBox/ABox is initialized before the first request and first-turn clarification is still accepted. |
| Phase 3.2 | implemented | One persisted document, CAD, or live-observation request is served exactly and recorded without fallback evidence. |
| Phase 3.3 | partially implemented | Numbered retrieval can repeat, but served evidence is not yet interpreted into an ABox and PA can still clarify or complete from raw summaries. |
| Phase 3.4 and Phase 3.5 | not implemented | User replies and the post-Phase 4 `context understanding complete` handoff are unavailable. |
| Phase 4.0 | implemented; Phase 3 integration pending | Shared RDFLib TBox loading and PA-owned product-context support for independent interaction ABoxes, controlled-tool capability descriptors, and evidence-backed triple-delta validation and persistence exist. Invalid deltas are rejected before mutation. |
| Phase 4.1 onward | not implemented | No document-diagram VLM interpretation, CAD/RGB-D grounding, assembly plan, RA communication, primitive composition, validation, or execution exists. |

The Phase 4.0 foundation separates shared immutable TBox semantics from the
PA-owned writable product context. It loads a caller-supplied TBox, initializes
an independent interaction ABox from the exact requirement, and validates and
persists small tool-capability and generic evidence-backed triple-delta
contracts. The pasted OWL serves only as an RDF/XML parser and mixed-graph test
fixture until the authoritative TBox is provided: it contains both schema
axioms and named individuals, so its ABox facts never become runtime task facts.
The next small implementation is the Phase 3.3 integration revision that
orchestrates the dynamic retrieval-grounding loop against this foundation.
Controlled tests of retrieval and observation capture do not constitute
document-diagram understanding, metric grounding, Gazebo task execution, or
physical execution.

## ICRA implementation scope

The central implementation contribution is one RA-owned agentic composition
loop, not a symbolic primitive planner:

```text
grounded task transition for which the selected RA has no matching
predefined composite function, or selected recovery event with the same gap
        + required outcome and grounded context refs
        + fresh selected-RA state
        + a primitive-only catalog of exactly eight resource-owned
          semantic primitive interfaces
        + partial local executable contracts and validator endpoints
        ↓
RA LLM authors one complete candidate primitive_steps program
        ↓
non-mutating contract, backward, forward, physical, and outcome validation
        ↓
accept unchanged candidate or return findings for RA agentic revision
```

For this plan, a resource-owned execution primitive is the lowest callable unit
exposed at the RA composition boundary. A predefined composite function is a
callable whose internal primitive selection, ordering, and binding policy were
implemented before the current interaction. The prior-work functions
`move_to_pick_location`, `pick_part`, `move_loaded_to_destination`, and
`place_part` are therefore predefined composite functions, not primitives in
this study. They must not be supplied to the Spec2Primitives composer or invoked
as shortcuts in an evaluated composition.

The earlier RCIM workflow plans with and assigns predefined composite functions.
Spec2Primitives begins at the unresolved next layer: inherited allocation has
selected an RA, but that RA has no matching predefined composite function and
exposes only the exact eight execution primitives. The accepted
`primitive_steps` is a newly authored task-specific composite program for the
current interaction. Requirement interpretation, document retrieval, capability
matching, and resource assignment remain inherited or supporting mechanisms,
not re-claimed contributions.

The RA LLM is the only component that selects, orders, and parameterizes
primitives or structurally revises a rejected candidate. Deterministic checks
may accept, reject, and explain a candidate, but they must never add, remove,
reorder, parameterize, or repair steps. A candidate's claimed final state is not
authoritative; required outcomes must be supported by the validation and
execution evidence described in Phases 8 and 9.

Each primitive interface exposes its exact symbol, an operation description,
accepted parameter names and types, callable resource-owned validators or
executor, and direct returned evidence or guaranteed local result where one is
truthful. The eight interfaces do not collectively supply a complete symbolic
state vocabulary, task-specific goal decomposition, global transition model,
complete action preconditions and effects, primitive order, or a rule mapping a
product outcome to an expected primitive sequence. They are executable
interfaces with partial local contracts, not a task-specific PDDL domain.

Standard classical PDDL cannot be invoked directly on that input because a
complete planning domain and symbolic goal are not supplied. This is an input
and modeling boundary, not a claim that PDDL is theoretically incapable of the
task. A manually completed domain, an external compiler, or black-box search
could provide another composer and must be treated as a baseline rather than
silently excluded.

The ICRA implementation uses one composition core for two task origins:

- **Nominal case:** PA starts from the exact raw requirement, dynamically
  grounds it, and creates a robot-independent task-transition contract without
  a matching predefined composite function, stored nominal task program, or
  `primitive_steps`.
- **Recovery transfer case:** the existing recovery framework starts from the
  fault and runtime state, then supplies an already selected, validated, and
  resource-assigned recovery event. A read-only Spec2Primitives adapter maps it
  to the same task-transition contract without supplying a primitive
  decomposition.

Both cases must use the same RA composition entrypoint, exact eight primitive
symbols, prompt policy, context-request protocol, candidate schema, validator
sequence, revision loop, and execution boundary. A stored nominal program,
product-specific macro, `robot_task_program`, capability decomposition, or
preauthored recovery primitive sequence is a modeled baseline and cannot count
as dynamic primitive composition.

Both evaluated cases must also satisfy the same admission condition: the
selected RA has no callable predefined composite function that realizes the
required transition. A case solvable by invoking an existing
`move_to_pick_location`, `pick_part`, `move_loaded_to_destination`, `place_part`,
or equivalent task-specific function demonstrates prior-function reuse, not the
Spec2Primitives contribution. Each case must require a structural primitive
decision such as selection, ordering, repetition, or an intermediate step; a
parameter-only variant of an existing composite function is insufficient.
Internally, an admitted primitive may still use a controller, motion planner, or
estimator while remaining atomic at the RA composition boundary.

The lightweight ontology is the formal semantic backbone but not the composer.
The immutable TBox defines generic PPR meaning. PA fills only the
interaction-specific product, feature, requested-process, and outcome ABox from
approved evidence. The selected RA supplies a separate resource-catalog ABox
snapshot in which its resource individual is linked through `capableOf` to the
exact eight primitive-process interface individuals. Their executable contract
details and numeric runtime values remain in typed context records.

A Spec2Primitives-owned read-only projection adapter dynamically joins the
current task-relevant PA ABox assertions, selected-resource catalog assertions,
and authorized typed context refs for the RA LLM. The shared RobotAgent does not
depend on RDFLib or mutate either ABox. The requested assembly process may be
linked through `realizes` to its grounded product outcome, but it must not be
linked through `requires`, `precedes`, `capableOf`, or another relation to an
expected primitive set or order. This deliberately leaves the semantic gap that
the RA LLM crosses when it authors candidate `primitive_steps`; the candidate
record creates the task-to-primitive connection dynamically. Ontology
retrieval grounds the reasoning input but does not infer or validate the
sequence.

Resource discovery and optimal allocation are inherited from prior work and
remain out of scope. The nominal path consumes the registered RA roster and the
exact `resource_jid` returned by that mechanism; the recovery event already
contains its assigned `resource_jid`. Communication is unicast to that RA, not a
broadcast. Recovery event generation and selection, DES/CCA reasoning, fault
diagnosis, ontology design, a complete symbolic planner, controller design, and
proof of globally complete or optimal composition are not ICRA implementation
contributions.

## Shared PA and RA dynamic context-retrieval pattern

PA and RA use the same bounded control pattern but operate over different
authorities and context:

```text
current objective + currently owned context
        ↓
identify one concrete missing input
        ↓
select one authorized producer and context source
        ↓
retrieve or compute one auditable result
        ↓
validate, persist, and merge only supported information
        ↓
reassess the objective
        ↓
retrieve again, hand off, return missing_context, or stop fail-closed
```

Both loops must preserve the exact request, selected producer, returned refs,
provenance, failures, freshness, and reassessment result. Neither loop retrieves
every available source by default, repeats a failed request without new
justification, treats turn-limit exhaustion as completion, or asks the LLM to
self-certify sufficiency.

PA owns product-intent and scene grounding:

- PA starts from the exact requirement, current interaction ABox, typed product
  context, unresolved claims, and controlled-tool capability descriptors.
- PA may request one approved document or CAD ref, or one fresh permitted RGB-D
  observation, then invoke only the matching controlled interpretation or
  grounding tool.
- PA merges only evidence-backed product facts and retains numeric poses,
  calibration, tolerances, and observations in typed context records.
- PA hands off when no currently identified product-level ambiguity blocks a
  robot-independent task-transition contract. This does not imply that RA has
  enough resource context or that a primitive program is feasible.
- PA never retrieves the primitive catalog, robot state, IK, collision,
  trajectory, or execution evidence and never authors `primitive_steps`.

RA owns embodiment grounding and composition readiness:

- RA begins by retrieving a read-only task-relevant ontology projection, one
  fresh selected-resource snapshot, and the complete catalog snapshot for the
  exact eight semantic primitive interfaces. The projection includes the
  current requested process and grounded outcome, relevant product and feature
  assertions, the selected resource, and all eight `capableOf` primitive-process
  offerings. It contains no task-to-primitive mapping or expected order.
- The complete catalog is the allowed synthesis surface, so all eight offerings
  are retrieved rather than selecting only primitives that would reveal an
  expected solution. Their semantic interface cards accompany the ontology
  projection because a primitive name or class assertion alone is not enough
  for reliable composition.
- RA then derives further needs from unbound candidate parameters, stale state,
  primitive-interface requirements, or validator findings. It may retrieve or
  compute only resource-owned state, target-generation, IK, collision, grasp,
  release, trajectory, operational, and execution context.
- A product or scene binding gap is returned to PA as structured
  `missing_context`; RA does not reinterpret raw documents, CAD, RGB, or depth.
  PA returns a versioned grounded update, after which RA refreshes affected
  resource context and authors a new complete candidate.
- RA is composition-ready only when every parameter is bound to an authorized
  value or context ref and one unchanged candidate passes every applicable
  local-contract, physical, and projected-outcome validator. Execution outcome
  still requires post-action observation.
- RA never mutates the ABox, asks the user directly, invents a pose or state, or
  treats its own final-state statement as evidence.

The commonality is therefore the retrieve → validate → persist → reassess loop,
not shared data access. PA determines whether the product task is grounded; RA
determines whether a candidate program is bound and executable under fresh
resource conditions.

### Semantic-need to evidence-source resolution

Primitive interfaces must require grounded semantic inputs, not raw evidence
modalities. For example, an interface may need an authorized product binding,
pose-record ref, target ref, or constraint record; it must not directly require
"a PDF," "an STL," or "RGB-D." The controlled PA tools consume those raw
sources to produce the grounded inputs.

Resolve context dynamically with this dependency chain:

```text
task or candidate exposes one missing typed input
        ↓
check whether the current ABox or typed context already supports it
        ↓ missing
match the semantic input type to one authorized producer's can_produce entry
        ↓
resolve only that producer's may_require evidence refs
        ↓
retrieve the approved document or CAD, or capture fresh RGB-D and calibration
        ↓
run the producer, validate and persist its result, then reassess
```

`can_produce` and `may_require` are fields in controlled typed producer
descriptors whose values reference approved PPR types or typed-context kinds;
they are not new predicates inserted into the authoritative TBox. PPR facts are
matched through the ontology. Poses, calibration, CAD correspondence, and other
geometry values may remain in typed context records when the loaded TBox does
not provide vocabulary for them.

This design never declares all document, STL, RGB, depth, and calibration inputs
mandatory. Documentation is retrieved only when a missing process, product,
feature, outcome, or constraint relation can be produced from it. CAD and fresh
RGB-D are retrieved only when a missing observation, identity correspondence,
pose, grasp, target, or related typed binding requires that grounding producer.
If no authorized producer advertises the missing input, stop fail-closed rather
than asking the ontology or LLM to invent a source.

OWL inference supplies type and relation semantics but follows open-world
semantics, so absence of a fact alone is not a completeness result. A small
SHACL or equivalent boundary check may report a currently required grounded
fact as missing only after that need is identified from the task or primitive
interface. It must not be a static shape that hard-codes every modality or an
expected primitive recipe. Adding pySHACL remains separately authorized.

## Requirement-to-action workflow

The nominal case follows this end-to-end path without a matching predefined
composite function, stored nominal task program, or primitive recipe:

```text
exact product_requirement
        ↓
Phase 4.0 loads the fixed TBox and initializes one interaction ABox
        ↓
PA identifies one unresolved product or scene need
        ↓
PA retrieves one approved source and invokes its matching controlled tool
        ↓
PA validates and persists evidence-backed facts, then reassesses
        ↺ retrieve another relevant source only when the updated context requires it
        ↓
PA projects a versioned robot-independent task-transition contract
        ↓
inherited allocation supplies one selected resource_jid; PA unicasts the contract
        ↓
selected RA retrieves a task-relevant ontology projection, fresh state,
and the complete exact-eight interface catalog
        ↓
RA dynamically retrieves any additional selected-resource context it identifies
        ↓
RA LLM authors one complete candidate primitive_steps program
        ↓
non-mutating declared-contract, resource, physical, and outcome checks
   ↙ product/scene gap       ↓ robot-local finding         ↘ accepted unchanged
PA grounds a versioned    RA retrieves/reasons and          fresh-state recheck
task update; RA authors   authors a new candidate                    ↓
a new candidate                     ↺                            RA execution
                                                                     ↓
                                              observation-backed realized outcome
```

The recovery case enters the same composition path with an already selected,
validated, and resource-assigned recovery event instead of the nominal Phase 5
contract. The imported event must contain the required transition and grounded
bindings but no completed `primitive_steps`. A binding-only gap may use the same
PA update path. A change to the event identifier, required outcome, semantics,
feasibility, or `resource_jid` closes the transfer fail-closed until the existing
recovery framework supplies a newly validated event.

At no point does PA retrieve every modality by default or RA retrieve every
resource record by default. PA chooses relevant product evidence from the
evolving ABox. RA's bounded task-relevant ontology projection, fresh state, and
complete exact-eight interface catalog are mandatory initial prerequisites;
every later RA retrieval is selected from the current candidate need or
validator finding. Validators and ontology reasoning never create a primitive
step.

## Directory architecture

```text
spec2primitives/
├── agents/
│   ├── pa/
│   └── ra/
├── adapters/
├── cases/
├── ontology/
│   └── ppr_tbox.py
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

- `agents/pa/` contains Spec2Primitives-owned PA workflow code and the PA-owned
  product context. `agents/ra/` remains the location for future
  Spec2Primitives-owned RA adapter and workflow code. Shared ProductAgent and
  RobotAgent implementations remain outside this package and read-only.
- `adapters/` retains non-agent runtime boundaries, including the implemented
  scene-only Gazebo adapter.
- `ontology/` contains shared immutable ontology semantics and TBox profile
  validation. It does not own a writable interaction or resource ABox.
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
loop is present but its ontology-grounded assessment and clarification gate
remain future work. Phase 3.4 and Phase 3.5 are not implemented.

In the target runtime, Phase 3 retrieval and Phase 4 grounding form one loop.
Phase 4.0 loads the fixed TBox and initializes the interaction ABox immediately
after requirement intake. Phase 3.1 bootstraps the first source decision. For
each later cycle, Phase 4.3 returns the next `needed_context` decision from the
updated ABox; Phase 3.3 validates and routes it, Phase 3.2 serves that source,
the matching controlled Phase 4 tool interprets it, and a validated
evidence-backed triple delta updates the ABox. One source per operation is an
audit boundary, not a one-source limit for the interaction; source count and
order are determined at runtime, and PA does not retrieve every approved source
blindly.

PA asks for clarification only after relevant permitted evidence and matching
controlled tools cannot resolve the remaining user-intent question. PA treats
user expertise as unknown. The user is authoritative about the requested goal,
but user factual claims never override contradictory approved evidence.
Unsupported factual claims remain unresolved. An unresolved component
location, receiving feature, pose, diagram association, CAD association, or
current arrangement is a system evidence problem, not a user clarification
question.

### Phase 3.1: product requirement intake and first `needed context` decision - implemented; ontology integration and policy revision pending

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
- In the ontology-backed revision, invoke Phase 4.0 after preserving the exact
  requirement and before the first source decision. Seed the interaction ABox
  with the request as an unresolved goal or claim, not as evidence-backed facts.
- Configure the authoritative schema-only TBox path and namespace explicitly;
  never promote either test fixture to runtime status. Fail the interaction
  closed on TBox loading, profile validation, or ABox initialization errors,
  before PA can choose an evidence source.
- Give PA the unchanged `product_requirement`, `approved_context_refs()`, the
  compact unresolved product-context view, controlled-tool capability
  descriptors, permitted request shapes, and the option to request one fresh
  live RGB-D observation. Do not preload or interpret document, CAD, or
  observation content in this bootstrap slice.
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

- In the ontology-grounded target, require exactly one evidence request: an
  approved `context_ref` or `request_live_observation: true`.
  `clarification_question` must remain null on this first turn. Reject mixed,
  malformed, unknown-ref, unsupported, or first-turn clarification decisions.
- The current implementation accepts first-turn `clarification_question`. The
  ontology-grounded revision must stop accepting it as a terminal shortcut and
  must proceed through permitted evidence retrieval, controlled interpretation,
  and ABox updates before Phase 3.4 can ask the user.
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
- Treat this as one auditable retrieval operation. The Phase 3.3 orchestrator
  may call Phase 3.2 repeatedly for a dynamic sequence of relevant sources.
- Stop before the next PA turn. Do not call ProductAgent, an LLM, VLM, UI,
  grounding, planning, RA, CCA, or robot execution.

### Phase 3.3: configurable ReAct-style context understanding assessment - partially implemented

- Record `max_pa_turns` and `live_observation_timeout_sec` exclusively in
  `interaction_record/pa_context_settings.json`; count `turn_0001` in the limit.
- Make Phase 3.3 the outer orchestration loop, not a second decision authority.
  After the Phase 3.1 bootstrap request, it accepts only a persisted Phase 4.3
  decision over the current ABox; it must not declare sufficiency or select a
  different source from raw served summaries.
- Validate that the returned unresolved semantic need, selected exact approved
  `context_ref` or fresh observation, and controlled producer agree with the
  producer's `can_produce` and `may_require` descriptor. These descriptors route
  evidence; they do not define hard-coded assembly slots or expected answers.
- For every valid request, call Phase 3.2 once, invoke only the matching Phase
  4.1 or Phase 4.2 producer, validate and merge its generic triple delta, and
  invoke Phase 4.3 before choosing the next source. Number PA, retrieval,
  interpretation, delta, and decision records together and assign live captures
  the next `observation_000N`.
- Let PA dynamically retrieve several relevant sources when the evolving ABox
  requires them. Do not hard-code a PDF → CAD → RGB-D order, stop after one
  source by design, or retrieve the entire approved inventory by default.
- Require sufficient product-level grounding for the exact requirement, with no
  unresolved ambiguity, contradiction, outstanding evidence need, or failed
  required retrieval. Do not encode a fixed checklist of receiving features,
  observed components, CAD correspondence, or RGB-D fields in Phase 3.3.
- Tell PA that Phase 4 owns grounding, Phase 5 owns robot-independent assembly
  planning, Phase 6 introduces PA-to-RA communication, and Phase 7 lets RA
  retrieve robot state and the resource-owned primitive catalog. Do not contact
  RA or retrieve that catalog in Phase 3.
- Preserve exact prompts, response formats, raw outputs, served results,
  Evidence Sources through internal `provenance`, and failures. Do not preserve
  hidden model reasoning or RGB/depth arrays in JSON.
- Let `assemble Medium Gear` proceed directly to context retrieval. In the
  controlled case, the approved document, candidate CAD refs, and live
  observation are test evidence that PA may select when the evolving ABox and
  tool descriptors justify them; they are not a production required-source list
  or a hard-coded request order. PA must not ask the user to determine a
  gear-shaft location while permitted system evidence can still address it.
- Do not use a no-op transition from Phase 3 into a later one-way Phase 4 stage.
  Phase 4.0 already exists and Phase 4 interpretation and assessment run inside
  each retrieval cycle. Only a persisted Phase 4.3 result may request more
  context, enter Phase 3.4, or declare `context understanding complete`.
- When a user factual claim is unsupported or contradicts approved evidence,
  preserve the exact claim and conflict. Do not use it as grounded evidence.
- Track attempted sources, produced assertions, evidence gaps, failures, and
  observation freshness so the loop does not repeat a request without new
  justification.
- Stop on a terminal Phase 4.3 decision, unrecoverable PA, retrieval, or
  interpretation failure, invalid response, existing-record conflict, or
  `pa_turn_limit_reached`. Limit exhaustion never implies completion.
- The current implementation still permits early clarification and early
  `context understanding complete` from raw evidence and has no Phase 4 tool or
  ABox integration; that observed behavior is the remaining Phase 3.3
  correction, not completed context understanding.

### Phase 3.4: user clarification

- Enter Phase 3.4 only from a persisted Phase 4.3 result showing that every
  relevant permitted source identified by the current knowledge need and tool
  capabilities was processed or exhausted, and only unresolved user intent
  prevents completion.
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

- Allow `context understanding complete` only after the Phase 4.0 PPR-aligned
  representation and every Phase 4.1 document-diagram or Phase 4.2 CAD/RGB-D
  producer invocation selected by the dynamic loop has completed, and Phase 4.3
  finds no product-level ambiguity, contradiction, missing evidence, or failed
  required retrieval.
- Preserve the original requirement, retrieved refs, observations,
  clarification history, retrieval errors, Phase 4 outputs, and supporting
  provenance for the Phase 5 handoff.
- Treat `context understanding complete` as ready for Phase 5 PA assembly
  planning. It does not produce an assembly plan or `primitive_steps`, and it
  does not guarantee that RA can bind or execute every required primitive.

### Phase 3.1 verification

- Preserve the existing exact requirement, approved `context_ref`, and live
  observation coverage. Revise first-turn clarification coverage so that it is
  rejected instead of treated as a valid terminal decision.
- Add integration coverage proving that Phase 4.0 initializes the TBox/ABox
  before PA's first source decision and does not turn the requirement into
  unsupported factual assertions.
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
- Test rejection of clarification decisions and other invalid Phase 3.1
  records, resolver and capture failures, malformed results, and existing-record
  protection without fallback evidence.
- Verify that Phase 3.2 does not call ProductAgent, an LLM, VLM, UI, grounding,
  planning, RA, CCA, forbidden Gazebo-state inputs, or robot execution.
- Run the focused Phase 3.2 tests, the complete Spec2Primitives suite, Ruff,
  `poetry check`, repository compileall, and `git diff --check`.

### Phase 3.3 verification

- Test limits 2 and 12 plus the UI maximum 50, invalid values, exclusive settings,
  cumulative ABox state, exact requirement preservation, numbered records, and
  exactly one resolver or capture call per retrieval operation.
- Test a Medium Gear interaction that dynamically selects and interprets
  multiple relevant document, CAD, and observation sources before PA handoff.
  Preserve alternative PA-selected ordering and source counts, malformed or
  unjustified duplicate refs, PA and retrieval failure, record conflicts, and
  `pa_turn_limit_reached` coverage.
- Prove that each served source is processed by its controlled tool and its
  accepted triple delta is merged before another source is selected.
- Verify every decision and compact served result appears in the PA UI while the
  expandable audit view preserves full records and internal `provenance`.
- Verify Phase 3.3 invokes only the controlled Phase 4 producer selected by its
  capability descriptor and has no planning, primitive-catalog, RA, CCA,
  forbidden Gazebo-state, evaluator, or execution dependency.
- Add revised tests proving that Phase 3.3 requests relevant remaining evidence
  instead of asking which gear-shaft location receives the Medium Gear.
- Test rejection of clarification or completion that is not backed by a
  persisted Phase 4.3 decision over the updated ABox. Prove that one-source
  retrieval does not imply interaction completion and that irrelevant approved
  sources are not retrieved merely to exhaust the inventory.

## Phase 4: PA grounding

This is the context-understanding stage and requires separate, bounded
implementation requests. The Phase 4.0 foundation is implemented but not yet
invoked by Phase 3. The target stage participates inside the Phase 3 retrieval
loop; it does not wait for a batch of Phase 3 evidence and does not assume that
retrieving a PDF or STL means its diagrams or geometry were understood. Phase
4.0 runs before the first source decision, Phase 4.1 or Phase 4.2 runs after each
relevant source is served, and Phase 4.3 reassesses the updated ABox after each
accepted delta.

### Phase 4.0: PPR-aligned context representation - implemented; Phase 3 integration pending

- Keep shared immutable PPR semantics, TBox profile validation, fingerprinting,
  and class-hierarchy queries in `ontology/ppr_tbox.py`. Keep PA-owned writable
  ABox initialization, evidence-delta validation, and persistence in
  `agents/pa/product_context.py`.
- Load the authoritative PPR OWL TBox with RDFLib and create one Turtle PA ABox
  per interaction before PA's first source decision. Keep the fixed TBox
  immutable and seed the ABox from the exact `product_requirement` as an
  unresolved request, not as grounded scene or product facts.
- Use the minimal semantic bridge `specification defines required feature` and
  `requested process realizes the same feature`. These assertions express the
  requested outcome, not proof that execution has achieved it.
- Treat the pasted OWL as a parser and mixed-graph test sample until the
  authoritative TBox is provided. It combines classes and properties with named
  process, resource, feature, and performance individuals. Do not copy those
  individuals, their `capableOf` or `realizes` assertions, or task-specific
  `requires` and `precedes` restrictions into an interaction ABox. Preserve all
  accepted ontology and project symbols exactly.
- Require the production TBox input to pass an explicit TBox profile before PA
  uses it: no runtime named individuals, no product-specific primitive recipe,
  and no expected primitive set or order. Parsing a mixed test fixture
  successfully is not acceptance of that fixture as the runtime TBox.
- Define one generic triple-delta contract for every controlled evidence tool.
  Each proposed assertion contains a runtime subject, a predicate from the
  immutable RDF/PPR vocabulary, a runtime IRI or literal object, and exact
  evidence refs. The same envelope records uncertainty, unresolved evidence
  needs, and refs to typed context records.
- Keep evidence tools side-effect free with respect to the ABox. The Phase 3.3
  orchestrator validates fixed symbols, evidence refs, recognition boundaries,
  and delta shape before atomically merging accepted assertions into the
  interaction ABox and persisting their assertion-level provenance alongside
  it.
- Generate all entity IRIs, instance counts, selected CAD correspondences, and
  context-artifact refs at runtime. Do not precreate an `observed_product_1`, an
  `observed_feature_1`, an expected CAD match, a pose-record path, or a
  product-specific completion template. A failed or uncertain match produces no
  factual assertion and remains unresolved.
- Give each controlled tool a small capability descriptor expressed with fixed
  ontology and typed-context symbols: `can_produce` describes the kinds of
  assertions or typed records it may return, and `may_require` describes
  permitted evidence it may need. These descriptor fields are adapter-owned
  routing metadata outside the TBox, not ontology predicates or required
  assembly fields.
- Use the PPR `product`, `feature`, `process`, `resource`, and `capability` types
  as the common vocabulary for the future PA-to-RA handoff and typed primitive
  input bindings. The TBox constrains meaning and vocabulary; it does not
  prescribe a product-specific completion graph, required source set, fixed
  retrieval order, or prove operational sufficiency. Do not treat the ontology
  as a separate ICRA contribution.
- Keep the ontology layers explicit. Generic classes, properties, domains,
  ranges, and restrictions belong to the immutable TBox. PA-created requested
  process, product, feature, and grounded-outcome individuals belong to the
  per-interaction PA ABox. Exact resource and primitive-offering individuals and
  their `capableOf` assertions belong to a selected-RA resource-catalog ABox
  snapshot introduced in Phase 7, not to the TBox and not to PA's evidence
  assertions. Do not add a new primitive class or predicate merely to force this
  split when the authoritative PPR vocabulary can type the offering as a
  `process`.
- Keep CAD correspondence, observation association, evidence provenance, and
  context-record links outside the ABox when the authoritative TBox has no
  predicate for them. Preserve them in typed context and delta records and add
  only their supported PPR consequence to the ABox. Do not create a runtime
  predicate or silently extend the ontology; an unsupported semantic relation
  remains outside the graph or unresolved.
- Validate every proposed RDF assertion against the vocabulary declared by the
  loaded TBox and the producer descriptor's authorization. Poses, calibration,
  tolerances, insertion axes, observations, and other geometry values may remain
  in dynamically created typed context records referenced by delta provenance.
- Keep the prototype local to Spec2Primitives. PA owns and mutates only its
  interaction ABox. A future RA resource ABox remains separately owned and is
  not implemented in Phase 4.0; PA and RA share only the immutable TBox. Do not
  add ResourceAgent, shared ProductAgent, Stardog, GraphRAG, learned graph
  retrieval, ontology-authored decomposition, or execution graph updates.
  Phase 7 may add a bounded, deterministic RDFLib projection over the two ABox
  snapshots after separate implementation authorization. RDFLib is the only
  ontology dependency added in Phase 4.0; pySHACL remains unapproved and absent.
- Keep resource, capability, primitive-offering, and `capableOf` assertions out
  of the PA interaction ABox. Those RA-owned assertions remain reserved for the
  selected-resource catalog snapshot in Phase 7.
- Let Phase 4.3 decide whether the grounded assembly contract is ready for PA's
  handoff. Later primitive binding and robot-local validation determine whether
  RA has enough context to compose and execute `primitive_steps`.

### Phase 4.0 verification

- Test TBox loading, independent per-interaction ABoxes, exact requirement
  preservation, and evidence sources for added facts.
- Test that the pasted mixed OWL parses but is not accepted as a production
  schema graph, and that none of its named individuals or instance assertions
  enters a new interaction ABox.
- Test one common validator and merger with document and scene deltas, including
  atomic rejection of unknown predicates, missing evidence refs, malformed
  objects, and unsupported assertions.
- Test dynamic entity IRIs and one, two, and many delta merges in different
  source orders without precreated instances, fixed source counts, or a
  modality-specific completion form.
- Test exact PPR type references and confirm that unsupported vocabulary and
  hidden expected answers never enter an interaction ABox.
- Test that CAD/observation-specific records remain outside the ABox when no
  loaded PPR predicate represents them, while their refs and evidence remain
  available to Phase 4.3 and later primitive binding.
- Verify that the ontology representation does not decide composition readiness,
  contact RA, or perform primitive composition or validation.
- Verify that the shared ontology package has no PA or RA dependency and that
  PA product-context code has no RA dependency.

### Phase 4.1: document-diagram VLM interpretation

- Implement the VLM as a controlled tool under `tools/document_evidence/`, not
  as another agent.
- Advertise its fixed-symbol `can_produce` and `may_require` descriptor. For one
  served document request, return a generic evidence-backed triple delta without
  mutating the ABox or declaring context understanding complete.
- Let the VLM interpret approved document diagrams only. Do not give it RGB-D
  observations or use it to propose metric geometry.
- Preserve the exact document `context ref`, page-level provenance, structured
  VLM output, and uncertainty. Uncertain interpretations remain unresolved and
  do not become factual assertions.

### Phase 4.2: CAD and RGB-D grounding

- Keep RGB segmentation, depth geometry, and CAD registration under
  `tools/rgb_d_cad_grounding/`.
- Advertise the tool's fixed-symbol `can_produce` and `may_require` descriptor.
  For the currently served approved evidence, dynamically detect zero or more
  instances, generate their IRIs, evaluate candidate CAD correspondences, and
  return the same generic triple-delta contract without mutating the ABox.
- Retain detailed geometry in dynamically created typed context records
  referenced by the assertion provenance. Propose RDF assertions only through
  vocabulary declared by the loaded TBox and authorized by the producer
  descriptor.
- Load approved CAD geometry for grounding; the existing filename, units,
  triangle count, and bounds summary alone does not establish a component match,
  receiving feature, or pose.
- Use only approved RGB, depth, camera calibration, document interpretation, and
  CAD evidence. Never use Gazebo model names, entity state, world contents,
  configured spawn poses, or evaluator data.
- Do not let VLM output independently establish `target_feature`, target pose,
  insertion axis, or tolerances.
- Produce a provenance-backed `target_feature`, target pose, insertion axis,
  tolerance, or other product fact only when the exact goal, selected evidence
  path, later assembly plan, or primitive contract requires it and the retrieved
  evidence supports it. This list is not a fixed completion form.
- Leave missing or contradictory evidence unresolved instead of inventing a
  value.
- Preserve the exact supporting `context ref` and provenance for each grounded
  value.
- Keep recognition inputs within the approved document, candidate CAD, RGB,
  depth, and camera calibration boundary.

### Phase 4.3: post-understanding decision

- Assess the current interaction ABox, typed context records, unresolved
  assertions, evidence status, and controlled-tool capability descriptors
  against the exact `product_requirement`. Phase 4.1 and Phase 4.2 may each run
  zero or more times; neither modality is mandatory by itself.
- Resolve a missing semantic fact or typed binding through the matching
  producer's `can_produce` and `may_require` descriptor. Do not infer that the
  primitive itself consumes a PDF, STL, RGB, depth, or calibration payload, and
  do not retrieve a modality merely because it is available.
- Derive the next unresolved semantic need from the current goal and ABox, then
  select one relevant producer and permitted source. Process sources dynamically
  until every retrieved source has been interpreted, every accepted delta has
  been merged, and no relevant evidence request is outstanding. Do not require
  corpus exhaustion.
- Assess PA handoff readiness only when no currently identified blocking
  product-level ambiguity remains. Enter clarification only when the remaining
  unresolved value is user intent; do not claim that PA handoff readiness proves
  composition or execution readiness.
- Use one existing structured decision:
  - an active `needed_context` with `context understanding complete: false`
    returns to Phase 3.2 for exactly that approved ref or fresh observation;
  - an active `clarification_question` with
    `context understanding complete: false` enters Phase 3.4 only when the
    unresolved value is user intent that system evidence cannot determine;
  - `needed_context: null` with `context understanding complete: true` enters
    Phase 3.5;
  - a VLM, retrieval, CAD, RGB-D, contradiction, or grounding failure is
    recorded; an approved alternative may be selected only when its advertised
    capability can resolve the same need, otherwise the interaction stops
    fail-closed and must not be converted into a user clarification question.
- Preserve the Phase 4 input, evidence refs, output, uncertainty, unresolved
  values, and decision in the ordered interaction record without hidden model
  reasoning.
- Allow the runtime to loop Phase 4.3 → Phase 3.2 → Phase 4.1 or Phase 4.2 →
  delta validation and merge → Phase 4.3 when more permitted evidence can
  resolve the missing context. Phase numbering does not require a one-way
  runtime sequence.

## Phase 5: PA assembly plan

This phase requires a separate implementation request.

- Have PA convert the grounded product requirement into an assembly plan.
- Show PA building the assembly plan from grounded evidence in the PA card and
  ordered interaction record.
- Treat each nominal assembly-plan entry as a robot-independent task-transition
  contract containing the exact required outcome, grounded product and scene
  bindings, typed context refs, constraints, provenance, and version. It must
  not contain primitive names, primitive ordering, or a preauthored executable
  decomposition.
- Bound the task-transition contract to conditions and constraints supported by
  grounded evidence. It is a versioned goal and binding record, not a complete
  PDDL problem, exhaustive transition model, primitive-reachability proof, or
  closed-world declaration that every omitted fact is false.
- Project only the relevant ABox assertions and typed context refs into this
  task contract. Keep the downstream RA composer independent of RDFLib and the
  internal ABox representation.
- Give each task-transition contract the exact requested-process individual,
  grounded-outcome individual, PA ABox version and fingerprint, and authorized
  typed context refs needed for the later read-only ontology projection. Do not
  add a relation from that high-level process to an expected primitive set.
- Keep the assembly plan separate from robot-specific `primitive_steps`. Do not
  have PA author, prescribe, or complete `primitive_steps`.
- Do not express the plan as calls to the prior predefined composite functions
  `move_to_pick_location`, `pick_part`, `move_loaded_to_destination`, or
  `place_part`, or to an equivalent product-specific callable. PA specifies the
  required robot-independent transition; it does not select a function whose
  internal primitive program already solves that transition.
- Version the grounded assembly contract and plan. When RA-triggered
  `missing_context` adds product or scene knowledge, revalidate the plan before
  the next RA update; revise it when the new fact changes product-level steps,
  or explicitly preserve it when the update only supplies a missing binding.

## Phase 6: PA-to-RA communication

This phase requires a separate implementation request.

- Add a Spec2Primitives-owned RobotAgent adapter under `agents/ra/` for the
  structured exchange between PA and the selected RA.
- Consume the configured RA roster and exact `resource_jid` already supplied by
  the inherited upstream selection/allocation authority. Persist the bounded
  routing snapshot supplied with that decision, then unicast to the exact RA.
  Do not implement broadcast discovery or a new allocation optimizer. The
  routing summary is not a substitute for the selected RA's fresh authoritative
  state and primitive catalog in Phase 7.
- Define one versioned composition-task envelope for both origins. A nominal
  envelope contains one validated Phase 5 task-transition contract; a recovery
  envelope contains an already selected, validated, and resource-assigned
  recovery event imported through a read-only adapter under `agents/pa/`.
  Preserve the exact upstream task or event identifier, task origin, required
  outcome, requested-process and grounded-outcome refs, product and target
  bindings, PA ABox fingerprint when applicable, typed context refs, provenance,
  version, and selected `resource_jid`.
- Send relevant grounded assertions and typed context refs, not raw PDF, CAD,
  RGB, or depth evidence for RA to reinterpret. Do not include primitive names,
  primitive ordering, a capability decomposition, or completed
  `primitive_steps`.
- Reject a nominal or recovery envelope that contains an expected primitive
  sequence, task-specific capability decomposition, stored recipe, or completed
  `primitive_steps` presented as the answer.
- Reject an envelope that substitutes a predefined composite function such as
  `move_to_pick_location`, `pick_part`, `move_loaded_to_destination`, or
  `place_part` for the required transition. Those functions belong to the
  prior-work planning abstraction and are unavailable to the composer.
- Make the exchange bidirectional. Let RA return a structured `missing_context`
  result containing the unbound semantic input, owning authority, primitive or
  validation reason, and supporting record refs.
- Route only product or scene knowledge gaps back to PA. For a nominal task, PA
  resumes the same interaction ABox and Phase 3.3/Phase 4 loop, then Phase 5
  revalidates or revises the task contract. For a recovery task, preserve a
  binding-only update when the selected event remains valid; if the new fact
  changes its identifier, required outcome, semantics, feasibility, or assigned
  `resource_jid`, stop that transfer attempt fail-closed. The inherited framework
  may produce a newly validated event outside Spec2Primitives and import it as a
  new transfer input; the adapter never invokes the recovery selector or CCA.
  Robot state, capability, primitive-catalog, IK, collision, and trajectory gaps
  remain RA-owned.
- Keep upstream recovery-event feasibility records distinct from downstream
  primitive-program validation. Importing the selected recovery event does not
  make recovery event generation, selection, allocation, DES/CCA reasoning, or
  safety planning a Spec2Primitives contribution.
- Display exchanged messages in the corresponding live PA and RA cards and the
  ordered interaction record.
- Keep shared PA and RobotAgent implementations unchanged.
- Preserve exchanged inputs, outputs, and provenance as reviewable context
  records.

## Phase 7: RA context retrieval and primitive composition

This phase requires a separate implementation request.

- Retrieve fresh state from the selected RA and its resource-owned composition
  catalog containing exactly the eight user-defined primitive symbols. Reject a
  missing, extra, renamed, or substituted primitive instead of adapting the
  catalog silently.
- Require the composition catalog to be primitive-only. Reject a predefined
  composite function, product-specific macro, stored primitive program, or
  callable wrapper that already fixes primitive selection or order. In
  particular, the composer must not receive `move_to_pick_location`,
  `pick_part`, `move_loaded_to_destination`, or `place_part`.
- Verify from the selected-resource capability snapshot that no separate
  predefined composite function available to the interaction directly realizes
  the required transition. If one exists, classify the case as
  prior-function reuse and exclude it from evidence for dynamic primitive
  composition.
- Materialize the returned catalog as an immutable, versioned resource-catalog
  ABox snapshot for this composition attempt. Represent the selected resource
  and each exact primitive offering as individuals using only authoritative PPR
  types, with `selected_resource capableOf primitive_process_interface` as the
  availability relation. PA must not author these assertions.
- Use a Spec2Primitives-owned read-only projection adapter to retrieve the
  current task process and grounded outcome, relevant product and feature
  individuals from the versioned PA ABox, the selected resource, and all eight
  of its primitive-process interface individuals from the resource-catalog
  ABox. Include authorized typed context refs and evidence provenance. Persist
  the exact projection and both source fingerprints for every RA turn.
- Explicitly reject any projection containing a task-to-primitive `requires` or
  `precedes` recipe, an assertion that the selected resource is directly
  `capableOf` the requested high-level assembly process, or another relation
  that supplies the expected primitive set or order. The semantic mismatch
  between the grounded requested process and the available lower-level process
  interfaces is intentional and is resolved only by the RA LLM candidate.
- Treat each catalog entry as one semantic executable interface with its exact
  primitive symbol and operation description, typed parameters and returned
  status or evidence, invocation binding, truthful resource limits, applicable
  evaluator endpoints, and only the resource-local state conditions or effects
  that are explicitly modeled. An omitted condition or effect is unmodeled; RA
  and the validators must not infer that it is satisfied.
- Treat the fresh snapshot and complete exact-eight catalog as mandatory initial
  RA context, not as a complete PDDL domain. Supply no task-specific domain or
  problem file, exhaustive action model, closed-world state, expected sequence,
  or mapping from the requested outcome to a primitive decomposition.
- Show RA building context from the retrieved `context ref` in the RA card and
  ordered interaction record.
- Give the selected RA LLM the versioned composition-task envelope, required
  outcome, fresh initial state, exact eight semantic primitive interfaces and
  partial local executable contracts, the read-only task-relevant ontology
  projection, relevant typed context refs, resource limits, and any findings
  from the previous candidate. Do not give it hidden expected answers or a
  completed decomposition. The ontology projection supplies shared meaning;
  the semantic interface cards supply callable behavior details that a process
  class or primitive name alone cannot provide.
- After loading the three mandatory prerequisites, use the same demand-driven
  control pattern as PA. Each RA turn either authors one complete candidate,
  identifies one exact selected-resource `needed_context`, returns one
  cross-authority `missing_context`, or returns bounded `cannot_compose`. Serve
  and persist one requested result, then reassess before another request. Do not
  preload all RA records or follow a fixed resource-context retrieval order.
- Let an unbound input on a primitive interface or candidate identify the
  semantic need, not the evidence modality. For a product or scene input, RA
  returns that exact typed need to PA; PA selects the matching controlled
  producer and its approved evidence. RA must not request a raw PDF, STL, RGB,
  depth, or calibration payload directly.
- Have the RA LLM author one complete candidate `primitive_steps` program by
  selecting, ordering, parameterizing, and binding only those eight primitives.
  This agentic step is the composer. No deterministic composer, backward or
  forward search, validator, stored recipe, product-specific macro, or
  capability decomposition may create or complete the candidate.
- Use the dynamically retrieved PPR projection only as semantic and binding
  context, then use the primitive contracts plus typed context refs to bind
  runtime inputs. Treat any missing required binding as
  composition-not-ready context instead of inventing a value or claiming
  execution readiness. The RA LLM may reason over the projection, but neither
  OWL inference nor the projection adapter may author `primitive_steps`.
- Have RA perform the binding. If a primitive contract exposes a missing
  product or scene input, RA sends `missing_context` to PA; PA grounds additional
  knowledge through controlled evidence tools. RA recomposes or retries binding
  only after the task-origin-specific upstream authority returns the versioned
  assertion, typed context refs, and task update. PA does not bind robot
  primitive parameters.
- If the missing input is RA-owned, retrieve or validate it inside the RA loop
  without routing raw robot state through PA.
- Feed nominal and recovery envelopes through the exact same composition
  entrypoint and prompt policy. `task_origin` is audit and upstream-routing
  metadata; it must not select a recovery-specific composer or recipe.
- Allow the RA to return a bounded, structured failure when it cannot produce a
  candidate. Agentic generation is not assumed to be correct or complete on the
  first attempt, and an unvalidated candidate must never proceed to execution.
- Preserve every retrieved `context ref`, candidate `primitive_steps`, and
  revision, every `missing_context` exchange, and every ABox update in the
  active interaction under `contexts/`.

## Phase 8: RA validation and revision

This phase requires a separate implementation request.

Validate each candidate without mutating it through five stages:

1. PA validates only the composition-task envelope and evidence refs; RA
   validates candidate schema, exact catalog membership, parameters, bindings,
   and context refs.
2. Backward modeled-condition coverage traverses only the proposed steps from
   the modeled required conditions and checks that declared effects and their
   recursively introduced preconditions have causal support from earlier
   candidate effects or, at the initial-state boundary, fresh state or evidence.
   It does not search the catalog, select a primitive, complete a plan, or prove
   reachability beyond the represented contract surface.
3. Forward resource-state validation starts from the fresh RA snapshot, checks
   every declared precondition and data dependency in order, and projects only
   guaranteed local effects. It does not interpret an omitted condition or
   effect as satisfied.
4. RA-owned primitive evaluators perform physical, operational, IK, collision,
   grasp, release, and trajectory checks where applicable.
5. A separate deterministic task-outcome validator checks only explicitly
   represented projected-outcome constraints and returns `supported`,
   `unsupported`, or `unmodeled`. Only `supported` may pass. `unmodeled` stops
   fail-closed; the validator does not infer missing semantics, search for
   primitives, or accept a final-state assertion authored by the RA LLM.

- Keep primitive effects truthful and local. For example, `release_part` may
  establish direct release and custody effects, but it must not establish that
  the part is on the target or assembled unless a separate outcome check has
  the required pose, process, and evidence support.
- Treat backward and forward results as bounded checks over the declared
  contract surface, not as proof over a complete symbolic world model. A
  causally or safety-critical intermediate requirement must appear in a
  primitive contract, resource invariant, primitive-specific evaluator, or
  physical/outcome validator to be detectable. Do not claim that an unmodeled
  requirement was validated.
- No validation stage may interpret an omitted precondition, effect, invariant,
  or evaluator as satisfied. Distinguish `unmodeled` coverage from a modeled
  contradiction and from an execution failure.
- Preserve a candidate fingerprint before and after every validator so
  non-mutation is auditable. No validator may propose, add, remove, reorder,
  parameterize, or repair a primitive.
- Return concrete categorized findings to RA when a candidate is rejected. A
  newly exposed product or scene binding gap may use the Phase 6 RA-to-PA
  `missing_context` path; other failures remain RA-local revision evidence. The
  RA LLM, and only the RA LLM, authors the next complete candidate.
- Accept only the unchanged candidate that passes every applicable check. Stop
  fail-closed when the bounded revision budget is exhausted.
- Show validation results and revisions in the RA card and ordered interaction
  record.
- Store validation traces and revision histories in the active interaction
  under `contexts/`.

### Phase 7 and Phase 8 verification

- Test that only the exact eight primitive symbols can appear and that no
  template, decomposition, validator, or fallback code creates candidate steps.
- Test that the task envelope, composition prompt, resource-catalog snapshot,
  and execution dispatch contain none of `move_to_pick_location`, `pick_part`,
  `move_loaded_to_destination`, or `place_part`, and contain no equivalent
  predefined composite callable or hidden primitive expansion.
- Test admission explicitly: a requirement with an available matching composite
  function is labeled prior-function reuse and is not counted as a
  Spec2Primitives composition; the nominal and recovery study inputs have no
  such function and expose only the exact eight primitives.
- Test that each catalog item contains its semantic executable interface and
  truthful partial local contract while the combined input contains no PDDL
  domain/problem, exhaustive action model, expected sequence, or task-specific
  primitive decomposition.
- Test the TBox/ABox split: generic PPR vocabulary remains in the TBox; PA
  evidence creates task/product/outcome individuals only in its interaction
  ABox; selected-resource and exact-eight primitive offerings appear only in
  the versioned resource-catalog ABox; and RA receives a read-only projection
  without mutating either graph.
- Verify that the projection includes the high-level requested process and
  outcome plus all eight available primitive-process interfaces, but contains
  no `requires`, `precedes`, direct high-level `capableOf`, expected sequence,
  or other task-to-primitive decomposition leak.
- Test the same composition interface, prompt policy, eight-symbol catalog
  interface, candidate schema, validators, and revision budget with nominal and
  recovery-origin envelopes. Hold the exact eight primitive symbols and contract
  schema fixed while preserving each selected RA's truthful resource-owned
  parameter limits and evaluators.
- Test rejection of a missing modeled intermediate primitive even when the RA
  output claims the requested final state. Test an equivalent accepted sequence
  whose intermediate preconditions are established in order.
- Test that `release_part` alone cannot establish a placed or assembled outcome,
  and keep predicted pre-execution outcome separate from observed
  post-execution confirmation.
- Test invalid parameters, unresolved refs, stale state, unsupported backward
  dependencies, forward precondition failures, IK, collision, grasp, release,
  trajectory, and outcome rejection with categorized findings.
- Test validator non-mutation with candidate fingerprints and prove that only a
  subsequent RA agentic turn can structurally revise the candidate.
- Test `missing_context` ownership and task-origin-specific routing without
  allowing PA to bind RA primitive parameters or the ontology to decide
  composition readiness.
- Test the PA and RA retrieval loops with the same retrieve, validate, persist,
  and reassess invariant while enforcing their different authorities. Verify
  that neither follows a fixed all-source order and that exhaustion of a turn or
  revision budget produces failure rather than readiness.
- Test dependency-driven source selection: a need already supported by current
  context triggers no retrieval; a document-producible semantic relation selects
  only approved document evidence; a CAD/RGB-D-producible typed binding selects
  only that producer's declared evidence; and an unknown need fails closed.
- Verify that primitive interfaces name grounded inputs rather than raw evidence
  modalities and that RA never receives or requests PDF, STL, RGB, depth, or
  calibration payloads.
- Remove one causally required item from the represented contract and evaluator
  surface and verify an `unmodeled` fail-closed result rather than silent
  acceptance or validator-authored repair.

## Phase 9: simulation execution and evaluation

This phase requires a separate implementation request and explicit execution
authorization.

- Enable RA simulation execution only after the earlier contracts and
  validators are tested.
- Recheck fresh robot state before simulation dispatch.
- Keep robot safety and runtime authority with RA.
- Show RA as executing only while a live simulation dispatch is active.
- Confirm the realized task outcome from permitted execution and observation
  evidence after execution. Keep this observed outcome distinct from the Phase
  8 pre-execution prediction and from any final-state claim in an LLM response.
- Keep the ground-truth evaluator separate from recognition and expose ground
  truth only after the prediction is finalized.
- Record simulation results and execution logs in the active interaction under
  `contexts/`. Store ground truth and post-prediction evaluation under
  `evaluations/`.
- Report fixture, replay, contract validation, simulation, and physical
  execution as distinct evidence.
- Evaluate the primary nominal case from the raw requirement and the recovery
  transfer case from the imported selected recovery event with the same RA
  composer, exact eight primitive symbols and contract schema, model and prompt
  policy, validator stack, budgets, and execution boundary. Preserve each
  selected RA's truthful contract values and evaluators. Recovery-specific
  information may change the input contract but not the composition mechanism.
- For each reported composition case, audit and preserve the evidence that no
  matching predefined composite function was available to the selected RA.
  Treat a prior RCIM-style planner over predefined composite functions as a
  boundary baseline: its unmatched result establishes why primitive
  composition is invoked, but by itself does not establish that the proposed
  composer is superior.
- Define nominal success as realization of the requested assembly outcome.
  Define recovery success as realization of the selected event's declared
  successor or nominal-reentry condition. Both also require an accepted Phase 8
  primitive program and successful simulation execution; upstream recovery
  candidate selection alone is not composition success.
- Compare at minimum: an equal-input symbolic composer limited to the
  machine-readable fields in the same partial interfaces; bounded black-box
  sequence search using the same validator access and query budget; LLM-only
  candidate generation without validator-guided revision; and the full RA
  agentic-composition plus validation-revision method. Report represented-domain
  coverage, abstention, and failure separately. A classical planner supplied
  with a separately engineered complete PDDL domain may be reported as an
  extra-information oracle, not as an equal-input baseline. Do not weaken a
  baseline solely to make the agentic method appear necessary; failure outside
  its represented domain is not evidence that an LLM is universally necessary.
- Report first-pass candidate validity, accepted-composition rate, missing-step
  rejection and false-acceptance rates, context requests, revision count and
  convergence, physical feasibility, execution success, observed task outcome,
  latency, and generalization to unseen initial states separately by task
  origin.
- Interpret the comparison as evidence about agentic composition under the
  evaluated partial symbolic model. Do not claim that LLMs are universally
  necessary, that a complete symbolic planning domain could not solve the task,
  or that the composer is complete or optimal.
- Keep physical-robot execution optional and separately authorized. Simulation
  results must not be described as physical execution evidence.

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
