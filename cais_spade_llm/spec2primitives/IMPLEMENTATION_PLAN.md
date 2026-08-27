# Spec2Primitives Implementation Plan

Phase 0, the Phase 0.1 operator shell, Phase 1, Phase 1.1, Phase 1.2, the Phase
2 PA interaction UI, Phase 2.1, Phase 3.1, Phase 3.2, the Phase 3.3
contract-first ontology integration, Phase 3.4, Phase 3.5, Phase 4.0, Phase
4.1, Phase 4.2A
preprocessing, Phase 4.2B1 minimal automatic RGB-D segmentation, and Phase
4.2B2A CAD-size candidate association, the simple remaining Phase 4.2B2
camera-frame pose increment, simple camera-to-robot frame conversion, and the
pre-RA Phase 4.3 grounding runtime are implemented. Phase 3.1
now initializes Phase 4.0 before PA selects its first source and rejects
first-turn clarification. Phase 3.3 now orchestrates numbered retrieval,
controlled interpretation, validated ABox deltas, and persisted Phase 4.3
decisions without assessing raw evidence summaries. Phase 4.3 now evolves and
persists a robot-independent `TaskTransitionDraft`, computes exact
`ContextNeed` values from a validated `ProductContextView`, and selects only an
authorized output-capable producer. The production runtime and its read-only
ontology-grounding UI are enabled only when an authoritative runtime TBox and
`OPENAI_API_KEY` are configured; otherwise the interaction fails closed. The
separate Phase 4.1 UI diagnostic has the same configuration boundary.
Phase 5 onward and later Phase 4.2 grounding work remain future work
requiring separate, explicitly scoped implementation requests. Phase numbers
identify capabilities, not a required one-way runtime order. This plan does not
authorize an end-to-end implementation.

`PA` refers to ProductAgent and `RA` refers to RobotAgent throughout this plan.
Only PA and RA participate in the current Spec2Primitives roadmap.

## Proposed-versus-implemented status

| Phase | Status | Current boundary |
|---|---|---|
| Phase 0 and Phase 0.1 | implemented | Isolated package and no-hardware NIST operator scene. |
| Phase 1, Phase 1.1, and Phase 1.2 | implemented | Approved exact-ref retrieval plus stored and request-scoped live RGB-D observations. |
| Phase 2 and Phase 2.1 | implemented | PA UI, configurable turn limit, live PA turn counter, transcript, compact evidence summaries, and complete audit records. |
| Phase 3.1 | implemented; production ontology configuration pending | Exact requirement intake loads an injected schema-only TBox, initializes the ABox before PA's first request, exposes the unresolved view and approved evidence types, and rejects first-turn clarification. |
| Phase 3.2 | implemented | One persisted document, CAD, or live-observation request is served exactly and recorded without fallback evidence. |
| Phase 3.3 | contract-first integration implemented | Each served result is routed through an authorized output-capable producer descriptor, its delta is validated and merged, and only a persisted Phase 4.3 decision can request more evidence, clarify, or complete. |
| Phase 3.4 and Phase 3.5 | implemented | Exact user-intent replies and cancellation resume the same interaction; completion persists one fully referenced, tamper-checked `PAContextGroundingCompletion` that is ready only for Phase 5. |
| Phase 4.0 | implemented and wired through injected contracts | Shared RDFLib TBox loading, independent PA ABoxes, safe reload, compact views, output-capable descriptor routing, and atomic evidence-backed delta validation are used by Phase 3. |
| Phase 4.1 | implemented as a separate diagnostic | All six approved NIST PDF pages are rendered and sent through an injected OpenAI vision boundary in one structured request; its compiled delta is persisted and accepted only through the shared ABox validator. |
| Phase 4.2A | implemented as a separate diagnostic | Exact approved binary STL meshes and fresh validated four-camera RGB-D bundles become atomic typed geometry records and assertion-free deltas; correspondence and pose remain not evaluated. |
| Phase 4.2B1 | implemented supporting infrastructure | An observation-only entrypoint automatically captures, validates, preprocesses, and minimally segments four camera-local RGB-D views with fixed internal parameters. The UI is status-only. |
| Phase 4.2B2A | implemented supporting infrastructure | One exact preprocessed approved CAD record is compared with validated segmented candidates by its two largest principal dimensions. A unique size match yields only a candidate center in its camera optical frame. |
| Simple remaining Phase 4.2B2 pose increment | implemented supporting infrastructure | One intact size correspondence is refined across loose source candidates with deterministic principal-axis multistart and trimmed ICP. It persists an accepted, ambiguous, or rejected camera-frame pose without claiming a robot pose or pick point. |
| Simple camera-to-robot frame conversion | implemented supporting infrastructure | One accepted camera-from-CAD transform is composed with one injected, hash-validated camera-to-robot calibration for an exact caller-selected target frame. Ambiguous and rejected poses remain coordinate-free. |
| Later Phase 4.2 grounding | partially implemented in the pre-RA producer chain | Approved CAD and fresh observations can be preprocessed on demand, then segmented, associated, and estimated in a camera frame only for a current need. No cross-camera transform or complete target grounding exists. |
| Phase 4.3 production grounding | implemented pre-RA | `TaskTransitionDraft`, `ContextNeed`, `TypedContextBinding`, `ProductContextView`, and `GroundingProducerDescriptor` drive document and typed-geometry grounding, persistence, progress checks, and read-only UI inspection. |
| Phase 5 | not implemented | No `TaskTransitionContract` or assembly-plan handoff record exists. |
| Phases 6--9 | not implemented | No RA adapter, resource-catalog ABox, `CompositionContextBundle`, `PrimitiveProgramDraft`, `MissingContextBatch`, primitive composer, validator stack, or execution path exists. |

The Phase 4.0 foundation separates shared immutable TBox semantics from the
PA-owned writable product context. It loads a caller-supplied TBox, initializes
an independent interaction ABox from the exact requirement, and validates and
persists generic evidence-backed triple-delta contracts. The pasted OWL serves
only as an RDF/XML parser and mixed-graph test
fixture until the authoritative TBox is provided: it contains both schema
axioms and named individuals, so its ABox facts never become runtime task facts.
The next separately authorized workflow capability is Phase 5.
Production grounding requires an authoritative schema-only TBox;
the mixed and minimal fixtures remain test-only.
Controlled tests of retrieval and observation capture do not constitute
document-diagram understanding, metric grounding, Gazebo task execution, or
physical execution.

The runtime now implements `ContextNeed`, `GroundingProducerDescriptor`,
`TypedContextBinding`, `ProductContextView`, `TaskTransitionDraft`, and
`PAContextGroundingCompletion`. The
planned contracts `TaskTransitionContract`, `CompositionContextBundle`,
`PrimitiveProgramDraft`, and `MissingContextBatch` remain future boundaries.

## ICRA implementation scope

The central implementation contribution is one RA-owned agentic composition
loop, not a symbolic primitive planner:

```text
grounded task transition for which the selected RA has no matching
predefined composite function, or selected recovery event with the same gap
        + required outcome and grounded context refs
        + fresh selected-RA state
        + the complete selected-RA-authoritative primitive-only catalog
          of semantic executable interfaces
        + partial local executable contracts and validator endpoints
        ↓
RA LLM authors a structural PrimitiveProgramDraft
        ↓
binding preflight resolves RA-owned inputs locally
        + batches product/scene ContextNeeds to PA when required
        ↓
PA returns a versioned CompositionContextBundle
        ↓
RA LLM authors one fully bound candidate primitive_steps program
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
exposes only its complete current primitive-only catalog. The accepted
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
truthful. The catalog interfaces do not collectively supply a complete symbolic
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

Both cases must use the same RA composition entrypoint, catalog contract,
prompt policy, context-request protocol, candidate schema, validator sequence,
revision loop, and execution boundary. The catalog symbols and cardinality may
vary by selected RA and version, but each attempt pins the complete exact
snapshot and fingerprint. A stored nominal program,
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
complete set of primitive-process interface individuals in the pinned catalog.
Their executable contract
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
broadcast. Recovery event generation and selection, DES reasoning, fault
diagnosis, ontology design, a complete symbolic planner, controller design, and
proof of globally complete or optimal composition are not ICRA implementation
contributions.

## Shared PA and RA dynamic context-retrieval pattern

The stable mechanism is consumer-driven rather than modality-driven:

```text
required consumer inputs
        - valid current authority-owned context
        = ContextNeeds
        ↓
select one authorized GroundingProducerDescriptor per resolvable output
        ↓
retrieve or compute auditable evidence through the owning authority
        ↓
validate, fingerprint, persist, and reassess
```

The TBox defines legal PPR meaning but cannot decide operational completeness or
which sensor, document, CAD, or calibration source to use. The planned
`ProductContextView` combines the current PA ABox with validated
`TypedContextBinding` summaries, uncertainty, unresolved needs, attempted
evidence, status, frames, freshness, hashes, and provenance. A
`GroundingProducerDescriptor` advertises which semantic or typed outputs one
controlled producer can establish and which authorized evidence dependencies it
uses. This registry is application configuration, not an ontology predicate,
task recipe, or model-authored route.

PA owns product-intent and scene grounding:

- Before allocation, PA evolves a `TaskTransitionDraft`. It dynamically selects
  only the approved evidence needed to resolve currently blocking product,
  process, feature, outcome, constraint, or task-binding needs.
- Existing Phase 3.2 document, CAD, and observation retrieval remains exactly
  one source per operation. This is an internal audit boundary, not a required
  modality order or cross-agent message limit.
- PA merges only evidence-backed product facts and retains numeric poses,
  calibration, tolerances, and observations in typed records.
- `context understanding complete` means every currently blocking PA-owned need
  referenced by the draft is satisfied for Phase 5. It does not mean every
  later primitive input exists.
- PA never retrieves the primitive catalog, robot state, IK, collision,
  trajectory, or execution evidence and never authors `primitive_steps`.

RA owns embodiment grounding and composition readiness:

- RA retrieves a read-only task/resource projection, a fresh selected-resource
  snapshot, and the complete current selected-RA-authoritative primitive-only
  catalog. Every offering in the pinned snapshot and its exact symbol is
  included, regardless of runtime cardinality; no task-filtered subset reveals
  an expected solution.
- RA LLM authors a structural `PrimitiveProgramDraft`. A deterministic binding
  preflight inspects that draft and the primitive-interface contracts to gather
  every currently unbound input without selecting, inserting, reordering,
  parameterizing, or repairing a step.
- Robot state, resource limits, IK, collision, grasp, release, trajectory,
  operational, and execution gaps remain RA-owned. Product or scene gaps are
  deduplicated by semantic need plus frame and freshness into one
  `MissingContextBatch` for that round.
- PA may resolve one batch through several internal single-source operations,
  then returns one new versioned `CompositionContextBundle`. RA refreshes
  affected local context and alone authors the next structural draft or fully
  bound candidate.
- RA is composition-ready only when every parameter is bound to an authorized
  value or context ref and one unchanged candidate passes every applicable
  local-contract, physical, and projected-outcome validator.

There is no fixed semantic PA/RA batch-round count. Another round is permitted
only when the prior round adds accepted context, reclassifies or resolves a
need, or leads RA to author a structurally different draft. An identical request
against unchanged inputs, an unsupported need, ambiguous or unavailable
evidence, a repeated failed route, or no new accepted binding is `no_progress`
and stops fail-closed. Operational cancellation, deadlines, and tool budgets
remain valid failure boundaries; their exhaustion never implies readiness.

Primitive interfaces and task drafts request grounded outputs such as an
authorized product binding, pose-record ref, target ref, or constraint record;
they never request raw PDF, STL, RGB, depth, or calibration payloads from RA.
The controlled PA producer may use those permitted sources. The exact CAD is
caller- or corpus-authorized rather than inventory-searched, and calibration is
injected by an approved environmental authority and matched by frame and time.
If no authorized producer advertises a missing output, the system stops instead
of asking the ontology or LLM to invent a source.

OWL inference remains open-world. A small SHACL or equivalent boundary check may
flag an already identified required fact as missing, but it must not encode a
static all-modality checklist or expected primitive sequence. Adding pySHACL
remains separately authorized.

## Requirement-to-action workflow

The nominal case follows this end-to-end path without a matching predefined
composite function, stored nominal task program, or primitive recipe:

```text
exact product_requirement
        ↓
Phase 4.0 loads the fixed TBox and initializes one interaction ABox
        ↓
PA evolves a TaskTransitionDraft over the current ProductContextView
        ↓
PA resolves currently blocking ContextNeeds through output-capable producers
        ↓
PA validates and persists evidence-backed facts, then reassesses
        ↺ each producer retrieval remains one auditable source operation
        ↓
PA projects a robot-independent TaskTransitionContract
        ↓
inherited allocation supplies one selected resource_jid; PA unicasts the contract
        ↓
selected RA retrieves a task-relevant ontology projection, fresh state,
and the complete current primitive-only catalog snapshot
        ↓
RA LLM authors a structural PrimitiveProgramDraft
        ↓
binding preflight gathers every currently unbound input
        ├── RA-owned gaps resolve locally
        └── product/scene gaps form one MissingContextBatch
                                      ↓
                         PA runs required controlled producers
                                      ↓
                         versioned CompositionContextBundle
        ↓
RA LLM authors a fully bound primitive_steps candidate
        ↓
non-mutating declared-contract, resource, physical, and outcome checks
   ↙ new product/scene gap   ↓ robot-local finding         ↘ accepted unchanged
new batched PA round only  RA retrieves/reasons and          fresh-state recheck
after measurable progress authors a new candidate                    ↓
        └──────────────────────────↺                            RA execution
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
complete current primitive-only catalog snapshot are mandatory initial
prerequisites. Every later RA retrieval is selected from the current candidate
need or validator finding. Validators and ontology reasoning never create a
primitive step.

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
  not perform grounding, planning, RA, or robot execution.

## Phase 3: PA context retrieval and clarification

Phase 3.1, Phase 3.2, the contract-first Phase 3.3 ontology integration, Phase
3.4 clarification resumption, and the Phase 3.5 completion boundary are
implemented together with the production Phase 4 grounding producers.

In the target runtime, Phase 3 retrieval and Phase 4 grounding support PA while
it evolves a `TaskTransitionDraft`. Phase 4.0 loads the fixed TBox and
initializes the interaction ABox immediately after requirement intake. PA
computes the currently blocking task-level `ContextNeed` values from the
consumer inputs required by that draft minus valid current `ProductContextView`
bindings. A `GroundingProducerDescriptor` maps each need to an authorized
producer, which may select approved document, exact approved CAD, RGB-D,
calibration, or an existing record. Phase 3.2 still serves only one source per
producer operation so each result is auditable. That internal boundary is not a
one-source or fixed-order limit for the interaction, and it is distinct from the
later batched PA-to-RA exchange.

PA asks for clarification only after relevant permitted evidence and matching
controlled tools cannot resolve the remaining user-intent question. PA treats
user expertise as unknown. The user is authoritative about the requested goal,
but user factual claims never override contradictory approved evidence.
Unsupported factual claims remain unresolved. An unresolved component
location, receiving feature, pose, diagram association, CAD association, or
current arrangement is a system evidence problem, not a user clarification
question.

### Phase 3.1: product requirement intake and first `needed context` decision - implemented with contract-first ontology integration

- Add a Spec2Primitives-owned adapter under `agents/pa/` using composition with
  the shared ProductAgent. Do not subclass ProductAgent or LlmAgent.
- Preserve the implemented package-local entrypoint:

  ```python
  async def start_pa_context_interaction(
      product_agent: ProductAgentContextRuntime,
      interaction_root: Path,
      product_requirement: str,
      *,
      ontology_config: PAOntologyConfig | None = None,
      grounding_runtime: ProductContextGroundingRuntime | None = None,
  ) -> dict[str, object]:
      ...
  ```

- Expose only the inherited public `ask_llm_structured(...)` operation through
  `ProductAgentContextRuntime`. Do not call `ProductAgent.setup()`, start SPADE
  behaviours, build a plan, or contact RA.
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
  compact unresolved product-context view, approved evidence types, permitted
  request shapes, and the option to request one fresh
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
- The implementation rejects first-turn `clarification_question` as a terminal
  shortcut and requires a permitted evidence request before Phase 3.4 can ask
  the user.
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
  grounding, planning, RA, or robot execution.

### Phase 3.3: configurable ontology-backed context orchestration - contract-first integration implemented

- Record `max_pa_turns` and `live_observation_timeout_sec` exclusively in
  `interaction_record/pa_context_settings.json`; count `turn_0001` in the limit.
- Make Phase 3.3 the outer orchestration loop, not a second decision authority.
  After the Phase 3.1 bootstrap request, it accepts only a persisted Phase 4.3
  decision over the current ABox; it must not declare sufficiency or select a
  different source from raw served summaries.
- Validate the returned unresolved semantic need and selected exact approved
  `context_ref` or fresh observation, then dispatch only the application-owned
  producer route for that evidence type. Routing does not define hard-coded
  assembly slots or expected answers.
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
- The production UI loads a caller-configured authoritative TBox and production
  producer runtime when model configuration is present. It returns
  `grounding_unavailable` before PA evidence selection when that configuration
  is absent. Controlled tests use schema-only fixtures and deterministic model
  responses rather than claiming live perception or model validation.

### Phase 3.4: user clarification — implemented

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

### Phase 3.5: `context understanding complete` handoff — implemented

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
- Persist one `PAContextGroundingCompletion` only after independently reloading
  its decision, draft, source and current product-context views, typed-record
  hashes, and answered clarification refs. The UI consumes that validated
  record rather than treating a bare model decision as completion.

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
- Verify that Phase 3.1 does not call `setup()`, planning, RA, the resolver,
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
  planning, RA, forbidden Gazebo-state inputs, or robot execution.
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
  capability descriptor and has no planning, primitive-catalog, RA,
  forbidden Gazebo-state, evaluator, or execution dependency.
- Add revised tests proving that Phase 3.3 requests relevant remaining evidence
  instead of asking which gear-shaft location receives the Medium Gear.
- Test rejection of clarification or completion that is not backed by a
  persisted Phase 4.3 decision over the updated ABox. Prove that one-source
  retrieval does not imply interaction completion and that irrelevant approved
  sources are not retrieved merely to exhaust the inventory.

## Phase 4: PA grounding

This is the context-understanding stage and requires separate, bounded
implementation requests. The Phase 4.0 foundation is invoked by Phase 3 through
injected contracts. The target stage participates inside the Phase 3 retrieval
loop; it does not wait for a batch of Phase 3 evidence and does not assume that
retrieving a PDF or STL means its diagrams or geometry were understood. Phase
4.0 runs before the first source decision, Phase 4.1 or Phase 4.2 runs after each
relevant source is served, and Phase 4.3 reassesses the updated ABox after each
accepted delta.

### Phase 4.0: PPR-aligned context representation - implemented and contract-first integrated with Phase 3

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
- Use the application-owned `GroundingProducerDescriptor` registry rather than
  selecting a producer from evidence type alone. Each descriptor advertises the
  exact semantic or typed outputs its authorized producer can establish. Neither
  representation is part of the TBox or ABox: the TBox defines valid meaning,
  not which sensor, document, or producer PA must use.
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

### Phase 4.1: OpenAI document interpretation - implemented diagnostic

- Implement the VLM as a controlled tool under `tools/document_evidence/`, not
  as another agent.
- Accept one exact served NIST document, render all six pages at a bounded
  resolution, and send the ordered text and page images through an injected
  `DocumentVisionRuntime` in one structured request.
- The production adapter uses the OpenAI Responses API with `store: false`, no
  tools, a strict JSON schema, and the model settings in
  `config/model_runtime.json`. Credentials remain environment-only.
- Compile model entity keys into interaction-owned IRIs and return a generic
  evidence-backed triple delta without mutating the ABox or declaring context
  understanding complete.
- Let the VLM interpret approved document diagrams only. Do not give it RGB-D
  observations or use it to propose metric geometry.
- Preserve the exact document `context ref`, page-level provenance, structured
  VLM output, configured and returned model, response identifier, rendered-page
  hashes, compiled delta, and uncertainty. Uncertain interpretations remain
  unresolved and do not become factual assertions.
- Expose a separate UI diagnostic that initializes and persists its own ABox.
  Keep the main PA loop fail-closed until the remaining grounding and assessment
  producers are implemented.

### Phase 4.2: CAD and RGB-D grounding

#### Phase 4.2A: CAD and RGB-D preprocessing - implemented diagnostic

- `tools/rgb_d_cad_grounding/preprocessor.py` accepts only an exact served CAD
  or observation context. Approved CAD access is bound to the fixed inventory;
  the served SHA-256 must still match before the full binary STL is loaded.
- CAD preprocessing converts every triangle vertex from millimetres to metres
  and atomically stores compressed `triangles_m` and `facet_normals` arrays. Its
  typed record preserves source provenance and hashes, triangle and vertex
  counts, bounds, centroid, units, coordinate frame, and artifact metadata.
- Observation preprocessing reloads the complete validated four-camera bundle,
  accepts only supported calibrated distortion models, and deprojects every
  finite positive registered metric-depth pixel. Each independent optical-frame
  artifact stores lossless valid-resolution `points_m`, `colors_rgb`, and
  `pixels_uv` arrays with timestamps, calibration, evidence refs, counts, depth
  range, and hashes. No camera extrinsics or hidden downsampling are introduced.
- Each exclusive operation persists atomically under
  `products/grounding/rgb_d_cad_grounding/`. The returned generic delta contains
  no RDF assertions, references the typed record, and preserves correspondence
  and pose as explicit unresolved needs.
- The controlled preprocessing diagnostic remains callable through an injected
  `ObservationCaptureRuntime` for focused tests. It is no longer exposed as an
  operator workflow and cannot report context completion.
- The production PA loop remains fail-closed. Phase 4.2A is not registered as a
  complete CAD or observation grounding producer because it performs no
  segmentation, correspondence, pose estimation, or Phase 4.3 assessment.

#### Phase 4.2B1: minimal automatic RGB-D segmentation - implemented supporting infrastructure

- `segmenter.py` accepts only an intact Phase 4.2A
  `ColoredPointCloudSetRecord`, verifies its referenced point-cloud hashes and
  array contracts, and atomically persists one compact segmentation record plus
  one `uint16` label mask per camera.
- Use fixed deterministic internal parameters for conservative support-plane
  detection, depth-connected regions, minimum candidate size, and candidate
  limits. They are implementation details, not operator configuration.
- Treat `cam_mk3`, `cam_mk4_1`, and `cam_mk4_2` as source cameras: remove a
  reliable dominant support plane and segment remaining loose regions. Treat
  `cam_assembly` as the assembly-target camera: retain its dominant surface and
  segment its plate area in that camera's independent optical frame.
- Record camera role, frame, candidate count, point bounds, centroid, depth and
  pixel bounds, source and mask hashes, and the fixed parameter set. A camera
  with no candidate is explicitly unresolved. Identity, `CAD_correspondence`,
  pose, and cross-camera fusion remain `not_evaluated`.
- `run_automatic_rgbd_segmentation_pipeline(...)` performs one automatic
  capture → validation → observation preprocessing → segmentation sequence.
  It accepts only the contexts root and injected capture runtime; CAD
  preprocessing remains a separate operation for later correspondence work.
- The operator UI polls a compact `idle`, `running`, `ready`, or `failed` status
  and displays only source and assembly candidate counts plus the unevaluated
  identity and pose states. It has no CAD, timeout, role, threshold, mask, or
  artifact controls. Until an authorized runtime caller invokes the automatic
  observation path, production status remains `idle`.
- This phase is perception plumbing, not the research contribution. It does not
  use an ontology, VLM, PA decision, identity model, CAD matching, pose
  estimator, planner, resource agent, primitive composer, or executor.

#### Phase 4.2B2A: simple CAD size matching and candidate location - implemented supporting infrastructure

- `size_correspondence.py` accepts one intact `RGBDSegmentationRecord` and one
  already-preprocessed exact approved `CADMeshRecord`. It revalidates record
  paths, hashes, mesh bounds, point-cloud arrays, label masks, camera frames,
  candidate summaries, and aggregate counts before measuring anything.
- Reconstruct every candidate from its label mask and colored point-cloud
  artifact. Compare its two largest principal dimensions with the two largest
  CAD dimensions and record the relative error plus coordinate-wise median
  center in the candidate's camera optical frame.
- Accept only when both dimension errors are at most 15 percent and the next
  ranked reliable candidate is at least ten percentage points worse. Similar
  valid sizes are `ambiguous`; no valid size is `rejected`. A candidate touching
  an image boundary is treated as unreliable partial visibility and cannot be
  accepted.
- Persist one exclusive atomic `CADSizeCorrespondenceRecord` under the existing
  geometry product directory. Preserve the exact CAD ref and hashes, source
  segmentation ref and hash, complete deterministic ranking, selected camera,
  frame, candidate identifier, dimensions, errors, and camera-frame center.
- `CAD_correspondence` is `accepted`, `ambiguous`, or `rejected`; `location` is
  `available`, `ambiguous`, or `unavailable`. Rotation, complete pose,
  cross-camera fusion, and robot-frame conversion remain `not_evaluated`.
- Process only the exact CAD supplied by the caller. Do not scan or preload the
  approved CAD inventory, use a VLM, or read simulator identities, configured
  positions, detector output, or evaluator data.
- The operator UI remains status-only. It can display CAD-correspondence and
  location states plus `pose: not_evaluated`, but exposes no CAD selector,
  coordinates, thresholds, masks, scores, or processing controls.
- This size association is supporting perception infrastructure. It is not
  installed as a complete Phase 3 producer and cannot authorize context
  completion, assembly readiness, planning, or execution.

#### Simple generalized camera-frame pose estimation - implemented supporting infrastructure

- `pose_estimation.py` consumes one intact `CADSizeCorrespondenceRecord` and
  revalidates its complete CAD, segmentation, point-cloud, label-mask, and hash
  chain before estimating anything.
- Consider only size-plausible loose `source` candidates. Deterministically
  sample the exact approved CAD, create 24 principal-axis orientation
  hypotheses, and refine each with trimmed point-to-point ICP using the existing
  NumPy and SciPy dependencies.
- Rank registration fitness and inlier error. Accept a pose only when one
  candidate and rotation is clear, retain materially different qualified
  hypotheses as `ambiguous`, and reject weak or absent fits.
- Persist one exclusive atomic `CADPoseEstimationRecord` containing complete
  upstream provenance, fixed parameters, ranked and qualified hypotheses, and,
  only for an accepted pose, translation, rotation matrix, quaternion, and the
  camera-from-CAD transform.
- Keep output in the selected camera optical frame. It is not a robot-frame
  pose or pick point and does not establish `target_feature`, target pose,
  insertion axis, tolerances, assembly readiness, or context completion.
- The UI remains status-only. It can display only the compact pose state and
  exposes no CAD selector, coordinates, rotation, score, threshold, mask, or
  processing control.
- This simple generalized pose estimator is supporting infrastructure, not the
  research contribution. It never reads simulator identity, configured pose,
  detector response, or evaluator data.

#### Simple camera-to-robot frame conversion - implemented supporting infrastructure

- `frame_conversion.py` records one caller-approved camera-to-robot rigid
  transform with exact frames, validity window, source provenance, and a
  deterministic payload hash. No transform value is derived from simulator
  identity, world contents, or configured spawn pose.
- Revalidate the complete camera-pose hash chain and use the originating
  observation timestamp to check calibration validity. Require the calibration
  source frame and exact caller-requested target frame to match.
- Compose `robot_from_CAD = robot_from_camera × camera_from_CAD` and persist one
  exclusive atomic `RobotFramePoseRecord` with translation, rotation matrix,
  quaternion, transform, and complete input hashes only for an accepted pose.
- Propagate `ambiguous` and `rejected` pose states without robot-frame
  coordinates. The diagnostic and UI expose only compact conversion status.
- This standard frame conversion is supporting infrastructure, not the research
  contribution. It does not select a robot, contact RA, plan, or execute.

#### Later Phase 4.2 grounding work - not implemented

- Keep later cross-camera transforms under
  `tools/rgb_d_cad_grounding/`.
- Register the tool under the application-owned `CAD` and `observation` producer
  routes. For the currently served approved evidence, dynamically detect zero or more
  instances, generate their IRIs, evaluate candidate CAD correspondences, and
  return the same generic triple-delta contract without mutating the ABox.
- Retain detailed geometry in dynamically created typed context records
  referenced by the assertion provenance. Propose RDF assertions only through
  vocabulary declared by the loaded TBox and accepted by the shared validator.
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

### Phase 4.3: post-understanding decision — implemented pre-RA

The production assessor and its `TaskTransitionDraft`/`ContextNeed` contract
are implemented. The configured production path persists each draft, context
view, producer choice, evidence result, and reassessment; an unconfigured path
still stops at `grounding_unavailable`.

- Assess the evolving `TaskTransitionDraft`, current interaction ABox,
  `ProductContextView`, unresolved assertions, evidence status, and authorized
  `GroundingProducerDescriptor` records against the exact
  `product_requirement`. Phase 4.1 and Phase 4.2 may each run zero or more
  times; neither modality is mandatory by itself.
- Compute task-level `ContextNeed` values as required draft-consumer inputs
  minus valid current `TypedContextBinding` values. Resolve only the currently
  blocking needs by selecting descriptors whose declared outputs and evidence
  policy match. Do not retrieve a modality merely because it is available.
- Run each selected producer as one internally auditable source operation,
  validate and merge its result, update the `ProductContextView`, and reassess.
  PA may dynamically select approved document, exact approved CAD, RGB-D,
  calibration, or an existing record; it does not exhaust the corpus or follow
  a hard-coded evidence sequence.
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

- Have PA evolve the grounded product requirement into a
  `TaskTransitionDraft`, resolve only its currently blocking task-level
  `ContextNeed` values, and accept it as a `TaskTransitionContract` for the
  inherited allocator.
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
- Interpret `context understanding complete` narrowly: the currently blocking
  task-level inputs required to produce the Phase 5 contract are valid. Later
  primitive contracts may expose additional runtime inputs after allocation;
  Phase 5 need not predict or eagerly retrieve them.
- Do not express the plan as calls to the prior predefined composite functions
  `move_to_pick_location`, `pick_part`, `move_loaded_to_destination`, or
  `place_part`, or to an equivalent product-specific callable. PA specifies the
  required robot-independent transition; it does not select a function whose
  internal primitive program already solves that transition.
- Version the grounded assembly contract and plan. When an RA-triggered
  `MissingContextBatch` adds product or scene knowledge, return a versioned
  `CompositionContextBundle`, revalidate the plan before the next RA update,
  revise it when a new fact changes product-level steps, or explicitly preserve
  it when the update only supplies missing bindings.

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
- Make the exchange bidirectional. Let RA return one deduplicated
  `MissingContextBatch` containing all currently known unbound product or scene
  inputs, their owning authority, primitive or validation reasons, consumer
  contracts, and supporting record refs.
- Route only product or scene knowledge gaps back to PA. For a nominal task, PA
  resumes the same interaction ABox and Phase 3.3/Phase 4 loop, then Phase 5
  revalidates or revises the task contract. For a recovery task, preserve a
  binding-only update when the selected event remains valid; if the new fact
  changes its identifier, required outcome, semantics, feasibility, or assigned
  `resource_jid`, stop that transfer attempt fail-closed. The inherited framework
  may produce a newly validated event outside Spec2Primitives and import it as a
  new transfer input; the adapter never invokes the recovery selector.
  PA resolves the batch through controlled producers and returns one versioned
  `CompositionContextBundle` of `TypedContextBinding` values. Robot state,
  capability, primitive-catalog, IK, collision, and trajectory gaps remain
  RA-owned and are resolved locally without a PA round trip.
- Permit another batched exchange only when binding or validation exposes new
  needs and the prior round made measurable progress. There is no fixed semantic
  round count. Reject replayed or repeated needs, stale versions, unavailable
  producers, ambiguity, or a round that adds no valid binding, and terminate
  fail-closed. Operational deadlines and cancellation remain allowed.
- Keep upstream recovery-event feasibility records distinct from downstream
  primitive-program validation. Importing the selected recovery event does not
  make recovery event generation, selection, allocation, DES reasoning, or
  safety planning a Spec2Primitives contribution.
- Display exchanged messages in the corresponding live PA and RA cards and the
  ordered interaction record.
- Keep shared PA and RobotAgent implementations unchanged.
- Preserve exchanged inputs, outputs, and provenance as reviewable context
  records.

## Phase 7: RA context retrieval and primitive composition

This phase requires a separate implementation request.

- Retrieve fresh state from the selected RA and the complete
  selected-RA-authoritative primitive-only catalog. Catalog cardinality is
  runtime-determined. Preserve every exact symbol and pin the catalog version,
  fingerprint, and reported cardinality for the composition attempt; never
  normalize, rename, replace, or silently supplement an entry.
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
  individuals from the versioned PA ABox, the selected resource, and every
  primitive-process interface individual in the pinned resource-catalog ABox
  snapshot. Include authorized typed context refs and evidence provenance.
  Persist the exact projection and both source fingerprints for every RA turn.
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
- Treat the fresh snapshot and complete current catalog as mandatory initial RA
  context, not as a complete PDDL domain. Supply no task-specific domain or
  problem file, exhaustive action model, closed-world state, expected sequence,
  or mapping from the requested outcome to a primitive decomposition.
- Show RA building context from the retrieved `context ref` in the RA card and
  ordered interaction record.
- Give the selected RA LLM the versioned composition-task envelope, required
  outcome, fresh initial state, complete current semantic primitive interfaces
  and partial local executable contracts, the read-only task-relevant ontology
  projection, relevant typed context refs, resource limits, and any findings
  from the previous candidate. Do not give it hidden expected answers or a
  completed decomposition. The ontology projection supplies shared meaning;
  the semantic interface cards supply callable behavior details that a process
  class or primitive name alone cannot provide.
- After loading the mandatory prerequisites, have the selected RA author a
  structural `PrimitiveProgramDraft` using only the pinned catalog. A binding
  preflight compares the draft's declared consumer inputs with fresh valid
  context, gathers all currently visible gaps, and partitions them by authority.
  It may identify missing inputs, but it must not select, add, remove, reorder,
  parameterize, or repair primitive steps.
- Let an unbound input on a primitive interface or candidate identify the
  semantic need, not the evidence modality. For a product or scene input, RA
  returns that exact typed need to PA; PA selects the matching controlled
  producer and its approved evidence. RA must not request a raw PDF, STL, RGB,
  depth, or calibration payload directly.
- Have the RA LLM author the structural draft and later the fully bound
  `primitive_steps` candidate by selecting, ordering, parameterizing, and
  binding only symbols in the pinned catalog. This agentic step is the composer.
  No binder, deterministic composer, backward or forward search, validator,
  stored recipe, product-specific macro, or capability decomposition may create
  or complete the candidate.
- Use the dynamically retrieved PPR projection only as semantic and binding
  context, then use the primitive contracts plus typed context refs to bind
  runtime inputs. Treat any missing required binding as
  composition-not-ready context instead of inventing a value or claiming
  execution readiness. The RA LLM may reason over the projection, but neither
  OWL inference nor the projection adapter may author `primitive_steps`.
- Have RA perform the binding. Resolve RA-owned inputs from fresh local state.
  Collect all currently known product or scene gaps into one stable,
  deduplicated `MissingContextBatch`; PA runs the required controlled producers
  and returns a versioned, fingerprinted `CompositionContextBundle`. RA retries
  binding only after validating the bundle, assertion versions, typed context
  refs, and task update. PA does not bind robot primitive parameters.
- Repeat the batched round only when a new need appears and the previous round
  made progress. Stop fail-closed on ambiguity, repeated requests, replayed or
  stale bundles, unavailable producers, a changed catalog fingerprint, or no
  new valid binding. No fixed semantic round count is part of the contract.
- Feed nominal and recovery envelopes through the exact same composition
  entrypoint and prompt policy. `task_origin` is audit and upstream-routing
  metadata; it must not select a recovery-specific composer or recipe.
- Allow the RA to return a bounded, structured failure when it cannot produce a
  candidate. Agentic generation is not assumed to be correct or complete on the
  first attempt, and an unvalidated candidate must never proceed to execution.
- Preserve every retrieved `context ref`, catalog snapshot,
  `PrimitiveProgramDraft`, `MissingContextBatch`, `CompositionContextBundle`,
  fully bound candidate, revision, progress decision, and ABox update in the
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
  newly exposed product or scene binding gap joins a deduplicated Phase 6
  `MissingContextBatch`; other failures remain RA-local revision evidence. A
  binder or validator may report the gap but may never create or repair a step.
  The RA LLM, and only the RA LLM, authors the next complete candidate.
- Accept only the unchanged candidate that passes every applicable check. Stop
  fail-closed when the bounded revision budget is exhausted.
- Show validation results and revisions in the RA card and ordered interaction
  record.
- Store validation traces and revision histories in the active interaction
  under `contexts/`.

### Phase 7 and Phase 8 verification

- Test complete selected-RA-authoritative primitive-only catalogs at multiple
  cardinalities. Each candidate may use only exact symbols from its pinned
  per-attempt catalog, and no template, binder, decomposition, validator, or
  fallback code may create or repair candidate steps.
- Test that the task envelope, composition prompt, resource-catalog snapshot,
  and execution dispatch contain none of `move_to_pick_location`, `pick_part`,
  `move_loaded_to_destination`, or `place_part`, and contain no equivalent
  predefined composite callable or hidden primitive expansion.
- Test admission explicitly: a requirement with an available matching composite
  function is labeled prior-function reuse and is not counted as a
  Spec2Primitives composition; the nominal and recovery study inputs have no
  such function and expose only their selected RA's complete primitive-only
  catalog.
- Test that each catalog item contains its semantic executable interface and
  truthful partial local contract while the combined input contains no PDDL
  domain/problem, exhaustive action model, expected sequence, or task-specific
  primitive decomposition.
- Test the TBox/ABox split: generic PPR vocabulary remains in the TBox; PA
  evidence creates task/product/outcome individuals only in its interaction
  ABox; selected-resource and current primitive offerings appear only in
  the versioned resource-catalog ABox; and RA receives a read-only projection
  without mutating either graph.
- Verify that the projection includes the high-level requested process and
  outcome plus every available primitive-process interface in the pinned
  snapshot, but contains
  no `requires`, `precedes`, direct high-level `capableOf`, expected sequence,
  or other task-to-primitive decomposition leak.
- Test the same composition contracts, prompt policy, candidate schema,
  validators, and revision policy with nominal and recovery-origin envelopes.
  For equal-input comparisons, pin the same per-case catalog snapshot,
  fingerprint, cardinality, exact symbols, resource-owned limits, and evaluators.
  Cross-resource experiments may vary catalog size and symbols and must report
  those differences rather than treating cardinality as an admission condition.
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
- Test `MissingContextBatch` ownership and task-origin-specific routing without
  allowing PA to bind RA primitive parameters or the ontology to decide
  composition readiness.
- Test stable batch deduplication, multiple productive context rounds, replay
  rejection, stale bundle and catalog versions, unavailable producers,
  authority routing, ambiguity, and deterministic no-progress termination.
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
  composer, contract schema, model and prompt policy, validator stack, budgets,
  and execution boundary. Pin the same catalog snapshot and fingerprint within
  each equal-input case, and report its exact symbols and cardinality. Catalog
  size and symbols may vary across selected resources or cross-resource cases.
  Preserve each selected RA's truthful contract values and evaluators.
  Recovery-specific information may change the input contract but not the
  composition mechanism.
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
