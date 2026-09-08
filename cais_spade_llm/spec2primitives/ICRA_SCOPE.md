# ICRA 2027 Spec2Primitives Scope

## Research title

**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**

## Generalization objective

The paper’s primary objective is a manufacturing-process-general framework for
unforeseen product requirements, not an assembly-specific planner. A requirement
may concern assembly, milling, printing, lathing, painting, or another
manufacturing process. The framework should ground the requested
current-to-desired state transition from approved evidence and compose from the
capabilities and primitive contracts available at runtime, without embedding a
case-specific component, destination, evidence source, or expected answer in
shared prompts, schemas, ontology templates, or controller feedback.

The NIST assembly manual and Medium Gear task are one controlled case study used
to instantiate and evaluate the framework. They do not define the architecture’s
process boundary. This is an architectural objective—not a claim that every
process is already implemented. Open-ended requirements compose over the finite
evidence, validation, feasibility, and physical-capability catalogs deployed at
runtime. A new process name or phrasing does not itself require a new adapter.
Additional providers, validators, or robot primitives are required only when the
requested evidence or physical operation cannot be expressed by the existing
contracts. When a required capability is unavailable, the runtime must fail
closed rather than reinterpret the requirement as assembly.

## Proposed-versus-implemented status

The end-to-end composition/execution architecture in this document remains proposed. Current product grounding accepts one complete target and a collection of pairwise assembly relationships. Current observations and desired relationship membership are separate. PA owns semantic interpretation and corrections. Deterministic contract feedback permits up to six proposals sharing 24 evidence operations and one pinned observation per grounding invocation; exact repeated failures stop the loop earlier.

PA checks every configured capable arm using the same bound current/destination coordinate references and live MoveIt position planning, then selects an arm with an accepted check. Missing arm checks retain their separate single correction; unavailable results remain distinct from rejected planning. Selection and completion record that ontology assignment with position-planning evidence and explicit unvalidated grasp/insertion constraints. Multiple relationships do not force a unique Cartesian pair. Required locations are checked before target commit; unresolved inputs or assignment prevent completion. Contract acceptance does not establish semantic correctness.

Phase 5.1 requires current completion before exact selected-RA envelope and paired context snapshots. Primitive composition starts directly from those authorities. RA chooses a single `primitive_steps` program with available parameters and can inspect existing pinned evidence and accepted ontology assertions. Missing geometry permits omitted parameters, displayed as `<unbound>` when required. Supplied structure and references are checked; there is no separate draft stage. The bounded loop obtains supplemental evidence, measures robot geometry, calculates selected helpers and validates direct Cartesian segments in a private scene. RA alone revises its program. Force/contact validity, execution and observed outcomes remain future work. Historical drafts and draft-dependent attempts remain immutable and are excluded from new composition inputs.

See [implementation status](IMPLEMENTATION_PLAN.md), [ontology semantics](ASSEMBLY_ONTOLOGY.md) and [bias experiments](BIAS_VALIDATION.md). Generic deterministic checks and offline tests do not establish model accuracy or absence of bias.

## Correct primitive-input boundary

Phase 4 supplies observed product information and desired assembly relationships. Phase 5 determines the robot targets needed by RA's selected operations. The current bounded loop obtains supplemental evidence and calculates RA-selected targets before modeled validation; missing geometry remains explicit.

Fresh captures preserve full runtime contracts and nested schemas, plus configured planning frame/EE/TCP names. Composition exposes only `held_part` grasp/release formal conditions/effects and filters recovery metadata from initial state and every record read. No richer causal model or sequence rule is added.

The derived report checks RA-selected references against input shape, frame and binding meaning. Raw CAD is not complete placement geometry; descriptive destination labels and semantic part labels are not controller bindings. Missing inputs stay visible without target fabrication, source substitution or program repair. Only explicitly selected, grounded helper calls are numerically evaluated. `approach_pose` and conditional insertion outputs remain available declarations; RA chooses their use and order.

Use [VALIDATION_AND_REVISION.md](VALIDATION_AND_REVISION.md) for implemented bounded resolution/validation and subsequent physical work and [COMPOSITION_EVALUATION.md](COMPOSITION_EVALUATION.md) for the partial-contract hypothesis, fair baselines, held-out conditions and claim limits. No offline test establishes reliable physical composition.

## Fundamental research challenge

The central research question is:

> Given an ontology-grounded task and selected-resource assignment, fresh
> selected-resource context, and the
> complete selected-RA-authoritative primitive-only catalog with partial local
> executable contracts, can one RA-owned LLM composer author and revise
> validator-accepted
> `primitive_steps` without a product-specific recipe, completed
> `primitive_steps`, complete PDDL domain/problem, or hidden ground truth?

The intentional research gap inside the representation is that broad
`resource capableOf assembly` assertions do not establish which robot can reach
the product now and encode no task-to-primitive decomposition. Phase 4.4 closes
the first gap with hash-pinned location evidence and manifest reach checks, then
the system records one execution assignment. The selected RA's later typed catalog
exposes lower-level primitive interfaces without primitive-level `capableOf`
assertions. The RA LLM must interpret the process meaning, outcome, grounded
bindings, primitive descriptions, and fresh state to propose the missing
composition; deterministic validators then determine whether that unchanged
proposal is admissible.

The paper studies the transformation across abstraction levels:

```text
product specification
        ↓
evidence-grounded target feature / desired product state
        ↓
robot-independent grounded product outcome/task
        ↓
robot-specific primitive_steps
        ↓
validator-accepted primitive program
        ↓
execution
        ↓
observation-backed realized outcome
```

The implemented contract supports one complete requirement transition with generic typed state values and zero or more pairwise assembly relationships. Each relationship states current/desired membership independently of endpoint observation bindings. The host preserves exact identities and compiles associations separately under their Assembly owners. PA interprets goal coverage, task roles, current attachment and intended relationships. Deterministic checks establish valid bindings and evidence integrity; citations alone do not prove meaning.

All bound coordinate-bearing state references determine the required arm check. Non-coordinate state values remain semantic evidence. PA chooses among configured capable arms; the host cannot substitute one. Current completion records contract-validated product grounding and live MoveIt reachability-backed assignment. Primitive-program validity, semantic correctness and assembly outcome remain separate claims.

The research contribution is not that PA and RA exchange messages. It is the
combination of:

1. PA dynamically grounding an incomplete product requirement from only the
   relevant approved document, CAD, and RGB-D evidence, followed by an
   evidence-backed PA selection from the configured capable resources and configured
   workspace/gripper reachability of that exact choice.
2. RA, where RA means RobotAgent, interpreting that required transition against
   fresh selected-resource state and the complete current semantic primitive
   catalog with partial local executable contracts, then agentically authoring
   primitive selection, order, bindings, and parameters. Catalog cardinality is
   runtime-determined rather than part of the research claim.
3. Non-synthesizing contract and robot-local validators returning concrete
   findings so the same RA LLM can author a revised complete candidate without
   a validator inserting or repairing steps.

Exact-ref retrieval, PA and RA live cards, ordered interaction records, and
simulation are supporting infrastructure and evaluation evidence. They are not
the fundamental contribution by themselves.

The evidence-first manual pipeline is also supporting infrastructure for the
composition paper, not a general document-understanding contribution. A
deterministic, requirement-blind source index preserves ordered page text,
rendered-page references, and hashes. PA may then author an exact document
question whenever it judges that source relevant; the bounded document VLM sees
only that question and the selected document pages. Retrieval is not globally
document-first, no vector database is required for the current manual, and no
controller-authored question prescribes a component, destination, evidence
modality, or expected answer. Deterministic validation checks shape, authority, references, hashes and bindings.
Missing planning inputs and unissued references return concrete deterministic
feedback within the shared grounding budget. PA authors each correction; the host
does not supply an expected answer or judge complete goal coverage.

Automatic RGB-D capture, deprojection, support-plane segmentation, exact-CAD
size comparison, and camera-frame candidate-center measurement are also
supporting perception infrastructure. A Phase 4.2B2A size match is not a
complete pose, robot pick coordinate, task-completion assessment, or primitive
composition result.

## Partial formal-model boundary

Each entry in the complete resource-owned catalog snapshot exposes its exact
primitive symbol, operation description, typed parameters and results,
invocation binding, truthful limits, direct evidence, applicable evaluator
endpoints, and only the local conditions or effects explicitly modeled. An
omitted condition or effect is `unmodeled`, not satisfied.

The RA composition view retains primitive symbols and product/motion inputs while
omitting the execution-only `model_name` binding from parameters and results.
Full runtime signatures stay captured. A future adapter must supply the
identifier for the recognized physical instance before Gazebo execution;
discovering simulator names is outside the composition research task.

Together, these interfaces are not a task-specific PDDL domain/problem,
closed-world state model, product-to-primitive mapping, expected primitive
sequence, or exhaustive transition model. Their detailed executable cards stay
in typed resource records because the sample PPR vocabulary does not define
contract fields.

The ontology split is:

- The immutable TBox contains generic PPR classes, properties, restrictions,
  and the narrow `processExecution` assignment vocabulary.
- PA proposes one per-interaction target feature only from the requirement and
  authorized, hash-pinned evidence. The host generates `feature_0001`,
  `currentstate_0001`, and `desiredstate_0001` and compiles their exact
  variable-size feature/state/association ABox projection under the current TBox.
- The workcell-profile-derived ABox records the configured process and resource
  individuals plus each resource's explicit process capabilities. The production
  profile currently yields `xarm6` and `ur5e` broadly `capableOf assembly`.
  Only the system adds the selected `processExecution` after PA-cited two-state
  reach evidence and exact selected-RobotAgent plan-only validation.
- The selected RA later supplies a separate typed primitive-catalog
  snapshot. It pins exact symbols, fingerprint, and cardinality without
  publishing primitive implementations through `capableOf` assertions.

The pasted OWL is a mixed schema-and-instance graph rather than a pure TBox. It
remains a parser fixture only; its named individuals and task-specific
restrictions are not runtime context. The project-authoritative production TBox
is `ontology/spec2primitives_ppr_tbox.owl`; it contains only the approved
schema-only PPR view.

The PA-selected `required_process.process_iri` is accepted only when it cites
authorized evidence and matches configured process authority. The host then
compiles the correctly typed `realizes` assertion; PA does not freely author
relations. A high-level process must not be linked to primitive offerings
through `requires`, `precedes`, or another expected set or order. Broad
resource-to-configured-process capability supports semantic candidate discovery
only. A
read-only runtime projection later joins the relevant task and Workcell
assertions with the selected RA's typed primitive catalog. The RA-authored
`primitive_steps` candidate, not an ontology entailment, creates the task-to-
primitive connection dynamically.

Backward Derivation and Forward Validation inspect only the unchanged candidate
over explicitly represented contract fields and resource-owned evaluator
results. They reject or report `unmodeled`; they do not search for, insert,
remove, reorder, parameterize, or repair a primitive. A complete engineered
PDDL model or black-box search could still provide another composer, so the
paper does not claim that classical planning is theoretically incapable or that
an LLM is universally necessary.

## Dynamic context-retrieval boundary

The requirement and TBox are understood together, while PA dynamically chooses
which approved document, file, observation, or existing record to retrieve.
The TBox defines legal shared meaning; it does not prescribe a document, CAD,
observation, calibration, or primitive recipe. After allocation, the selected
RA's composition and primitive interfaces expose further
runtime inputs.

```text
requirement + ontology
        ↓
retrieve only currently relevant approved evidence
        ↓
PA authors one cited target_feature with current and desired states
        ↓
validate the final shape, process authority, references, hashes,
and exact ontology projection plus required planning inputs;
return repairable contract failures to PA within the shared budget
        ↓
commit the accepted feature-state ABox
        ↓
present every capable resource and neutral location handle
        ↓
derive all bound coordinate references; PA checks and chooses one capable resource
        ↓
check_reachability reports each submitted location independently
        ↓
validate the unchanged cited selection once
        ↓
commit the four processExecution/resource assertions and completion ```

Provider-owned `GroundingProducerDescriptor` values declare exact IDs,
descriptions, accepted evidence types, produced record types, prerequisites,
availability, and estimated cost. Registration order is not a priority. PA may
dynamically choose an approved document, exact caller- or corpus-authorized
CAD, fresh RGB-D, matching injected calibration, or an existing accepted
record. Numeric values remain in typed records. OWL open-world absence is never
treated as completeness, and no static all-modality checklist or expected
primitive sequence is encoded.

Derived provider state is keyed by the exact prerequisite record refs and
SHA-256 values, not merely by record type. An unchanged ambiguous or rejected
result runs once and exposes a new external evidence revision; a newer RGB-D or
CAD binding invalidates only the downstream correspondence, pose, calibration
use, and robot-frame location conversion. Hash-changed embedded evidence is represented as
stale and reopens the need. Manifest or authority failures remain terminal and
do not masquerade as a reason to recapture the scene.

For the current sole approved six-page NIST PDF, one document retrieval supplies
pages 1 through 6 together and in order. PA may ask its own bounded document question after retrieval;
there is no page ranking or RAG, and large-document retrieval remains outside this scope.

The model never manages hidden host-authored information-need state. The system
validates native tool calls, exact replay keys, citations, hashes, and TBox
signatures. Before feature/association assertions commit, the host checks required
location inputs. Missing locations or unissued references receive generic feedback
for PA correction within the shared budget. Invalid provenance remains terminal.
PA then checks/selects a capable resource using all bound locations. Unresolved
inputs, missing capability or failed reachability prevent completion without
the host manufacturing an assertion.

When allocation succeeds, `PAContextGroundingCompletion` pins the accepted
proposal with its pre-commit context and complete evidence manifest, both feature states,
evidence and allocation presentation records, bound location lists and PA-selected resource,
registry and workcell snapshots,
persisted MoveIt position-planning results, the post-assignment ontology
projection, `ResourceSelectionRecord`, and referenced typed records without
copying the target feature or creating a new-run `TypedGroundingContract`. There
is no separate `TaskTransitionContract`. The
selected RA then receives a reconstructed transient `target_feature`, bounded
resolved state values, selected-resource projection, fresh state, and its
complete current primitive-only catalog. The RA LLM authors `primitive_steps`
with available parameters. The current derived binding report identifies
missing, incompatible, unverified and deferred inputs without creating or repairing steps. Robot state and later
primitive-level feasibility remain RA-owned. Phase 4 reachability checks only
the submitted locations through the owned MoveIt adapter without contacting
a RobotAgent or executing motion.
The proposed later context loop deduplicates product or scene gaps into one
`MissingContextBatch` per round.

Formal PA completion pins both the exact `ResourceSelectionRecord` and the
`resource_grounding_host` assignment delta. The four execution assertions must
all cite that same selection record, and its selected resource, grounding-record hash,
per-location reach verdicts, and final `runsOnResource` value must agree when the
completion is reloaded.

For a supported requirement, PA may retrieve any relevant approved evidence,
author current and desired feature states, then freely assign neutral locations
and one capable resource. Rejection ends without an ABox assignment or
host-selected replacement. Acceptance commits exactly the PA choice; no
resource identity is a registry-order default, retrieval rule, or ontology
entailment.

PA may service one batch through several existing single-source audited
operations, then returns one new `CompositionContextBundle`. There is
no fixed semantic round count. Another batch round is allowed only after a new
accepted binding, changed need classification, or structurally different RA
program proposal demonstrates progress. Ambiguous or unavailable evidence, an unsupported
need, an identical request against unchanged context, or no new accepted result
stops fail-closed. `context understanding complete` means readiness for Phase 5,
not that every later primitive input is already available.

Phase 4.4 candidate discovery is a semantic join over the pinned workcell-profile
and registry snapshots for the process selected from the authorized catalog.
Configured capability and live MoveIt position checks are deterministic evidence providers
for current Phase 4; position plans are validated; grasping and insertion remain unvalidated. PA
remains the allocation authority and no optimality claim is made. Both
nominal and recovery cases consume the selected exact
`resource_jid` and use unicast; Spec2Primitives does not broadcast or reallocate
during composition.

## Perception tool boundary

```text
PA
├── document evidence tool
│   ├── registered PDF validation and content-addressed overview cache
│   ├── complete ordered-page ontology-neutral overview
│   └── native retrieve call with audited source identity and hashes
│
└── RGB-D/CAD grounding tool
    ├── RGB-D capture and preprocessing
    ├── minimal camera-local segmentation
    ├── all-candidate CAD-size and raw layout measurements
    ├── location conversion from injected calibration
    └── optional pose estimation for orientation-sensitive consumers
```

Only PA and RA are agents. The observation provider and the perception
components shown above are controlled tools invoked within the Spec2Primitives
workflow. Related vision algorithms are combined into the two tools instead of
being modeled as additional agents.

These tools expose uniformly segmented observations and neutral candidate
handles. They do not label a source or target region. The architecture has no
`TargetFeatureGeometryRecord`: target-feature state values may reference any
accepted typed evidence, while PA independently assigns neutral location handles
during allocation. Numeric robot-frame locations are derived only for the
handles submitted to reachability. No fixed document, CAD, RGB-D, candidate, or
resource order is prescribed.

Retrieving an approved live observation invokes
deterministic preprocessing and segmentation; PA does not select the
segmentation algorithm or author that record. PA authority begins with its
choice of approved evidence handle and continues through its selection of the
target-feature values, state-location lists, and resource. This is bounded PA
autonomy inside a fixed perception contract, not unrestricted tool creation.

RGB-D capture, calibrated deprojection, support-plane removal, connected-region
segmentation, mask persistence, one-CAD principal-size comparison,
camera-frame candidate-center measurement, generalized camera-frame
registration, injected-calibration frame conversion, and read-only processing
status are supporting infrastructure rather than the ICRA research
contribution. The active CAD comparison reports every evaluated candidate
without ranking or a built-in winner. The layout tool accepts any two or more
PA-selected same-frame candidates and reports positions, pairwise displacement
vectors, distances, and collinearity without a built-in relation verdict. These
tools do not establish general identity recognition, a grasp point, primitive
composition, planning, or execution.

Intermediate mask, depth, CAD-fit, uncertainty, and source evidence remains
separately auditable even though the algorithms are grouped behind the two tool
boundaries. RA does not receive raw document or camera data. The official
future PA→RA input is the validated ontology projection, Phase 4 completion,
and hash-pinned typed evidence records.

## Research thesis and evaluation requirement

The paper thesis is:

> Given an ontology-grounded task and selected-resource assignment, RA-owned
> agentic composition over the complete selected-resource semantic primitive
> catalog, coupled to non-synthesizing
> resource-validator feedback, improves candidate validity and robot-local
> feasibility relative to the same-input LLM composer without
> validation-driven revision.

The controlled Medium Gear case is the development starting point, not
sufficient evidence of general primitive composition. The final evaluation must
withhold product-specific assembly programs, completed `primitive_steps`, and a
complete task-specific action model; provide only the complete selected-RA
primitive-only catalog and partial contracts; record its cardinality and
fingerprint; include CAD distractors; vary
observations, object placements, and fresh resource state; exercise vague or
incomplete requirements; and require different composition structures beyond
Small, Medium, and Large variants of one gear sequence.

The primary nominal case starts at the raw requirement with no predefined
nominal task. The recovery case starts from an already selected, validated, and
resource-assigned recovery event with no primitive decomposition and tests the
same RA composer as a transfer case. Recovery-event generation and selection
remain outside this paper.

Evaluation must report PA and RA context requests, represented-contract
coverage, first-pass candidate validity, revision convergence, robot-local
feasibility, validator-accepted programs, simulation execution, and
observation-backed realized outcomes separately. Equal-input comparisons are:
an interface-only symbolic composer, bounded black-box sequence search with the
same validator access and query budget, LLM-only generation, and the full LLM
composition plus validator-feedback revision loop. A separately engineered
complete PDDL domain is an optional extra-information oracle, not an equal-input
baseline. Every equal-input comparison pins the same per-case selected-RA
catalog snapshot, fingerprint, cardinality, and exact symbols. Cross-resource
experiments may vary catalog size and symbols and must report those differences.
Monolithic, no-retrieval, and stale-context conditions are supporting PA-pipeline
ablations. Retrieval, schema validation, simulation, observed outcome, and
physical execution remain distinct claims.

### Phase 4 PA bounded-autonomy experiments

Use the preregistered audit and paired-trial protocol in [BIAS_VALIDATION.md](BIAS_VALIDATION.md). Test presentation permutations, physical destination changes, ambiguous task roles, missing/contradictory evidence, installation uncertainty, whole-goal coverage, changed arm reachability and held-out scenes. Use fresh interactions, preserve every attempt and compare canonical source identities only in the separate evaluator after prediction finalization.

Audit PA requests/projected tool results, deterministic feedback, proposal attempts, evidence manifests, reachability, selection and completion. Record the shared operation/proposal budgets and stop reasons; the host never supplies a preferred candidate or substitutes an arm. Evaluate semantic correctness separately after finalization. Report all outcomes, correct abstention, false acceptance, sample sizes and uncertainty. Offline tests establish contracts only, and neither five successful examples nor a fixed assertion count is an accuracy/generalization result.

Previously identified contaminated-prompt runs remain excluded from baseline counts and retained only as explicitly labeled historical ablations. Frozen evaluator answers never enter runtime recognition.

## Starting case

`product requirement: assemble Medium Gear`

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

## Proposed workflow

```text
exact product_requirement: assemble Medium Gear
        ↓
PA loads the fixed TBox and initializes one interaction ABox
        ↓
PA receives the exact requirement, TBox, available sources, and retrieved records
        ↓
PA optionally calls retrieve(evidence_id) zero or more times
        ↓
PA returns one complete cited target_feature, genuine clarification,
or unsupported_process
        ↓
validate the final shape, process authority, evidence, hashes,
and exact ontology projection plus required planning inputs;
return repairable contract failures to PA within the shared budget
        ↓
commit the accepted feature-state ABox
        ↓
present all capable resources and neutral location handles
        ↓
derive all bound coordinate references; PA checks and chooses one capable resource
        ↓
check_reachability reports each submitted location independently
        ↓
validate the unchanged cited selection once
        ↓
system commits four assignment assertions and completion
        ↓
Phase 5.1 activates the selected exact resource_jid through the current adapter
        ↓
selected RA receives reconstructed target_feature + resource projection
        + fresh state + complete current primitive-only catalog
        ↓
RA LLM authors primitive_steps with available parameters
        ↓
current derived binding report identifies selected-input gaps
        ↓
bounded supplemental evidence and audited selected calculation
        ├── RA-owned gaps resolve locally
        └── product/scene gaps form one MissingContextBatch
                                      ↓
                         PA runs required controlled producers
                                      ↓
                         CompositionContextBundle
        ↓
RA LLM authors a fully bound primitive_steps candidate
        ↓
non-mutating declared-contract + resource + physical + outcome checks
   ↙ new product/scene gap   ↓ robot-local finding        ↘ accepted unchanged
new batched PA round only   finding returns to the RA         fresh-state recheck
after measurable progress  LLM for a new candidate                    ↓
        └───────────────────────────↺                         RA execution
                                                                    ↓
                                             observation-backed realized outcome
```

The recovery path replaces the nominal PA contract at the composition-task
envelope boundary with the inherited selected recovery event, then uses the
same exact-JID unicast, RA retrieval loop, composer, candidate schema,
validators, revision policy, and execution boundary. If new grounding changes
the recovery event's semantics, feasibility, outcome, or assigned resource, the
transfer stops until the existing recovery framework supplies a newly validated
event.

## Historical scene-only milestone boundary

The initial scene-only milestone provided the isolated structure, research
workflow, narrow no-hardware `gazebo_dual_spec2primitives` launcher, dedicated
`table_spec2primitives.world`, and local placeholder chat. Later milestones added
the narrow PA context boundary, standalone supporting perception records, and
the Phase 5.1 contract-first RA assignment/state/catalog boundary and an earlier
structural draft stage. The current adapter can reuse or start the exact selected
context-only RobotAgent and obtain one RA-authored primitive program directly
from captured context, with bounded read-only evidence access and available parameters.
Bounded supplemental evidence, numerical target calculation and private Cartesian
validation are now implemented. Live SPADE task delivery, insertion-physics
validation and robot execution remain later work.

The starting scene pre-installs the static NIST `Gear_Plate` and three
`Gear_Shaft` fixtures while leaving `gear_small`, `gear_medium`, and
`gear_large` loose on `prusa_mk4_2`. This is scene configuration, not a completed
robot-execution claim.

The case study will use a controlled local corpus.
`NIST_assembly_instructions.pdf` is retained under `references/products/`;
existing NIST STL files are referenced in place. ProductAgent and RobotAgent
remain shared runtime authorities outside this package.

## Out of scope for the scene-only milestone

- ProductAgent, RobotAgent, `SystemBridge`, or `bridge.py` changes
- functional recognition or agent adapter implementations
- committed schemas or case payloads
- recognition, VLM, agent, detector, automatic attachment, or hardware changes
- insertion-physics accuracy claims
- experiment execution or recognition, planning, or robot-execution claims
