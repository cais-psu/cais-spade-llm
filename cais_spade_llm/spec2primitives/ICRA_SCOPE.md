# ICRA 2027 Spec2Primitives Scope

## Research title

**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**

## Proposed-versus-implemented status

The end-to-end workflow in this document is proposed architecture. **The Phase 4
runtime is implemented through Phase 4.4 under the current framework.** Under
the Gate-1 evidence policy, however, the present Medium Gear scene intentionally
stops at its ambiguous desired-state correspondence; it is not a successful
baseline until the separately authorized relational-grounding gate is complete.
The package also
implements the PA interaction boundary through Phase 3.5. PA uses native
`retrieve` and `compare_cad_size` tools and returns a direct ontology proposal. Phase 4.4 validates and
commits one v8 PA-authored, evidence-grounded `target_feature` with explicit
current and desired states, each with a uniquely supported PA-selected neutral
candidate; deterministic evidence validation runs before a separate scaffold-free
semantic consistency review accepts the unchanged pair. PA then chooses a
provisional resource through `check_reachability(resource_symbol)`, which reuses
those bindings. The exact selected RobotAgent performs
live, no-motion Cartesian pick/place validation in simulation, and only
acceptance commits that resource as a
`processExecution` assignment. The production profile configures only
`assembly`, `xarm6`, and `ur5e`; this is an assembly case study, not runtime
process discovery. Their identities and configured capabilities remain pinned
to the workcell profile and manifests; broad `capableOf assembly` assertions
do not encode reach or a primitive sequence. No
live SPADE Phase 5 delivery or execution path is connected.
Phase 5.1 implements the contract-first assignment envelope and injected-runtime
state/catalog snapshot boundary. Phase 5.2A implements one RA-LLM-authored
structural `PrimitiveProgramDraft` from a reconstructed target feature and exact
catalog symbols.

Phase 4 completion is the implemented grounded, plan-only resource-assignment
boundary. Cross-camera fusion, object-pose estimation,
grasp/contact/tolerance/insertion validation, parameter binding,
primitive-level robot validation, execution, and observed outcomes are not part
of the completed Phase 4 claim.

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

The implemented PA outcome contract is process-independent even though the
current controlled case is assembly. For one requirement, PA authors one
`target_feature` with an evidence-cited authorized process, complete
evidence-cited current- and desired-state statements, and zero, one, or multiple
optional typed state-value references for each state. PA chooses the text,
process, value count and
names, record refs, JSON Pointer paths, and citations from retrieved evidence;
deterministic code validates them but does not fill their semantics. Later
welding, painting, milling, or other processes may use the same shape when
their evidence, providers, and primitive contracts exist; those process cases
are not implemented claims.

The host generates `feature_0001`, `currentstate_0001`, and
`desiredstate_0001`; compiles their types; links both states to the feature;
and adds specification `ppr:defines` plus selected process `ppr:realizes`.
The rich state meaning stays in the accepted v8 proposal.
`TargetFeatureSemanticReview` v2 checks semantic adequacy separately;
an incomplete gap re-enters PA retrieval/revision and is not accepted missing
information.

The research contribution is not that PA and RA exchange messages. It is the
combination of:

1. PA dynamically grounding an incomplete product requirement from only the
   relevant approved document, CAD, and RGB-D evidence, followed by an
   evidence-backed PA selection from the configured capable resources and live
   Cartesian validation of that exact choice.
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
generic cached overview is separated from generalized PA understanding and a
late evidence-cited ontology/context proposal; deterministic validation and its
typed-evidence gate alone accept assertions. One document retrieval returns all
ordered pages without a targeted inspection question. This supports newly
registered specifications and manuals without product-purpose prompts or a
hard-coded question such as where a particular gear is located.

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
  seven-assertion ABox projection under the current TBox.
- The workcell-profile-derived ABox records the configured process and resource
  individuals plus each resource's explicit process capabilities. The production
  profile currently yields `xarm6` and `ur5e` broadly `capableOf assembly`.
  Only the system adds the selected `processExecution` after PA-cited two-state
  reach evidence and exact selected-RobotAgent plan-only validation.
- The selected RA later supplies a separate versioned typed primitive-catalog
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
RA's future `PrimitiveProgramDraft` and primitive interfaces expose further
runtime inputs.

```text
requirement + ontology
        ↓
retrieve only currently relevant approved evidence
        ↓
PA authors one cited target_feature with current and desired states
        ↓
validate its exact seven-assertion projection without merging
        ↓
validate both unchanged candidate refs, hashes, and unique CAD size correspondences ↺
        ↓
separate PA semantic consistency review accepts or returns a revision gap ↺
        ↓
commit the accepted feature-state ABox
        ↓
unresolved processExecution activates PA allocation
        ↓
PA chooses a provisional resource without reselecting either state value
        ↓
check_reachability derives the gear pick and installed-shaft place targets ↺
        ↓
exact selected RobotAgent validates chained live Cartesian phases ↺
        ↓
system commits only the accepted PA choice
```

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
pages 1 through 6 together and in order. It uses neither a targeted question,
page ranking, nor RAG; large-document retrieval remains outside this scope.

The model never manages hidden host-authored information-need state. The system
validates native tool calls, exact replay keys, citations, hashes, and TBox
signatures. After grounding the feature and both states, unresolved
`processExecution` activates PA allocation directly. PA supplies the resource
choice to `check_reachability`; the tool derives both numeric verifier inputs
from the accepted state bindings and returns Cartesian phase evidence. Invalid,
unavailable, stale, ambiguous, or exhausted paths stop fail-closed without a
manufactured assertion.

The PA handoff is `PAContextGroundingCompletion` v6, which pins the accepted v8
proposal, both feature states, semantic review, evidence and allocation
presentation records, process and state-evidence choices, registry and workcell
snapshots, two-state reachability, plan-only RobotAgent validation, the
post-assignment ontology projection, `ResourceSelectionRecord` v4, and referenced
typed records without copying the target feature. There is no separate
`TaskTransitionContract`. The
selected RA then receives a reconstructed transient `target_feature`, bounded
resolved state values, selected-resource projection, fresh state, and its
complete current primitive-only catalog. The RA LLM authors a structural
`PrimitiveProgramDraft`; a deterministic binding preflight gathers all currently
unbound inputs without creating or repairing steps. Robot state and later
primitive-level feasibility remain RA-owned. Simulation allocation performs
only collision-aware Cartesian pick/place path validation through the exact
selected RobotAgent and never executes motion.
Product or scene gaps are deduplicated into one `MissingContextBatch` per round.

Formal PA completion pins both the exact `ResourceSelectionRecord` and the
`resource_grounding_host` assignment delta. The four execution assertions must
all cite that same selection record, and its selected resource, grounding-record hash,
ordered reach verdicts, and final `runsOnResource` value must agree when the
completion is reloaded.

For `assemble medium gear`, PA may retrieve the NIST manual, approved CAD, and
live RGB-D evidence; author current and desired feature states; and assign a
neutral candidate to each state. It then freely proposes one capable resource
and calls two-state reachability. The exact proposed RobotAgent validates both
chained Cartesian phases in plan-only mode. Rejection returns evidence to PA without an ABox
assignment or host-selected replacement. Acceptance commits exactly the PA
choice. `xarm6` is therefore a possible PA-authored outcome for the evaluated
scene, not a registry-order default, retrieval rule, or ontology entailment.

PA may service one batch through several existing single-source audited
operations, then returns one new versioned `CompositionContextBundle`. There is
no fixed semantic round count. Another batch round is allowed only after a new
accepted binding, changed need classification, or structurally different RA
draft demonstrates progress. Ambiguous or unavailable evidence, an unsupported
need, an identical request against unchanged context, or no new accepted result
stops fail-closed. `context understanding complete` means readiness for Phase 5,
not that every later primitive input is already available.

Phase 4.4 candidate discovery is a semantic join over the pinned workcell-profile
v2 and registry snapshots for the process selected from the authorized catalog.
Live Cartesian checks are deterministic evidence providers in simulation;
the retained physical path continues to use its configured safety checks. PA
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
    ├── one-CAD size association and camera-frame candidate center
    ├── location conversion from injected calibration
    └── optional pose estimation for orientation-sensitive consumers
```

Only PA and RA are agents. The observation provider and the perception
components shown above are controlled tools invoked within the Spec2Primitives
workflow. Related vision algorithms are combined into the two tools instead of
being modeled as additional agents.

These tools expose uniformly segmented observations and neutral candidate
handles. They do not label a source or target region. The architecture has no
`TargetFeatureGeometryRecord`: PA assigns approved candidates to current and
desired states, and numeric robot-frame locations are derived only when a
verifier needs them. No fixed document, CAD, RGB-D, or resource order is
prescribed.

The controller's grounding-readiness projection prescribes no evidence modality,
component, CAD choice, or candidate. The current deterministic implementation
accepts state values resolved from `RGBDSegmentationRecord` evidence only when
each cited CAD correspondence is unique. Retrieving an approved live observation invokes
deterministic preprocessing and segmentation; PA does not select the
segmentation algorithm or author that record. PA authority begins with its
choice of approved evidence handle and continues through its selection of the
neutral segmentation candidate, its `current_state` and `desired_state`
assignments, and its provisional resource. This is bounded PA autonomy inside a
fixed perception contract, not dynamic perception-chain selection or
unrestricted freewill. Descriptor-derived selection through
`_producer_descriptors`, `_required_record_plan`, and `_grounding_gap` remains
deferred unless the research claim is expanded to dynamic perception/context
chain selection.

RGB-D capture, calibrated deprojection, support-plane removal, connected-region
segmentation, mask persistence, one-CAD principal-size comparison,
camera-frame candidate-center measurement, generalized camera-frame
registration, injected-calibration frame conversion, and read-only processing
status are supporting infrastructure rather than the ICRA research
contribution. Phase 4.4 connects the required pose and frame-conversion records
to PA grounding only when the ontology-derived resource assignment cannot be
validated without them. These tools can report accepted, ambiguous, or rejected
records but do not establish general identity recognition, a pick point,
primitive composition, planning, or execution.

Intermediate mask, depth, CAD-fit, uncertainty, and source evidence remains
separately auditable even though the algorithms are grouped behind the two tool
boundaries. RA does not receive raw document or camera data. The official
future PA→RA input is the validated ontology projection, typed grounding
contract, and hash-pinned typed evidence records.

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

The Phase 4 study begins with five fresh interactions for each condition below.
Each interaction uses `product requirement: assemble Medium Gear`, a new
interaction root, the same approved source set and workcell snapshots, and no
records copied from an earlier run. Five repetitions provide an initial
artifact-complete study set; results are reported as counts and traces, not as a
statistical generality claim.

`interaction_d93d297a1ddc4bdc85cbbe60579a2b57` is excluded from every baseline
count and retained only as an immutable contaminated-prompt ablation. Fresh study
runs begin only after the relational-grounding gate and its smoke run pass.

| ID | Condition | Required observation |
| --- | --- | --- |
| 1 | Baseline with independently pinned presentation orders | PA authors the evidence retrievals, target feature, state-evidence assignments, and provisional resource. |
| 2 | Reverse the baseline evidence, neutral segmentation-candidate, and capable-resource presentation orders | A first-item policy must not explain all candidate or resource selections; every selection must still cite PA-visible evidence. |
| 3 | Return a rejected plan-only validation whenever PA provisionally chooses `xarm6`; accept no host-selected replacement | Rejection returns to PA, which authors a new choice or reports `insufficient_evidence`. A run in which PA initially chooses another resource is recorded as a non-triggered run, not rewritten. |
| 4 | Present two segmentation candidates that remain indistinguishable under the allowed document, CAD, RGB, depth, and calibration evidence | PA requests admissible additional evidence or clarification, or reports `insufficient_evidence`; it must not silently adopt the first candidate. |

The study audits `EvidencePresentationRecord`, `ProductAgentToolCall`,
`OntologyGroundingProposal`, `TargetFeatureSemanticReview`,
`AllocationPresentationRecord`, `ProductAgentAllocationToolCall`,
`ReachabilityCheckRecord`, `ResourceSelectionRecord`, and
`PAContextGroundingCompletion` in creation order. Ground-truth evaluation may
start only after the PA prediction is finalized. The experiment does not claim
that RGB-D segmentation, reachability, RobotAgent validation, or the Phase 4
stage order is PA-authored.

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
PA authors one cited target_feature, clarification, or insufficient evidence
        ↓
deterministic validation creates a transient provisional ABox
        ↓
host generates feature_0001, both states, and seven transient assertions
        ↓
validate both unchanged candidate refs, hashes, and unique CAD size correspondences ↺
        ↓
separate scaffold-free PA semantic consistency review accepts or returns a revision gap ↺
        ↓
commit the accepted feature-state ABox
        ↓
unresolved processExecution activates PA allocation
        ↓
PA chooses a provisional resource without revising currentstate or desiredstate
        ↓
check_reachability derives the gear pick and installed-shaft place targets ↺
        ↓
exact selected RobotAgent validates chained Cartesian phases in plan-only mode ↺
        ↓
system commits the exact accepted choice and v6 plan-only completion
        ↓
Phase 5.1 activates the selected exact resource_jid through the current adapter
        ↓
selected RA receives reconstructed target_feature + resource projection
        + fresh state + complete current primitive-only catalog
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
the Phase 5.1 contract-first RA assignment/state/catalog boundary and Phase 5.2A
structural draft. The current adapter can reuse or start the exact selected
context-only RobotAgent and obtain one LLM-authored symbol sequence, but it has
no live SPADE delivery, parameter binding, insertion-physics validation, or
robot execution.

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
