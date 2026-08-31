# ICRA 2027 Spec2Primitives Scope

## Research title

**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**

## Proposed-versus-implemented status

The end-to-end workflow in this document is proposed architecture. **Phase 4 is
implemented through Phase 4.4 under the current framework.** The package also
implements the PA interaction boundary through Phase 3.5. PA uses one native
`retrieve` tool and returns a direct ontology proposal. Phase 4.4 validates that
proposal against a transient ABox, derives the active consumer's typed
prerequisite closure, and returns any source-evidence gap to PA without
committing semantic assertions.
After the target-specific location chain passes, the system performs
manifest-backed coarse reach checks and commits the selected resource as a
`processExecution` assignment. The exact `xarm6` and `ur5e`
identities remain pinned to their shared manifests; their broad
`capableOf assembly` assertions do not encode reach or a primitive sequence. No
live Phase 5 RA connection, primitive composer, validator stack, or execution
path is connected. Phase 5.1 now implements only the contract-first assignment
envelope and injected-runtime state/catalog snapshot boundary.

Phase 4 completion is the implemented location-based pre-RA grounding and
resource-assignment boundary. Cross-camera fusion, an active
orientation-sensitive consumer, RA composition, robot-local validation,
execution, and observed outcomes are not part of the completed Phase 4 claim.

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
grounded assembly requirements
        ↓
robot-independent assembly plan
        ↓
robot-specific primitive_steps
        ↓
validator-accepted primitive program
        ↓
execution
        ↓
observation-backed realized outcome
```

The research contribution is not that PA and RA exchange messages. It is the
combination of:

1. PA dynamically grounding an incomplete product requirement from only the
   relevant approved document, CAD, and RGB-D evidence, followed by an
   evidence-backed system assignment of the task to one predefined resource.
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
- PA proposes a per-interaction ABox only from the requirement and authorized,
  hash-pinned evidence; the system accepts only individuals and relations that
  the current TBox can represent.
- The predefined Workcell ABox records `xarm6` and `ur5e` as resources broadly
  `capableOf assembly`. Only the system adds the selected `processExecution` after
  typed evidence and manifest-backed reach validation.
- The selected RA later supplies a separate versioned typed primitive-catalog
  snapshot. It pins exact symbols, fingerprint, and cardinality without
  publishing primitive implementations through `capableOf` assertions.

The pasted OWL is a mixed schema-and-instance graph rather than a pure TBox. It
remains a parser fixture only; its named individuals and task-specific
restrictions are not runtime context. The project-authoritative production TBox
is `ontology/spec2primitives_ppr_tbox.owl`; it contains only the approved
schema-only PPR view.

Any proposed `realizes` assertion is accepted only when it cites authorized
evidence, uses a correctly typed process and feature, and passes the property
signature validation. No relation is required merely because the requirement
names a process. A high-level process must not be linked to primitive offerings
through `requires`, `precedes`, or another expected set or order. Broad
resource-to-`assembly` capability supports semantic candidate discovery only. A
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
validate a proposal against a transient semantic ABox
        ↓
derive the active consumer's typed prerequisite closure
        ↓
missing source evidence returns to PA retrieval ↺
        ↓
accepted target-specific location enables reach evaluation
        ↓
system commits the semantic ABox and processExecution assignment
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

The model never manages intermediate statements or information-need state. The
system validates native tool calls, exact replay keys, citations, hashes, and
TBox signatures. After a provisional proposal, it computes the non-persisted
`ResourceAssignmentNeed` from the provisional graph and current typed records.
Descriptor prerequisites determine which records are missing and which approved
evidence handles can produce them. PA may retrieve again within the bounded
investigation; invalid, unavailable, stale, ambiguous, or exhausted paths stop
fail-closed without a manufactured assertion.

The PA handoff is the validated post-assignment ontology projection,
`ResourceSelectionRecord`, and minimum hash-pinned typed evidence. There is no
separate `TaskTransitionContract`. The selected RA then retrieves that bounded
task/resource projection, fresh state, and its complete current primitive-only
catalog. RA authors a structural
`PrimitiveProgramDraft`; a deterministic binding preflight gathers all currently
unbound inputs without creating or repairing steps. Robot state, limits, IK,
collision, grasp, release, trajectory, and execution gaps remain RA-owned.
Product or scene gaps are deduplicated into one `MissingContextBatch` per round.

Formal PA completion pins both the exact `ResourceSelectionRecord` and the
`resource_grounding_host` assignment delta. The four execution assertions must
all cite that same selection record, and its selected resource, grounding-record hash,
ordered reach verdicts, and final `runsOnResource` value must agree when the
completion is reloaded.

For `assemble medium gear`, PA may first retrieve the NIST manual and
`Gear_Medium.STL` and propose the feature realized by `assembly`. Validation
creates a provisional graph but writes no semantic assertion. The active coarse
resource consumer declares `RobotFrameLocationRecord`; if observation-produced
segmentation is absent, the descriptor closure reports that source-evidence gap
to PA. After PA retrieves the approved live observation, target-cited CAD plus
segmentation yields correspondence, correspondence plus calibration yields the
required location, and only then can the system accept the semantic ABox and
select a reachable resource. In the current evaluated scene this produces
`xarm6`; the identifier is not a retrieval rule or an ontology entailment.

PA may service one batch through several existing single-source audited
operations, then returns one new versioned `CompositionContextBundle`. There is
no fixed semantic round count. Another batch round is allowed only after a new
accepted binding, changed need classification, or structurally different RA
draft demonstrates progress. Ambiguous or unavailable evidence, an unsupported
need, an identical request against unchanged context, or no new accepted result
stops fail-closed. `context understanding complete` means readiness for Phase 5,
not that every later primitive input is already available.

Phase 4.4 candidate discovery is a semantic join over the predefined Workcell;
coarse allocation is deterministic supporting infrastructure, not a claimed
optimal allocator. Both nominal and recovery cases consume the selected exact
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
PA returns a direct ontology proposal, clarification, or insufficient evidence
        ↓
deterministic validation creates a transient provisional ABox
        ↓
the unique defines/realizes join identifies the primary feature
        ↓
the active consumer declares RobotFrameLocationRecord
        ↓
descriptor closure reports any missing source evidence to PA ↺
        ↓
accepted target-specific location gates semantic acceptance
        ↓
manifest-backed reach selects one resource in configured profile order
        ↓
system commits the semantic ABox and processExecution assignment
        ↓
future Phase 5 unicasts to the selected exact resource_jid
        ↓
selected RA retrieves the current task/resource ontology projection
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
the Phase 5.1 contract-first RA assignment/state/catalog boundary. The current
live production path still ends at the Phase 4.4 resource assignment. It has no
live RA connection, primitive composition, insertion-physics validation, or
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
