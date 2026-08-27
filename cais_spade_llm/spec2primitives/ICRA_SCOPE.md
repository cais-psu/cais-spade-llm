# ICRA 2027 Spec2Primitives Scope

## Research title

**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**

## Proposed-versus-implemented status

The end-to-end workflow in this document is proposed architecture. The current
package implements the PA interaction boundary through Phase 3.5, including
resumable user-intent clarification and a tamper-checked grounding-completion
record, together with the pre-RA Phase 4.3 grounding runtime, the Phase 4.0 ontology foundation, demand-driven document
and geometry producers, generalized camera-frame pose estimation, and separate
camera-to-robot frame-conversion infrastructure. The configured production path
stops when the `TaskTransitionDraft` inputs are grounded and records readiness
for Phase 5. No Phase 5 task
contract, RA adapter, resource-catalog ABox, primitive composer, validator
stack, or execution path is connected.

## Fundamental research challenge

The central research question is:

> Given a grounded task transition, fresh selected-resource context, and the
> complete selected-RA-authoritative primitive-only catalog with partial local
> executable contracts, can one RA-owned LLM composer author and revise
> validator-accepted
> `primitive_steps` without a product-specific recipe, completed
> `primitive_steps`, complete PDDL domain/problem, or hidden ground truth?

The intentional research gap inside the representation is that PA's grounded
high-level process has no asserted one-to-one `capableOf` match and no
task-to-primitive decomposition. The selected resource advertises only its
lower-level primitive-process interfaces. The RA LLM must interpret the process
meaning, outcome, grounded bindings, primitive descriptions, and fresh state to
propose the missing composition; deterministic validators then determine
whether that unchanged proposal is admissible.

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
   relevant approved document, CAD, and RGB-D evidence into a provenance-backed,
   robot-independent task-transition contract.
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

- The immutable TBox contains generic PPR classes, properties, and restrictions.
- PA fills a per-interaction ABox with evidence-backed requested-process,
  product, feature, and grounded-outcome individuals.
- The selected RA supplies a separate versioned resource-catalog ABox snapshot
  containing its resource individual, every primitive-process interface
  individual in that snapshot, and `resource capableOf primitive` assertions.
  The snapshot pins its fingerprint and cardinality for one composition attempt.

The pasted OWL is a mixed schema-and-instance graph rather than a pure TBox. It
remains a parser fixture only; its named individuals and task-specific
restrictions are not runtime context. The project-authoritative production TBox
is `ontology/spec2primitives_ppr_tbox.owl`; it contains only the approved
schema-only PPR view.

The requested assembly process may be linked through `realizes` to its grounded
outcome. It must not be linked to primitive offerings through `requires`,
`precedes`, direct high-level `capableOf`, or another expected set or order. A
read-only runtime projection joins the relevant task ABox and resource-catalog
ABox assertions for the RA LLM. The RA-authored `primitive_steps` candidate,
not an ontology entailment, creates the task-to-primitive connection
dynamically.

Backward Derivation and Forward Validation inspect only the unchanged candidate
over explicitly represented contract fields and resource-owned evaluator
results. They reject or report `unmodeled`; they do not search for, insert,
remove, reorder, parameterize, or repair a primitive. A complete engineered
PDDL model or black-box search could still provide another composer, so the
paper does not claim that classical planning is theoretically incapable or that
an LLM is universally necessary.

## Dynamic context-retrieval boundary

The next consumer, not the TBox, exposes required context. The TBox defines
legal meaning; it does not prescribe a document, CAD, observation, calibration,
or primitive recipe. Before allocation, PA evolves a `TaskTransitionDraft` and
compares its blocking inputs with a validated `ProductContextView`. After
allocation, the selected RA's `PrimitiveProgramDraft` and primitive interfaces
expose further runtime inputs.

```text
required consumer inputs - valid current context = ContextNeeds
        ↓
GroundingProducerDescriptor selects an authorized output-capable producer
        ↓
producer retrieves permitted evidence and returns supported facts or records
        ↓
validate, fingerprint, persist, and reassess
```

PA may dynamically choose approved document, exact caller- or corpus-authorized
CAD, fresh RGB-D, matching injected calibration, or an existing accepted record.
Those modalities are producer inputs, not the semantic needs exchanged by PA
and RA. Numeric values remain in `TypedContextBinding` records associated with
the ABox context. OWL open-world absence is never treated as completeness, and
no static all-modality checklist or expected primitive sequence is encoded.

The PA handoff requires only the minimum evidence-backed task identity, outcome,
and bindings needed for a robot-independent `TaskTransitionContract`. The
selected RA then retrieves a bounded task/resource projection, fresh state, and
its complete current primitive-only catalog. RA authors a structural
`PrimitiveProgramDraft`; a deterministic binding preflight gathers all currently
unbound inputs without creating or repairing steps. Robot state, limits, IK,
collision, grasp, release, trajectory, and execution gaps remain RA-owned.
Product or scene gaps are deduplicated into one `MissingContextBatch` per round.

PA may service one batch through several existing single-source audited
operations, then returns one new versioned `CompositionContextBundle`. There is
no fixed semantic round count. Another batch round is allowed only after a new
accepted binding, changed need classification, or structurally different RA
draft demonstrates progress. Ambiguous or unavailable evidence, an unsupported
need, an identical request against unchanged context, or no new accepted result
stops fail-closed. `context understanding complete` means readiness for Phase 5,
not that every later primitive input is already available.

Resource discovery and allocation remain inherited prior work. Both nominal
and recovery cases consume the selected exact `resource_jid` and use unicast;
Spec2Primitives does not broadcast or reallocate.

## Perception tool boundary

```text
PA
├── document evidence tool
│   ├── PDF text extraction
│   └── VLM diagram interpretation
│
└── RGB-D/CAD grounding tool
    ├── RGB-D capture and preprocessing
    ├── minimal camera-local segmentation
    ├── one-CAD size association and camera-frame candidate center
    ├── generalized camera-frame pose estimation
    └── camera-to-robot frame conversion from injected calibration
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
contribution. These standalone tools can report accepted, ambiguous, or
rejected records but do not establish general identity recognition, a pick
point, complete scene grounding, context assessment, primitive composition,
planning, or execution. They are not connected to the production PA/RA path.

Intermediate mask, depth, CAD-fit, uncertainty, and provenance evidence remains
separately auditable even though the algorithms are grouped behind the two tool
boundaries. RA does not receive raw document or camera data. PA sends RA only
the grounded assembly task and its provenance-backed requirements.

## Research thesis and evaluation requirement

The paper thesis is:

> Given a grounded task-transition contract, RA-owned agentic composition over
> the complete selected-resource semantic primitive catalog, coupled to non-synthesizing
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
PA evolves a TaskTransitionDraft over its ProductContextView
        ↓
PA resolves currently blocking ContextNeeds through authorized producers
        ↓
PA validates and persists evidence-backed facts, then reassesses
        ↺ each producer retrieval remains one auditable source operation
        ↓
PA projects a robot-independent TaskTransitionContract
        ↓
inherited allocation supplies one selected resource_jid; PA unicasts
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
the narrow PA context boundary and standalone supporting perception records.
The current production path still has no RA behavior, insertion-physics
validation, or robot execution.

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
