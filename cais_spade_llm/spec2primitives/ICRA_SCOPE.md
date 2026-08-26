# ICRA 2027 Spec2Primitives Scope

## Research title

**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**

## Fundamental research challenge

The central research question is:

> Given a grounded task transition, fresh selected-resource context, and exactly
> eight semantic primitive interfaces with partial local executable contracts,
> can one RA-owned LLM composer author and revise validator-accepted
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
   fresh selected-resource state and the current exact-eight semantic primitive
   interfaces with partial local executable contracts, then agentically
   authoring primitive selection, order, bindings, and parameters.
3. Non-synthesizing contract and robot-local validators returning concrete
   findings so the same RA LLM can author a revised complete candidate without
   a validator inserting or repairing steps.

Exact-ref retrieval, PA and RA live cards, ordered interaction records, and
simulation are supporting infrastructure and evaluation evidence. They are not
the fundamental contribution by themselves.

## Partial formal-model boundary

Each of the exact eight resource-owned catalog entries exposes its fixed
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
  containing its resource individual, the exact eight primitive-process
  interface individuals, and `resource capableOf primitive` assertions.

The pasted OWL is a mixed schema-and-instance graph rather than a pure TBox. It
remains a parser fixture only; its named individuals and task-specific
restrictions are not runtime context. The production ontology input must expose
an approved schema-only TBox view.

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

PA and RA use the same bounded pattern:

```text
objective + accumulated authority-owned context
        ↓
identify one concrete missing input
        ↓
retrieve or compute one result through its owning authority
        ↓
validate, persist, and reassess
        ↓
repeat, hand off, return missing_context, or stop fail-closed
```

Their authority and completion conditions differ:

- PA selects relevant approved document, CAD, or fresh RGB-D evidence from the
  evolving requirement and ABox. It finishes when it can project a grounded
  robot-independent task-transition contract with no currently identified
  blocking product or scene ambiguity.
- The selected RA first retrieves three mandatory prerequisites: a bounded
  read-only projection of the current task and selected-resource ABoxes, a fresh
  selected-resource snapshot, and the complete exact-eight interface catalog.
  The projection contains the requested process and outcome plus all eight
  available offerings, but no process recipe. RA then retrieves only
  resource-local context identified by an unbound input, stale state, interface
  need, or validator finding. It finishes only when one unchanged, fully bound
  candidate passes every applicable declared-contract and resource-owned check.
- Product or scene gaps return to PA as structured `missing_context`; robot
  state, resource limits, IK, collision, grasp, release, trajectory, and
  execution gaps stay with RA. PA never authors `primitive_steps`, and RA never
  reinterprets raw product evidence or asks the user directly.

Ontology-assisted retrieval operates over semantic needs, not modality names.
A primitive interface asks for a grounded typed input. If current ontology or
typed context does not support it, PA matches that need to an authorized
producer's `can_produce` descriptor and retrieves only the evidence listed by
that producer's `may_require` descriptor. Thus documentation may be selected to
ground process or product meaning, while STL plus RGB-D and calibration may be
selected to ground a scene binding or pose. These are runtime dependencies, not
a fixed requirement that every interaction load every source. The descriptor
fields are routing metadata outside the authoritative TBox; no new ontology
predicate is invented.

OWL provides type and relation inference but cannot use absence as a
closed-world completeness decision. SHACL or an equivalent boundary validator
may flag a currently requested input as missing; it must not encode a static
all-modality checklist or the expected primitive sequence.

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
    ├── RGB segmentation
    ├── depth geometry
    └── CAD registration
```

Only PA and RA are agents. The observation provider and the perception
components shown above are controlled tools invoked within the Spec2Primitives
workflow. Related vision algorithms are combined into the two tools instead of
being modeled as additional agents.

Intermediate mask, depth, CAD-fit, uncertainty, and provenance evidence remains
separately auditable even though the algorithms are grouped behind the two tool
boundaries. RA does not receive raw document or camera data. PA sends RA only
the grounded assembly task and its provenance-backed requirements.

## Research thesis and evaluation requirement

The paper thesis is:

> Given a grounded task-transition contract, RA-owned agentic composition over
> exact-eight semantic primitive interfaces, coupled to non-synthesizing
> resource-validator feedback, improves candidate validity and robot-local
> feasibility relative to the same-input LLM composer without
> validation-driven revision.

The controlled Medium Gear case is the development starting point, not
sufficient evidence of general primitive composition. The final evaluation must
withhold product-specific assembly programs, completed `primitive_steps`, and a
complete task-specific action model; provide only the exact-eight executable
primitive interfaces and partial contracts; include CAD distractors; vary
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
baseline. Monolithic, no-retrieval, and stale-context conditions are supporting
PA-pipeline ablations. Retrieval, schema validation, simulation, observed
outcome, and physical execution remain distinct claims.

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
PA identifies one unresolved product or scene need
        ↓
PA retrieves one relevant approved source and invokes its controlled tool
        ↓
PA validates and persists evidence-backed facts, then reassesses
        ↺ another document, CAD, or RGB-D request only if still needed
        ↓
PA projects a versioned robot-independent task-transition contract
        ↓
inherited allocation supplies one selected resource_jid; PA unicasts
        ↓
selected RA retrieves the current task/resource ontology projection
        + fresh state + complete exact-eight interface catalog
        ↓
RA dynamically retrieves any additional selected-resource context it needs
        ↓
RA LLM authors one complete candidate primitive_steps program
        ↓
non-mutating declared-contract + resource + physical + outcome checks
   ↙ product/scene gap       ↓ robot-local finding        ↘ accepted unchanged
PA grounds and versions    finding returns to the RA         fresh-state recheck
the task update            LLM for a new candidate                    ↓
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

## Scene-only milestone boundary

The scene-only milestone provides the isolated structure, the research workflow,
a narrow launcher for the no-hardware `gazebo_dual_spec2primitives` simulation, the
dedicated `table_spec2primitives.world`, and a local placeholder chat. It does not
define functional schemas, implement recognition, VLM, PA or RA behavior,
validate insertion physics, or execute robots.

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
