# Spec2Primitives Research Positioning

## Working title

**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**

`Spec2Primitives` is preferred over `Spec2Actions` because the proposed output
is not an arbitrary action sequence. It is a robot-specific program assembled
from a resource-owned primitive catalog. The term `actions` is too broad and
would make the distinction from Manual2Skill's existing robot action generation
less clear.

In the name, `Spec` refers to the supplied product specification together with
its approved supporting evidence. The specification may initially be incomplete
or underspecified; the framework does not assume that it is a complete formal
specification.

## Abstraction level

The research spans the following abstraction chain:

```text
product specification
        ↓
grounded assembly requirements
        ↓
ontology-backed robot-independent task transition
        outside the executable coverage of the
        predefined composite-function library
        ↓
selected-resource primitive-only catalog and fresh context
        ↓
robot-specific primitive_steps authored at runtime
        ↓
validator-accepted primitive program
        ↓
execution
        ↓
observation-backed realized outcome
```

For the nominal case, PA supplies the grounded robot-independent task. For the
recovery transfer case, the existing recovery framework supplies an already
selected, validated, and resource-assigned recovery event. The common
Spec2Primitives contribution begins at the selected RA composition boundary.

The composition contribution occurs at a finer-grained execution abstraction
than product-level assembly planning. It is not itself low-level robot control.
Trajectory generation, force control, velocity commands, and joint commands
remain below `primitive_steps`.

## Relationship to the prior RCIM work

The prior paper, [Adaptive task planning and coordination in multi-agent
manufacturing systems using large language
models](https://doi.org/10.1016/j.rcim.2026.103245), addresses unforeseen
requirement interpretation, manufacturing-knowledge retrieval, process planning,
capability matching, resource assignment, and runtime parameter discovery. Its
planner operates over predefined executable functions such as
`move_to_pick_location`, `pick_part`, `move_loaded_to_destination`, and
`place_part`.

Those functions are already composed of lower-level primitives: their internal
primitive selection, ordering, and binding policy exist before the new
requirement arrives. Consequently, selecting or ordering those functions is not
the primitive-composition problem studied here. PA grounding, PA–RA
communication, registered-resource selection, and heterogeneous context
retrieval are inherited or supporting mechanisms when reused in the ICRA
system; they are not new Spec2Primitives contributions.

Spec2Primitives studies a composite-function coverage gap. It begins after an RA
has been selected and no valid selection, ordering, and parameterization of its
available predefined composite functions can satisfy the required grounded
transition. The selected RA exposes only the exact eight resource-owned
execution primitives at the composition boundary, and the RA must author the
missing task-specific composite program at runtime:

```text
prior RCIM abstraction:
requirement → select and order predefined composite functions → assigned execution

Spec2Primitives abstraction:
grounded transition outside composite-function coverage
        + selected RA + eight execution primitives
        → newly authored primitive_steps → validation → revision → execution
```

Spec2Primitives is therefore positioned as a narrower, deeper continuation of
the RCIM work, not as a replacement for or globally better version of it. Its
novelty must be evaluated at the capability-to-executable-primitive-program
boundary. Merely moving function selection from PA to RA, or renaming an
existing function as a primitive, would not constitute the claimed advance.

## Neuro-symbolic research position

Spec2Primitives is explicitly a neuro-symbolic framework. The lightweight PPR
ontology is an architecturally required formal backbone, not an optional prompt
attachment. The proposed method requires it to maintain shared task and resource
meaning as PA retrieves evidence and RA constructs a primitive program. This is
not a claim that an ontology is universally necessary for every possible
primitive-composition method, and ontology schema design is not presented as a
standalone ICRA contribution.

The symbolic side consists of more than OWL alone:

- the immutable TBox defines the shared `product`, `feature`, `process`,
  `resource`, and `capability` vocabulary;
- the evidence-backed interaction ABox records the grounded requested process,
  product facts, features, and required outcome for the current interaction;
- the RA-authoritative, versioned primitive catalog snapshot enumerates the
  complete callable composition surface for that resource and interaction, and
  the selected-RA resource-catalog ABox mirrors its exact eight
  primitive-process offerings without encoding a task-specific decomposition;
- typed context records retain poses, observations, calibration, tolerances,
  resource state, and other values when the loaded TBox has no predicate for
  them; and
- partial primitive contracts and deterministic validators check the candidate
  over the explicitly represented symbolic, data-dependency, resource, and
  outcome surface.

Within the proposed method, the ontology is the canonical semantic interface for
entity identity, evidence links, PA–RA meaning, and primitive-input type and
entity bindings. Typed context and status records represent unresolved needs,
conflicts, numeric values, and freshness. Together with the external
`can_produce` and `may_require` producer descriptors, the bounded projection
supports routing an already identified typed need without hard-coding PDF, CAD,
or RGB-D retrieval order.

The composite-function coverage gap and `exact eight` composition surface are
established from RA-authoritative, versioned catalog snapshots that are treated
as closed operational enumerations for that resource and interaction. They are
not inferred from the absence of `capableOf` assertions under OWL open-world
semantics. The resource-catalog ABox mirrors the authoritative primitive snapshot
for semantic projection; it does not itself prove catalog completeness. A
bounded SHACL check may test only an explicitly declared shape or already
identified required input at a local validation boundary. It does not establish
global context completeness.

The neural side makes the decisions that are intentionally absent from the
formal model:

- PA identifies the next relevant product or scene knowledge need and uses only
  controlled evidence producers to extend the interaction context;
- the selected RA LLM authors a complete, variable-length `primitive_steps`
  candidate by choosing, ordering, repeating, and binding the exact eight
  primitives; and
- after deterministic rejection, the RA LLM interprets the categorized findings
  and authors a structurally revised candidate.

The resulting closed loop is:

```text
TBox + interaction ABox + selected-RA resource-catalog ABox
        + typed context refs + partial primitive contracts
        ↓ bounded grounded projection
        ├─ PA neural decision → controlled evidence producer
        │       → validated evidence delta → ABox/typed-context update → reassess
        │
        └─ RA neural decision → complete candidate primitive_steps
                → symbolic, resource-owned, physical, and outcome checks
                → runtime validation record → accept or neural structural revision
```

An LLM-authored candidate remains a hypothesis in a runtime record. It does not
become an evidence-backed ABox assertion, and neither OWL inference, SHACL, nor a
validator may insert, remove, reorder, bind, or repair its primitive steps.
Only a validated, evidence-backed producer delta may update the interaction
ABox; candidate findings and accepted or rejected `primitive_steps` remain typed
runtime records.

The finer-grained neural decision boundary relative to the prior RCIM
abstraction has a precise operational meaning:

```text
prior RCIM neural decision:
select, order, and parameterize predefined composite functions

Spec2Primitives neural decision:
author and structurally revise a previously unavailable program over
the exact eight primitives, including its length, selection, order,
repetition, intermediate steps, and grounded parameter bindings
```

The paper-ready position is:

> Spec2Primitives is a neuro-symbolic framework in which an ontology-backed
> context model provides a shared formal representation for evidence-backed
> product intent, resource capabilities, and primitive bindings, while the
> selected RA LLM dynamically synthesizes and revises a previously unavailable
> primitive program. The ontology deliberately does not encode the
> task-to-primitive decomposition; deterministic symbolic and physical
> validators assess the unchanged LLM-authored candidate without composing or
> repairing it.

A precise description is:

> Manual2Skill reasons primarily about product-level assembly structure and
> target poses, whereas Spec2Primitives reasons at the robot execution-program
> level by dynamically selecting, ordering, parameterizing, validating, and
> revising `primitive_steps`.

Manual2Skill may still produce trajectories that are numerically lower-level
than `primitive_steps`. The defensible claim is therefore that Spec2Primitives
performs explicit reasoning and composition at a finer-grained execution-program
abstraction, not that its entire pipeline is lower-level than Manual2Skill.

## Definition of a primitive

In this work:

> A robot primitive is a resource-owned, parameterized operation exposed as an
> atomic executable unit at the RA composition boundary. It encapsulates its
> lower-level controller, planner, estimator, or device service; accepts defined
> inputs; and returns explicit status and direct execution or observation
> evidence. A primitive performs one bounded physical or information-producing
> operation and does not encode a complete product-specific assembly procedure.

`Atomic` is relative to the RA composition interface. A primitive may internally
invoke a motion planner, execute many controller cycles, or run an estimator
while remaining one operation from the perspective of RA.

A predefined composite function is different: it is a callable capability whose
implementation already fixes a sequence or policy over multiple execution
primitives before the current interaction. Thus `move_to_pick_location`,
`pick_part`, `move_loaded_to_destination`, and `place_part` are predefined
composite functions in the prior RCIM abstraction, even if they appear as one
call in a resource API. They are not members of the Spec2Primitives eight-symbol
catalog. An accepted `primitive_steps` program may realize equivalent behavior,
but that task-specific composition did not exist before the RA authored it.

Each primitive catalog entry is a semantic executable interface. It should
describe:

- the exact primitive symbol and natural-language operation;
- the operation attempted;
- typed inputs, parameters, outputs, and limits;
- the resource invocation binding and termination status;
- direct evidence or guaranteed local results;
- the resource-owned evaluator used to determine feasibility; and
- only the resource-local state conditions and effects explicitly modeled for
  validation.

Those state conditions and effects are partial annotations. An omitted
condition or effect is unknown, not satisfied. The eight interfaces collectively
do not provide an exhaustive symbolic state vocabulary, global transition
model, task-specific goal decomposition, primitive order, or complete PDDL
domain/problem. The combined result of several primitives may establish an
assembly outcome that no individual primitive can establish alone.

Illustrative examples from the earlier framework design are:

- `compute_place_targets`: produces placement-target evidence;
- `move_cartesian`: attempts a bounded Cartesian motion; and
- `release_part`: commands release and produces direct gripper or custody
  evidence.

These examples illustrate the abstraction and are not a committed
Spec2Primitives schema or implemented primitive catalog. By contrast,
`assemble_medium_gear` would be a product-specific assembly program rather than
an atomic robot primitive.

This usage is consistent with the action-primitive and behavior-primitive
literature:

- An industrial robotics survey defines a primitive as the atomic operation
  closest to hardware, usable as a parameterized building block and potentially
  able to return information. Its examples include Move, Open, and Close.
  [Pantano et al.](https://elib.dlr.de/192304/1/pantano2022capability_copyright.pdf)
- MAPLE treats behavior primitives as parameterized, temporally extended
  functional modules such as grasping and pushing.
  [MAPLE](https://arxiv.org/abs/2110.03655)
- ARCH treats Grasp, Place, Move, and Insert as parameterized assembly
  primitives. Each may internally use a motion planner or learned controller,
  while a higher-level policy selects and composes them.
  [ARCH](https://arxiv.org/abs/2409.16451)
- Dynamical Movement Primitives refer more narrowly to trajectory-generating
  dynamical systems. Spec2Primitives does not use `primitive` in that sense.
  [Ijspeert et al.](https://pubmed.ncbi.nlm.nih.gov/23148415/)

Because the literature does not use one universal granularity, the paper must
state this operational definition before introducing `primitive_steps`. The
paper should use `robot primitive` or `behavioral primitive` when clarification
is needed and should not describe these operations as Dynamical Movement
Primitives or trajectory-level motion primitives.

## Definition of dynamic primitive composition

> Dynamic primitive composition is the selected RA's runtime, agentic proposal
> and revision of ordered, parameterized calls from its current resource-owned
> primitive-only catalog, conditioned on a versioned task-transition contract
> outside predefined composite-function coverage and fresh resource context. The
> resulting task-specific composite program did not exist as a callable function
> before the interaction.

The RA LLM authors each complete candidate `primitive_steps` program. Primitive
contracts and deterministic backward, forward, physical, and projected-outcome
checks only accept or reject that unchanged candidate and return findings. They
never select, insert, remove, order, parameterize, or repair primitive steps;
RA authors the next complete candidate from the findings.

The word `dynamic` requires more than replacing parameters in a fixed template.
The evaluation must demonstrate that:

1. No valid selection, ordering, and parameterization of the available
   predefined composite functions satisfies the task-transition contract, and
   no product-specific assembly program or completed `primitive_steps` is
   available or supplied.
2. RA receives the assigned nominal task or selected recovery event, its
   required outcome and grounded refs, the complete current eight-symbol catalog
   of semantic primitive interfaces with partial local executable contracts,
   and fresh resource state, but no completed `primitive_steps`, task-specific
   PDDL domain/problem, or product-specific executable program.
3. RA agentically authors primitive selection, ordering, bindings, and
   parameters at runtime.
4. Changes in robot state or the resource-specific availability and feasibility
   of the eight primitives can change the resulting primitive program.
5. Concrete contract, state, IK, collision, trajectory, or outcome findings can
   cause RA to author a structurally revised candidate rather than only perform
   a blind retry or parameter perturbation.
6. Different assembly requirements require different composition structures,
   rather than variants of one stored sequence.
7. The same RA composition entrypoint, prompt policy, eight-symbol catalog
   interface and versioning policy, context protocol, candidate schema,
   validators, and revision loop operate on nominal and recovery-origin tasks
   without an origin-specific recipe.

A stored nominal program, product-specific macro, `robot_task_program`,
capability decomposition, or preauthored recovery sequence is a modeled
baseline and is not evidence of dynamic primitive composition.

Likewise, invoking `move_to_pick_location`, `pick_part`,
`move_loaded_to_destination`, `place_part`, or an equivalent predefined
composite function is prior-function reuse, not dynamic primitive composition.
Every reported nominal or recovery composition case must preserve an auditable
selected-resource snapshot showing that the transition is outside the executable
coverage of the predefined composite-function library and that the composer
received only the exact eight primitives. Each case must require at least one
structural primitive decision such as selection, ordering, repetition, or an
intermediate step; merely changing parameters on an existing composite function
is insufficient.

Before an experiment, audit the selected RA's complete public callable interface
and establish that no valid selection, ordering, or parameterization of its
available predefined composite functions can satisfy the grounded
task-transition contract, while the necessary exact-eight primitive interfaces
remain available. Perform this coverage audit from the task contract and public
function semantics without reading an expected primitive program. Merely hiding
an applicable composite function from the prompt is not an admissible case.

The ICRA study freezes the allowed primitive vocabulary to exactly the eight
user-defined symbols and one contract schema. Each selected RA must still expose
truthful resource-owned parameter limits, state requirements, and evaluators;
those values may differ across resources and are not normalized to make the two
cases appear identical.

## Agentic composition and validation boundary

The operational method is:

```text
versioned task-transition contract + task/resource ontology projection
        + fresh selected-RA state
        + exact-eight semantic primitive interfaces and partial contracts
        + grounded context refs
        ↓
RA LLM proposes candidate primitive_steps
        ↓
schema, catalog, parameter, binding, and provenance checks
        ↓
backward modeled-condition coverage
        ↓
forward resource-state and data-dependency validation
        ↓
IK, collision, grasp, release, trajectory, and operational validation
        ↓
deterministic checks of explicitly represented projected-outcome constraints
        ↓
accept unchanged candidate or return findings to RA for agentic revision
```

Backward Derivation traverses only the candidate authored by RA. Over the
represented surface, it checks whether declared effects support modeled
required conditions and whether each recursively introduced precondition is
supported by an earlier candidate effect or, at the initial-state boundary, by
fresh state or evidence. Forward Validation checks the same candidate in
execution order and projects only declared guaranteed local effects. These
checks cover typed dependencies and explicitly modeled conditions; neither
operation searches the catalog, inserts or repairs a primitive, establishes
global plan completeness, or infers unmodeled causal semantics.

Primitive contracts expose only their declared parameters, preconditions,
guaranteed local effects, outputs, and direct evidence. They are intentionally
not an exhaustive symbolic model of perception, continuous geometry, IK,
collision, trajectory, or realized assembly state. Each selected-resource
offering is represented as a `process` individual for the read-only ontology
projection, while its detailed executable contract remains a typed resource
record. This instance representation is not a preauthored symbolic task graph;
contracts and validators must still represent every safety- or
causally-critical requirement that the system claims to check.

For example, `release_part` may guarantee that the gripper opens and custody is
released. It does not by itself guarantee that the part is on the target or
assembled. A separate task-outcome check requires the validated target-pose and
process evidence before execution, and observation evidence confirms the
realized result afterward. A projected-outcome check returns `unmodeled` and
stops fail-closed when the requested constraint lies outside its represented
coverage. If an intermediate requirement appears in no contract, invariant,
primitive evaluator, physical check, or outcome check, the system cannot claim
to have detected or validated its omission.

The paper therefore does not claim that LLMs are universally necessary for
primitive composition or that classical planning cannot solve these tasks. In
the evaluated partial-model setting, no complete product-specific executable
program or complete finite symbolic search domain is supplied, so the RA LLM is
the framework's agentic candidate generator and validators cannot replace it as
the composer. A sufficiently complete symbolic domain could use another search
or planning method. The empirical claim must come from equal-input comparisons:
a symbolic composer limited to machine-readable fields on the same partial
interfaces, bounded black-box sequence search with the same validator access
and query budget, LLM-only generation, and the full agentic generation plus
validation-revision method. Domain coverage, abstention, and applicability must
be reported separately. A classical planner supplied with a separately
engineered complete PDDL domain is an extra-information oracle rather than an
equal-input baseline. Its success would measure the value and cost of additional
modeling; symbolic failure outside represented coverage cannot support a
universal LLM-necessity claim.

## One composition core, two task origins

```text
nominal: grounded PA task-transition contract ─┐
                                                ├─→ same RA-owned agentic composition core
recovery: validated resource-assigned event ───┘
                                                           ↓
                                      candidate → validators → RA revision
```

The nominal case is the primary end-to-end study: PA begins with a raw assembly
requirement whose grounded transition is outside the selected RA's predefined
composite-function library coverage and has no stored nominal task program,
grounds the requirement, and supplies the robot-independent task-transition
contract. The recovery case is a transfer study: the existing recovery framework
begins with the fault and runtime state and supplies an already selected,
validated, and resource-assigned recovery event with the same
composite-function coverage gap. A read-only adapter maps that event into the
same composition contract without providing its primitive decomposition.

After this boundary, each origin uses its exact selected RA through the same
composer implementation, eight-symbol catalog interface, context-request
surface, candidate schema, validation sequence, feedback loop, and execution
boundary. Nominal perception and planning and the existing recovery event
generation, selection, allocation, DES/CCA reasoning, approval, and safety
validation remain distinct upstream mechanisms and are reported separately.
The recovery study tests transfer of the composer; it is not a second
recovery-planning contribution.

## PA and RA dynamic context retrieval

PA and the selected RA use the same bounded control pattern:

```text
current objective + accumulated authority-owned context
        ↓
identify one concrete blocking need
        ↓
retrieve or compute one result through its owning authority
        ↓
validate, fingerprint, persist, and merge supported information
        ↓
reassess
        ↓
retrieve again, hand off, return missing_context, or stop fail-closed
```

The similarity is control-flow symmetry, not a shared knowledge pool. Both
loops preserve the exact request, producer, refs, provenance, freshness,
failure, and reassessment result; neither uses a fixed all-source order or
treats budget exhaustion as readiness.

- PA starts from the exact requirement and interaction ABox. It dynamically
  selects relevant approved document, CAD, or fresh RGB-D evidence through
  controlled producer-capability descriptors, and produces a grounded
  robot-independent task-transition contract. It never retrieves robot state,
  primitive interfaces, or feasibility results.
- RA first retrieves a bounded task-relevant ontology projection, the mandatory
  fresh selected-resource snapshot, and its complete exact-eight interface
  catalog. It then dynamically retrieves only resource-local context exposed by
  an unbound parameter, stale state, interface requirement, or validator
  finding. It never reinterprets raw document, CAD, RGB, or depth evidence.
- A product or scene gap becomes structured `missing_context` routed to PA. A
  robot-state, resource-limit, IK, collision, grasp, release, trajectory, or
  execution gap stays with RA. RA never asks the user directly.
- PA readiness means that a robot-independent task contract is grounded. RA
  readiness means that one unchanged candidate is fully bound and accepted by
  every applicable declared-contract and resource-owned check. Neither agent
  certifies the next authority's work or the post-execution outcome.

The dependency direction is deliberately semantic:

```text
primitive interface or task needs one grounded typed input
        ↓
current ABox/typed context does not support it
        ↓
authorized producer can_produce that input kind
        ↓
producer may_require particular approved evidence
        ↓
PA retrieves that evidence and runs the producer
```

The primitive does not consume raw PDF, STL, RGB, depth, or calibration data.
Those sources belong to PA's controlled tools. The `can_produce` and
`may_require` fields are typed producer-descriptor metadata outside the
authoritative TBox; their values use approved ontology or typed-context kinds.
This keeps source selection dynamic without adding a static all-modality shape
or task-specific source order. OWL supports semantic matching, while a bounded
SHACL or equivalent check may detect that an already identified input is absent;
neither mechanism determines the primitive sequence.

## Ontology role and publication boundary

The ICRA paper uses a lightweight PPR ontology as the architecturally required
formal backbone joining the product specification to primitive composition. It
is the authoritative semantic interface through which PA and RA share grounded
product facts, requested-process and outcome meaning, resource capability
assertions, provenance, and primitive bindings. Removing it would change the
proposed method rather than merely remove an optional implementation aid.
Ontology schema design nevertheless remains outside the standalone ICRA novelty;
the contribution is the integrated neuro-symbolic grounding,
composition-validation, and revision structure.

The nominal-case PA runtime is:

```text
persist exact product_requirement
        ↓
load fixed TBox and initialize the interaction ABox
        ↓
derive the current unresolved knowledge need
        ↓
request and serve one relevant approved source
        ↓
interpret it with the matching controlled tool
        ↓
validate and merge its evidence-backed triple delta
        ↓
reassess the updated ABox and repeat dynamically
        ↓
create and validate the robot-independent assembly plan
        ↓
initial versioned PA handoff to RA
```

One source per cycle is an auditable retrieval operation, not a limit of one
source for the interaction. PA chooses the number and order of relevant
document, CAD, and RGB-D requests from the evolving goal and ABox state together
with controlled-tool capability descriptors. It does not retrieve the entire
approved corpus by default.

For ICRA:

- The immutable TBox supplies the exact `product`, `feature`, `process`,
  `resource`, and `capability` types before the first PA source decision.
- One independent interaction ABox contains the exact requirement as an
  explicitly unresolved request. Apart from that request, it accumulates only
  evidence-backed factual assertions, each paired with persisted provenance.
- The TBox contains generic classes, properties, and restrictions. PA-created
  requested-process, product, feature, and grounded-outcome individuals belong
  to the interaction ABox. The selected resource and its exact eight concrete
  primitive-process interface offerings belong to a separate, versioned
  resource-catalog ABox snapshot supplied by that RA. The resource links to
  those offerings through `capableOf`; PA does not author resource facts.
- Document and scene tools are controlled, side-effect-free producers. Both
  return one generic subject-predicate-object triple-delta contract; the
  orchestrator validates and merges accepted assertions.
- Tool `can_produce` and `may_require` descriptors help PA route an unresolved
  semantic need to relevant approved evidence. Neither the TBox nor PA uses a
  product-specific slot template, fixed source count, fixed modality set, or
  fixed retrieval order.
- PA-to-RA task records and resource-owned primitive contracts reference the
  same ontology types for auditable primitive-input binding.
- The requested assembly process may be linked through `realizes` to its
  grounded product outcome. It must not be linked to primitive offerings
  through `requires`, `precedes`, direct high-level `capableOf`, or another
  task-to-primitive mapping. The missing mapping is intentional: the RA LLM
  reasons from the requested process and outcome to a candidate sequence over
  the available lower-level interfaces.
- A Spec2Primitives-owned read-only projection adapter dynamically joins only
  the relevant PA ABox assertions, all eight selected-resource offerings,
  authorized typed context refs, and their source fingerprints for the current
  RA turn. The RA LLM consumes this bounded semantic projection plus the
  executable interface cards and fresh state. The shared RobotAgent does not
  depend on RDFLib, mutate either ABox, or receive an ontology-inferred
  decomposition.

The minimal relationship is illustrated below using three of the previously
supplied primitive symbols; the runtime catalog projection always includes all
exact eight:

```turtle
# PA interaction ABox
ctx:gear_assembly_1 a ppr:process ;
    ppr:realizes ctx:medium_gear_installed .

ctx:medium_gear_1 a ppr:product .
ctx:medium_gear_installed a ppr:feature .

# Selected-RA resource-catalog ABox
ctx:ur5e a ppr:resource ;
    ppr:capableOf ctx:primitive_move_arm_xyz,
                  ctx:primitive_grasp_part,
                  ctx:primitive_release_part .

ctx:primitive_move_arm_xyz a ppr:process .
ctx:primitive_grasp_part a ppr:process .
ctx:primitive_release_part a ppr:process .
```

The graph intentionally contains no assertion such as
`ctx:gear_assembly_1 ppr:requires ctx:primitive_grasp_part`, no primitive
`ppr:precedes` relation, and no claim that a primitive by itself
`ppr:realizes ctx:medium_gear_installed`. Consequently, TBox plus ABoxes do not
entail a sequence. The RA LLM proposes the missing connection as a candidate
runtime record, and deterministic validators return acceptance, rejection, or
`unmodeled` without converting that hypothesis into evidence-backed PA facts.

The remaining ontology boundaries are:

- Poses, RGB-D data, calibration, tolerances, and other numeric payloads remain
  in the existing typed context records when the loaded TBox cannot express
  them.
- CAD correspondence, observation association, and context-record links also
  remain in typed records when the fixed PPR vocabulary cannot express them.
  The runtime never invents an ontology predicate.
- Initial PA handoff occurs only after the dynamic grounding loop finds no
  currently identified blocking product-level ambiguity. If a later primitive
  contract exposes a missing product or scene binding, RA returns structured
  `missing_context`, PA resumes the same ABox loop, and the robot-independent
  assembly plan is revalidated or revised before PA sends a versioned update and
  RA recomposes. Robot-side gaps remain RA-owned.
- Composition readiness means RA can bind every required primitive input and
  pass every applicable candidate-schema, backward modeled-condition, forward
  resource-state/data-dependency, physical, and projected-outcome check over
  the represented contract surface. PA handoff readiness is therefore a
  product-level handoff decision, not a guarantee of composition or execution
  readiness.
- The ontology does not discover, rank, or allocate resources. Nominal resource
  assignment consumes the registered RA roster and exact `resource_jid` output
  from the mechanism established in prior work; recovery input already carries
  its assigned `resource_jid`. Both paths unicast to that RA.
- In the recovery transfer case, grounded ontology identifiers may describe the
  desired product outcome, but the fault, custody, live resource state, selected
  recovery event, and feasibility records remain typed runtime context. The
  work does not introduce ontology-based fault diagnosis, recovery planning,
  event generation, or allocation.

The ICRA contribution remains RA-owned agentic selection, ordering, and
parameterization of `primitive_steps` within a non-synthesizing
validation-feedback revision loop. ICRA uses agent-requested,
one-source-at-a-time exact-ref retrieval over ontology-backed interaction state;
RA additionally receives a bounded deterministic projection over the current
task and selected-resource ABoxes. ICRA does not claim ontology design, SHACL
completeness, learned or graph-ranked retrieval, GraphRAG, ontology-based
composition, or execution graph updates as contributions.

The separate ontology journal work may extend this foundation with a PPR
execution graph, graph-directed context retrieval, formal provenance and
context-completeness reasoning, validation and execution feedback, and dedicated
ontology-retrieval experiments. That journal must contribute a new method and
new results beyond the ICRA system, not only a more detailed description of the
same ontology. Its working scope remains in
`writing/Journal Paper 3 (Ontology RAG)/README.md`.

## Relationship to Manual2Skill

[Manual2Skill](https://arxiv.org/abs/2502.10090) takes an assembly manual and a
complete set of 3D parts, constructs a hierarchical assembly graph, predicts
per-step component poses, and uses heuristic grasping and collision-free motion
planning for execution. Its reported real-world pipeline leaves closed-loop
insertion to a human expert.

[Manual2Skill++](https://arxiv.org/abs/2510.16344) adds connector-aware assembly
graphs, assumes precise 3D models and known attachment-point locations, computes
part poses from connection constraints, and evaluates predefined connection
strategies in simulation.

The correct comparison is:

- Manual2Skill asks which parts should be assembled, in what structure, and at
  what poses.
- Manual2Skill++ additionally asks which connector relationships and geometric
  constraints define the assembly.
- Spec2Primitives asks which currently available robot primitives, ordering,
  bindings, and parameters form a feasible program now, and how that program
  should change when robot-local validation rejects it.

Manual2Skill does compose assembly steps and generates motion trajectories.
Manual2Skill++ decomposes its graph into connection operations and evaluates
predefined execution strategies. The paper must therefore not claim that these
systems perform no composition. The narrower distinction is:

> Manual2Skill composes assembly structure and uses a fixed execution pipeline;
> Spec2Primitives dynamically composes and revises a robot-specific primitive
> program from the currently available resource-owned primitives.

## Research gap

The central gap is not manual-to-action generation. Manual2Skill already
addresses that problem. The remaining gap is runtime construction and revision
of the robot execution program under current embodiment and resource
constraints.

It is also not the requirement-to-capability matching problem addressed in the
prior RCIM work. That workflow can adaptively select and coordinate predefined
composite functions, but it assumes that a suitable function implementation is
already present in the resource capability library. The remaining problem is
what the selected RA should do when no valid selection, ordering, and
parameterization of that function library satisfies the grounded transition and
only its execution primitives cover the required operations.

A paper-ready gap statement is:

> Manual2Skill and Manual2Skill++ demonstrate that instruction manuals and
> known 3D part models can be transformed into assembly hierarchies, connection
> constraints, and target poses. However, their execution stages rely on a fixed
> motion-planning pipeline or predefined connector-specific strategies. They do
> not investigate runtime synthesis of a robot-specific primitive program from
> a resource-owned primitive catalog under fresh robot state and embodiment
> constraints. They also do not evaluate a closed validation-revision loop in
> which concrete state, IK, collision, and trajectory failures cause structural
> changes to the generated `primitive_steps`. This leaves open how grounded
> product intent and validated recovery transitions can be transformed by one
> resource-owned composition mechanism into feasible and revisable primitive
> programs under current embodiment constraints, without providing a
> task-specific primitive recipe.

The corresponding research question is:

> Given a grounded task transition, fresh selected-resource context, and exactly
> eight semantic primitive interfaces with partial local executable contracts,
> where no valid selection, ordering, and parameterization of the selected RA's
> predefined composite functions satisfies that transition, can one RA-owned
> LLM composer author and revise validator-accepted `primitive_steps` for
> nominal and recovery-origin tasks without a task-specific recipe or complete
> PDDL domain/problem?

The concise contribution statement is:

> Spec2Primitives combines an architecturally required ontology-backed symbolic
> context with RA-owned neural candidate synthesis and a non-synthesizing
> symbolic and physical validation-feedback loop to produce a previously
> unavailable task-specific composite program from resource-owned execution
> primitives.

The architectural hypothesis is that the ontology-backed context provides the
stable formal interface through which dynamically retrieved product knowledge,
selected-resource capabilities, and primitive bindings enter composition. The
primary evaluated hypothesis is coverage of the admitted transitions outside
the predefined composite-function interface. The feasibility hypothesis is that
validation-guided neural revision improves accepted and executed primitive
programs over equal-input primitive composers. The recovery hypothesis is
limited to transfer of the same composition mechanism.

The ontology deliberately entails neither a one-to-one resource match for the
grounded high-level process nor an ordered primitive realization. It supplies
shared product, process, resource, outcome, and binding semantics; the semantic
primitive interface cards supply callable behavior descriptions. The evaluated
hypothesis is that the RA LLM can bridge this deliberately unmodeled
task-to-primitive relation by proposing and revising candidates that the
non-synthesizing validators can accept or reject. This motivates the agentic
composer in the framework without claiming that no alternative composer could
bridge the same relation.

The primary evidence comes from nominal end-to-end assembly without a
predefined nominal program and outside the executable coverage of the predefined
composite-function library. The secondary question is whether the exact same RA
composer transfers to an inherited, resource-assigned recovery event with the
same coverage gap, without recovery-specific composition logic.

## Claim boundaries

The paper should not claim that Manual2Skill lacks:

- scene or part grounding;
- assembly-step composition;
- target-pose prediction;
- action generation;
- collision-free motion planning; or
- connector-aware or contact-execution research in Manual2Skill++.

The paper should also keep the following evidence levels separate:

- product grounding;
- agentic candidate generation;
- `primitive_steps` schema validity;
- backward and forward modeled-condition validity;
- robot-local feasibility validation;
- projected task-outcome validity;
- validation-driven revision convergence;
- simulation execution;
- observation-backed realized outcome; and
- physical robot execution.

The paper must not claim that:

- LLMs are universally required for primitive composition;
- an ontology is universally required for every primitive-composition method;
- the eight primitive contracts form a complete classical planning domain;
- backward or forward contract validation proves physical feasibility or the
  realized task outcome;
- an LLM-authored final-state assertion is execution evidence;
- a validator composes, completes, or repairs `primitive_steps`;
- the lightweight ontology proves composition readiness or performs primitive
  composition;
- resource discovery or optimal allocation is a new contribution;
- requirement interpretation, predefined-function planning, capability
  matching, resource assignment, or PA–RA parameter discovery from the prior
  RCIM work is a new contribution;
- Spec2Primitives replaces or is universally better than the prior RCIM
  framework;
- moving a predefined composite function's internal sequence into an LLM prompt
  constitutes dynamic primitive composition; or
- the inherited recovery event-generation, selection, DES/CCA, safety, or
  approval framework is an ICRA contribution.

The task-transition contract is a grounded goal and binding record rather than
a PDDL problem or primitive decomposition. Acceptance over the partial interface
surface proves neither global symbolic plan soundness or completeness nor the
realized assembly outcome. Required conditions outside all declared contracts,
invariants, evaluator endpoints, and projected-outcome checks must be reported
as `unmodeled` and fail closed.

PA and RA being separate agents is an architectural mechanism, not sufficient
novelty by itself. The contribution depends on showing that the separation,
provenance-backed task-transition contract, fresh resource grounding, RA-owned
agentic candidate generation, and non-synthesizing validation-driven revision
improve primitive-program feasibility. The equal-input comparison includes a
symbolic composer over the represented interface fields, bounded black-box
search with the same validator access and budget, LLM-only generation, and the
full agentic generation plus validation-revision method under the same task
contract, catalog, state, and execution boundary. Report domain coverage and
abstention separately; treat a separately engineered complete PDDL domain as an
extra-information oracle. Monolithic, no-retrieval, stale-context, and
no-revision ablations may further isolate the nominal pipeline. Recovery results
support composer transfer only and must not be used to re-claim the inherited
recovery stack.

A prior RCIM-style predefined-function planner should be reported separately as
a boundary baseline. Its inability to match an admitted case explains why the
primitive-composition path is entered; that coverage-gap or abstention result is
not evidence of composition quality or LLM superiority. Give this baseline the
prior composite catalog and let it select, order, and parameterize its functions,
but not inspect, decompose, or recombine their internal primitive
implementations. Report its coverage and abstention separately. Composition
quality must instead be established against equal-input primitive composers
using first-pass validity, validator-rejection categories, revision convergence,
execution success, and observation-backed outcome evidence.
