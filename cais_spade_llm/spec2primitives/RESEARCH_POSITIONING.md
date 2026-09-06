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

## Proposed-versus-implemented status

The end-to-end composition/execution architecture in this document remains proposed. Current product grounding accepts one complete target and a collection of pairwise assembly relationships. Current observations and desired relationship membership are separate. PA owns semantic interpretation and corrections. Deterministic contract feedback permits up to six proposals sharing 24 evidence operations and one pinned observation per grounding invocation; exact repeated failures stop the loop earlier.

PA checks every configured capable arm using the same bound current/destination coordinate references and live MoveIt position planning, then selects an arm with an accepted check. Missing arm checks retain their separate single correction; unavailable results remain distinct from rejected planning. Selection and completion record that ontology assignment with position-planning evidence and explicit unvalidated grasp/insertion constraints. Multiple relationships do not force a unique Cartesian pair. Required locations are checked before target commit; unresolved inputs or assignment prevent completion. Contract acceptance does not establish semantic correctness.

Phase 5.1 requires current completion before exact selected-RA envelope and paired context snapshots. Phase 5.2A creates one unbound structural draft. Live SPADE delivery, binding/context exchange, primitive-level validation, execution and observed outcomes remain future work. Older supported records remain immutable history and cannot authorize new RA work.

See [implementation status](IMPLEMENTATION_PLAN.md), [ontology semantics](ASSEMBLY_ONTOLOGY.md) and [bias experiments](BIAS_VALIDATION.md). Generic deterministic checks and offline tests do not establish model accuracy or absence of bias.

## Abstraction level

The research spans the following abstraction chain:

```text
product specification
        ↓
evidence-grounded target feature / desired product state
        ↓
ontology-backed robot-independent grounded product outcome/task
        + selected-resource assignment
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

The implemented nominal contract supports one complete requirement transition in one proposal. PA authors process, current/desired statements, typed values and a collection of pairwise assembly relationships. Relationship state membership is explicit; two current observations may anchor a desired relationship. Owners are reused only by exact identity, with variable assertion count under unchanged TBox cardinality.

Deterministic checks cover structure, evidence integrity and required planning inputs before commit. PA may revise within the shared evidence/proposal budget, reusing pinned observations. PA then chooses an arm backed by configured capability and live MoveIt planning for all bound coordinate references. Non-coordinate values remain semantic evidence. Completion and the current evidence contract are required for all new RA work; they do not independently prove PA's interpretation.

Assembly is the controlled example, not the contract's only possible process.
The production profile currently contains only `assembly`, `xarm6`, and `ur5e`,
so the deployment is an assembly case study and makes no runtime
process-discovery claim. Future welding, painting, milling, and other configured
cases can use the same target shape when their evidence, typed providers,
primitive contracts, and validators exist; those cases are not currently
implemented claims.

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

The Phase 4.2B1--4.2B2 supporting perception path is also supporting
infrastructure. Automatic four-camera RGB-D capture, calibration validation,
deprojection, deterministic support-plane handling, connected-region
segmentation, compact mask records, strict one-CAD principal-size comparison,
standalone generalized CAD-to-RGB-D pose estimation, standard camera-to-robot
frame conversion, atomic records, and operator status reporting are necessary
plumbing for later experiments, not the claimed research contribution. The pose
and frame-conversion callables are connected to the production PA grounding
orchestration through typed prerequisites and an injectable approved-calibration
boundary; no default approved calibration is configured in the current UI
composition. They are not connected to RA composition or execution. A size-only
correspondence still does not establish general identity, task completion,
planning, or execution.

The perception records expose neutral candidates without assigning a semantic
source or target role. The architecture has no `TargetFeatureGeometryRecord`.
PA dynamically chooses approved evidence and may link typed values to
`current_state` and `desired_state`; calibration and numeric robot-frame
locations are derived only when a downstream verifier receives compatible
geometric bindings.
No fixed evidence or resource order is prescribed.

This PA claim is bounded. The controller prompt does not prescribe an evidence
modality or record type. Approved observation retrieval automatically runs the
deterministic preprocessing and segmentation implementation when PA requests
that evidence. PA does not choose that algorithm or its parameters. It chooses
approved evidence, current and desired statements, optional typed state values,
and—when geometric bindings permit allocation—the provisional resource;
deterministic reachability and RobotAgent checks can only accept or reject those
unchanged downstream choices. The paper
therefore describes evidence-conditioned bounded PA autonomy within the fixed
`assembly` process, not unrestricted freewill or dynamic perception-chain
selection. Descriptor-derived record planning remains deferred and is required
only for the latter claim.

Spec2Primitives studies a composite-function coverage gap. It begins after an RA
has been selected and no valid selection, ordering, and parameterization of its
available predefined composite functions can satisfy the required grounded
transition. The selected RA exposes its complete current authoritative
primitive-only catalog at the composition boundary, and the RA must author the
missing task-specific composite program at runtime:

```text
prior RCIM abstraction:
requirement → select and order predefined composite functions → assigned execution

Spec2Primitives abstraction:
grounded transition outside composite-function coverage
        + selected RA + complete current primitive-only catalog
        → newly authored primitive_steps → validation → revision → execution
```

Spec2Primitives is therefore positioned as a narrower, deeper continuation of
the RCIM work, not as a replacement for or globally better version of it. Its
novelty must be evaluated at the capability-to-executable-primitive-program
boundary. Merely moving function selection from PA to RA, or renaming an
existing function as a primitive, would not constitute the claimed advance.

## Relationship to A Closed-Loop Multi-Agent Framework for Robust Multi-Robot Manipulation

[A Closed-Loop Multi-Agent Framework for Robust Multi-Robot
Manipulation](https://arxiv.org/html/2607.06990v1) is direct prior art for
LLM/VLM-driven runtime composition from predefined action primitives. Its
Planning Agent produces a dependency-aware task DAG, parallelism decisions, and
robot allocation; it does not generate the low-level action sequence. Its
Manipulation Agent decomposes each assigned subtask and generates executable
Python calls by dynamically selecting, ordering, repeating, and parameterizing
a fixed library of 11 primitives: `Grasp`, `Place`, `Lift Up`, `Put Down`,
`Reset Home`, `Move XY`, `Move Pose`, `Rotate`, `Align`, `Open`, and `Close`.

That framework grounds arguments through adaptive perception and manipulation
tools, including object/part detection and segmentation, 3D keypoints,
AnyGrasp-based grasp candidates with kinematic filtering, and rotation
estimation. Short-term history helps avoid redundant operations, while an
experience pool can reuse a successful primitive sequence for an exact repeated
task signature. Post-operation visual verification triggers local corrective
steps; workspace or object-availability failures return to the Planning Agent
for global replanning or reallocation.

Therefore, neither runtime selection from a fixed primitive library nor an
LLM-generated action sequence is by itself a Spec2Primitives novelty claim. The
narrower proposed distinction must be evaluated as a combination of
ontology-driven product-to-resource grounding, explicit resource authority, a
complete selected-RA-owned and hash-pinned typed catalog, structured
`primitive_steps` with provenance, and deterministic non-synthesizing validation
that returns findings to the RA author. Both systems rely on predefined
primitive implementations; Spec2Primitives must demonstrate the claimed
contract, authority, provenance, and validation differences empirically rather
than treating different primitive names or granularity as novelty. The cited
paper also identifies its fixed primitive library as a limitation, so
Spec2Primitives must not claim primitive discovery unless such a mechanism is
separately implemented and evaluated.

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
  `resource`, `capability`, and `processExecution` vocabulary;
- the evidence-backed interaction ABox records the grounded requested process,
  product facts, features, and required outcome for the current interaction;
- the predefined Workcell ABox records `xarm6` and `ur5e` as broadly
  `capableOf assembly`, while a system-authored execution individual records the
  one resource selected using typed location evidence and manifest reach;
- the RA-authoritative primitive catalog snapshot enumerates the
  complete callable composition surface for that resource and interaction in a
  typed record without primitive-level `capableOf` assertions or a
  task-specific decomposition;
- typed context records retain locations, poses, observations, calibration, tolerances,
  resource state, and other values when the loaded TBox has no predicate for
  them; and
- partial primitive contracts and deterministic validators check the candidate
  over the explicitly represented symbolic, data-dependency, resource, and
  outcome surface.

Within the proposed method, the ontology is the canonical semantic interface for
entity identity, the requested process, broad Workcell capabilities, and the
selected process execution. Typed context and status records represent evidence
links, primitive-input bindings, unresolved needs, conflicts, numeric values,
and freshness. `GroundingProducerDescriptor` records describe available
authorized evidence operations without prescribing which operation or source
PA should use. The TBox defines valid shared symbols; it does not select a
producer, source, candidate, state interpretation, or resource.

Current product-grounding acceptance requires valid proposal structure and ontology projection, authorized/resolvable hash-pinned citations and bindings, required planning inputs, a configured process, and PA's unchanged capable arm choice backed by reachability for every bound current/destination coordinate reference.

PA interprets part type and task role, current installation and document intention, and observed destination references and final insertion poses. State values may contain generic typed evidence. The planning consumer checks its required location inputs without imposing semantic completeness rules on every ontology proposal. Missing inputs or unissued references return deterministic feedback, with up to six proposals sharing 24 evidence operations. A structurally valid but semantically incorrect proposal can pass these checks; accuracy and bias require separate empirical evaluation.

MoveIt position planning checks joint limits and the live collision scene; it does not certify manufacturing. Later primitive contracts may require pose, grasp, clearance, tolerance or other inputs without rewriting the historical target. See [BIAS_VALIDATION.md](BIAS_VALIDATION.md) for claim-level audits and controlled experiments.

SPARQL is an optional graph-access mechanism, not a requirement for every
ontology consumer. The host may use fixed, validated graph extraction to
materialize the bounded ontology projection; the selected RA consumes that
projection rather than receiving unrestricted access to the complete graph.
The ontology remains operationally consequential only when its accepted facts
constrain resource assignment, context binding, composition, or outcome
validation. If those facts are merely persisted and cannot affect a downstream
decision, the ontology would be decorative rather than part of the method.

The composite-function coverage gap and composition surface are established
from complete RA-authoritative typed catalog snapshots whose
cardinality is determined at runtime. Each attempt pins the snapshot fingerprint
and exact symbols. Completeness is not inferred from OWL open-world absence or
from the broad `capableOf assembly` assertions. A bounded SHACL check may test
only an explicitly declared shape or already identified required input at a
local validation boundary. It does not establish global context completeness.

The neural side makes the decisions that are intentionally absent from the
formal model:

- PA autonomously chooses approved evidence and tools, authors the target
  feature, then independently chooses state locations and a capable resource;
- the selected RA LLM first authors a structural `PrimitiveProgramDraft` and
  later a fully bound, variable-length `primitive_steps` candidate by choosing,
  ordering, repeating, and binding only exact symbols in the pinned catalog;
  and
- after deterministic rejection, the RA LLM interprets the categorized findings
  and authors a structurally revised candidate.

The resulting closed loop is:

```text
TBox + interaction ABox + predefined Workcell ABox + system assignment
        + selected-RA typed catalog + typed context refs + partial contracts
        ↓ bounded grounded projection
        ├─ PA neural grounding decision → controlled evidence producers
        │       → final target_feature → integrity/ontology validation
        │       → PA neural allocation decision → reachability validation
        │
        └─ RA neural decision → structural PrimitiveProgramDraft
                → binding preflight
                ├─ RA-owned inputs → local resolution
                └─ product/scene inputs → batched PA producer round
                → fully bound candidate primitive_steps
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
the complete selected-RA-authoritative primitive-only catalog, including its
length, selection, order, repetition, intermediate steps, and grounded
parameter bindings
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
call in a resource API. They are not admitted merely by appearing beside the
selected RA's primitive-only catalog. An accepted `primitive_steps` program may
realize equivalent behavior, but that task-specific composition did not exist
before the RA authored it.

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
condition or effect is unknown, not satisfied. The complete current catalog does
not necessarily provide an exhaustive symbolic state vocabulary, global
transition model, task-specific goal decomposition, primitive order, or
complete PDDL domain/problem. The combined result of several primitives may
establish an assembly outcome that no individual primitive can establish alone.

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
> primitive-only catalog, conditioned on an ontology-grounded task and
> exact selected-resource assignment outside predefined composite-function
> coverage, plus fresh resource context. The
> resulting task-specific composite program did not exist as a callable function
> before the interaction.

The RA LLM authors each structural `PrimitiveProgramDraft` and fully bound
candidate `primitive_steps` program. Primitive contracts, binders, and
deterministic backward, forward, physical, and projected-outcome checks may
identify missing inputs or accept or reject an unchanged candidate and return
findings. They never select, insert, remove, order, parameterize, or repair
primitive steps; RA authors the next complete candidate from the findings.

The word `dynamic` requires more than replacing parameters in a fixed template.
The evaluation must demonstrate that:

1. No valid selection, ordering, and parameterization of the available
   predefined composite functions satisfies the ontology-grounded task, and
   no product-specific assembly program or completed `primitive_steps` is
   available or supplied.
2. RA receives the assigned nominal task or selected recovery event, its
   required outcome and grounded refs, the complete current
   selected-RA-authoritative primitive-only catalog of semantic interfaces with
   partial local executable contracts, and fresh resource state, but no
   completed `primitive_steps`, task-specific PDDL domain/problem, or
   product-specific executable program.
3. RA agentically authors primitive selection, ordering, bindings, and
   parameters at runtime.
4. Changes in robot state or the resource-specific catalog availability and
   feasibility can change the resulting primitive program.
5. Concrete contract, state, IK, collision, trajectory, or outcome findings can
   cause RA to author a structurally revised candidate rather than only perform
   a blind retry or parameter perturbation.
6. Different assembly requirements require different composition structures,
   rather than variants of one stored sequence.
7. The same RA composition entrypoint, prompt policy, dynamic-catalog interface
   and versioning policy, context protocol, candidate schema,
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
received only the complete pinned primitive-only catalog. Each case must require
at least one structural primitive decision such as selection, ordering,
repetition, or an intermediate step; merely changing parameters on an existing
composite function is insufficient.

Before an experiment, audit the selected RA's complete public callable interface
and establish that no valid selection, ordering, or parameterization of its
available predefined composite functions can satisfy the grounded
task and required outcome, while the necessary primitive interfaces remain
available. Perform this coverage audit from the grounded task and public
function semantics without reading an expected primitive program. Merely hiding
an applicable composite function from the prompt is not an admissible case.

Each composition attempt freezes the selected RA's complete current catalog
snapshot, exact symbols, fingerprint, and reported cardinality. Equal-input
comparisons use that same per-case snapshot. Cross-resource experiments may
vary catalog size and symbols, and must report those differences. Each selected
RA must still expose truthful resource-owned parameter limits, state
requirements, and evaluators; those values are not normalized to make cases
appear identical.

## Agentic composition and validation boundary

The operational method is:

```text
reconstructed evidence-grounded target_feature
        + post-assignment feature/resource ontology projection
        + fresh selected-RA state
        + complete pinned semantic primitive interfaces and partial contracts
        + grounded context refs
        ↓
RA LLM authors structural PrimitiveProgramDraft
        ↓
binding preflight resolves RA-owned inputs and batches product/scene gaps
        ↓
RA LLM authors fully bound candidate primitive_steps
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
assembled. A future separate task-outcome check must derive its required numeric
geometry from PA-selected approved evidence and process constraints at validation
time; it cannot consume a predetermined target-geometry answer. Observation
evidence then confirms the realized result. That outcome-check contract is not
implemented now. A projected-outcome check returns `unmodeled` and stops
fail-closed when the requested constraint lies outside its represented coverage.
If an intermediate requirement appears in no contract, invariant, primitive
evaluator, physical check, or outcome check, the system cannot claim to have
detected or validated its omission.

## Why an RA LLM composer is justified relative to symbolic planning

Spec2Primitives does not claim that LLMs are universally necessary for
primitive composition or that classical planning cannot solve these tasks.
When a complete symbolic domain captures the goal, initial state, primitive
preconditions and effects, and relevant state transitions, a classical planner
is the stronger default because it provides systematic search over that model.
The RA LLM is justified at a different boundary: the selected resource exposes
a complete current catalog of semantic executable interfaces with typed inputs,
outputs, truthful limits, partial local conditions and effects, and evaluator
endpoints, but it does not expose a complete causal action model connecting the
new product outcome to every required intermediate state.

### Concrete minimal-contract assembly boundary

The current assembly example deliberately keeps each primitive's local contract
minimal. `grasp_part` requires `held_part = null` and establishes `held_part`
from the part parameter. `release_part` requires `held_part != null` and clears
`held_part`. `move_cartesian` has no modeled condition and updates
`current_pose` from `x`, `y`, and `z`. `compute_pick_targets` and
`compute_place_targets` expose typed `approach_pose` and `target_pose` results
but no symbolic conditions or effects.

Over exactly that represented surface, backward symbolic derivation can connect
`grasp_part` to `release_part` through `held_part`. It cannot derive that either
target-computation primitive or `move_cartesian` must occur, because no symbolic
predicate connects their typed pose results or `current_pose` update to the
custody contracts. The RA LLM is evaluated as the candidate composer that fills
this unmodeled structural gap from the task and semantic primitive interfaces;
deterministic checks still reject violations of the conditions and effects that
are represented.

This is not a claim that symbolic planning is bad or generally incapable of
assembly composition. A separately engineered PDDL domain that adds the missing
causal predicates and action relationships could derive the motion and target
steps. The narrower claim is that an interface-only symbolic planner cannot
derive them from these intentionally partial, truthful runtime contracts alone.

Continuous feasibility alone is not this justification. IK, collision,
visibility, motion, and other high-dimensional constraints can be integrated
with symbolic task planning through black-box procedures, as demonstrated by
[PDDLStream](https://arxiv.org/abs/1802.08705). The missing artifact studied
here is the task-level causal decomposition itself. In the evaluated
partial-model setting, no complete product-specific executable program or
complete finite symbolic search domain is supplied. The RA LLM therefore acts
as the agentic candidate composer over natural-language operation semantics and
typed bindings, while deterministic symbolic and physical validators retain
authority to reject the unchanged candidate. They do not become a hidden
composer. This generator-verifier division is consistent with the broader
[LLM-Modulo](https://arxiv.org/abs/2402.01817) position, while
[PlanBench](https://arxiv.org/abs/2206.10498) provides further reason not to
treat an unaided LLM as a sound replacement for model-based planning.

The empirical comparison must distinguish equal-input methods from a
separately engineered complete symbolic model:

1. An interface-only symbolic composer uses only the machine-readable fields on
   the same partial primitive interfaces and abstains outside represented
   coverage.
2. Bounded black-box sequence search receives the same validator access and
   candidate-query budget as RA but no semantic LLM composer.
3. LLM-only generation receives the same task, catalog, and state but no
   validation-driven revision.
4. The full RA method authors complete candidates and structurally revises them
   from categorized, non-synthesizing validator findings.
5. A classical planner receives a separately engineered complete PDDL domain as
   an extra-information oracle rather than an equal-input baseline.

The study must report domain coverage, abstention, validator-accepted and
executed outcomes, and the engineering effort required to construct and
maintain the complete symbolic model, including its predicates, action schemas,
external procedures, and task-to-goal mappings. The oracle's success measures
the value of that additional modeling, while its authoring and update effort
measures the cost. Symbolic failure outside represented coverage cannot support
a universal LLM-necessity claim. The defensible claim is conditional: RA-owned
agentic composition can extend execution-program coverage when the runtime
surface is semantically informative and deterministically checkable but
causally incomplete.

## One composition core, two task origins

```text
nominal: grounded PA task + resource assignment ─┐
                                                  ├─→ same RA-owned agentic composition core
recovery: validated resource-assigned event ─────┘
                                                           ↓
                                      candidate → validators → RA revision
```

The nominal case is the primary end-to-end study: PA begins with a raw assembly
requirement whose grounded task is outside the selected RA's predefined
composite-function library coverage and has no stored nominal task program,
grounds the requirement, and supplies the robot-independent post-assignment
ontology projection and evidence records. The recovery case is a transfer study:
the existing recovery framework
begins with the fault and runtime state and supplies an already selected,
validated, and resource-assigned recovery event with the same
composite-function coverage gap. A read-only adapter maps that event into the
same composition input without providing its primitive decomposition.

After this boundary, each origin uses its exact selected RA through the same
composer implementation, runtime-catalog interface, batched context-request
surface, candidate schema, validation sequence, feedback loop, and execution
boundary. Nominal perception and planning and the existing recovery event
generation, selection, allocation, approval, and safety
validation remain distinct upstream mechanisms and are reported separately.
The recovery study tests transfer of the composer; it is not a second
recovery-planning contribution.

## PA and RA dynamic context retrieval

The proposed cross-authority workflow is:

```text
Exact requirement + authoritative ontology + available sources
        ↓
PA understands the requirement and ontology together
        ↓
PA retrieves only currently relevant approved evidence
        ↓
PA authors one cited target_feature with current and desired states
        ↓
host validates the final shape, process authority, citations, hashes,
and any included typed-evidence links and required planning inputs
        ↓
PA corrects repairable contract failures within the shared budget; host commits accepted associations
        ↓
host presents all capable resources and neutral location handles
        ↓
derive all bound coordinate references; PA checks and chooses one capable resource
        ↓
check_reachability reports every PA-submitted location independently
        ↓
host validates the unchanged cited selection once
        ↓
system commits four assignment assertions and completion
        ↓
selected RA receives reconstructed target_feature + fresh state
        + complete current primitive catalog
        ↓
RA authors structural PrimitiveProgramDraft
        ↓
binding preflight gathers missing inputs
        ├── RA-owned → resolve locally
        └── product/scene-owned → MissingContextBatch to PA
                                      ↓
                         PA runs required controlled producers
                                      ↓
                         CompositionContextBundle
        ↓
RA authors fully bound primitive_steps candidate
        ↓
non-mutating validation and RA revision
        ↺ another batched context round only when new needs and progress exist
        ↓
accepted candidate or fail-closed result
```

- PA may make zero or more native tool calls and then returns one complete
  two-state target feature, a genuine requirement-meaning clarification, or
  `unsupported_process`.
  The system records tool-call IDs, resolved evidence, source revisions, hashes,
  and failures. `ProductContextView` exposes only the final accepted ABox and
  `TypedContextBinding` records; it is not PA's reasoning state.
- The official TBox defines valid meaning and types; it does not choose a
  document, sensor, CAD, tool, candidate, or resource. PA may dynamically select
  any approved source or controlled tool.
- Structural/provenance validation and required planning-input checks precede
  commit, with budgeted PA corrections from generic feedback. Generic state values
  remain supported; every bound coordinate-bearing value supplies the arm check.
- A second PA decision checks/selects a capable resource. The verifier covers
  all bound locations and never ranks or substitutes alternatives.
- PA does not author an insufficiency, failure reason, or unmet path. The
  controller exposes deterministic stage codes and concrete contract feedback for
  invalid structure, provenance, unavailable locations, capability, or reachability.
  Unissued references and missing planning inputs may return for PA correction
  within the shared budget. Feedback contains no suggested semantic answer.
- RA retrieves the complete selected-RA-authoritative primitive-only catalog
  and fresh local state after allocation. Catalog cardinality is runtime-
  determined; the attempt pins its fingerprint and exact symbols.
- Primitive contracts and binding preflight reveal the runtime inputs for the
  `PrimitiveProgramDraft`. RA resolves robot state, limits, IK, collision,
  grasp, release, trajectory, and other robot-owned needs locally. It sends all
  currently known product or scene needs to PA in one deduplicated
  `MissingContextBatch` and receives one fingerprinted
  `CompositionContextBundle`.
- There is no fixed semantic batch count. Another PA-to-RA round occurs only if
  a new need appears and the prior round added a valid binding or otherwise made
  measurable progress. Ambiguity, repeated requests, unavailable producers,
  stale or replayed bundles, catalog changes, and no progress stop fail-closed.
  Operational deadlines and cancellation remain permitted.
- Binders and validators may detect and categorize missing inputs, but may never
  create, reorder, bind, or repair primitive steps. Only RA authors structural
  drafts, fully bound candidates, and revisions.
- Product grounding and arm assignment completion means the target/association assertions and four
  assignment assertions are accepted and pinned by completion. There is no
  separate task-transition or new-run typed-grounding contract. RA
  readiness means that one unchanged candidate is fully bound and accepted by
  every applicable declared-contract and resource-owned check. Neither agent
  certifies the next authority's work or the post-execution outcome.

Internal PA retrieval remains deliberately simple and auditable:

```text
PA sees requirement + ontology + available sources + retrieved records
        ↓
PA optionally calls prompt-local retrieve(evidence_id) handles
        ↓
system resolves each exact catalog ref and source revision
        ↓
approved sources are retrieved and returned in the same PA conversation
        ↓
PA returns one final proposal; system either accepts it unchanged or stops
```

Each tool call is an internal evidence audit boundary, not a fixed PA decision
stage or PA-to-RA message contract. A single future `MissingContextBatch` may
cause PA to run several producer operations before returning a
`CompositionContextBundle`. A
primitive never consumes or requests raw PDF, STL, RGB, depth, or calibration
payloads. OWL supports semantic matching, while a bounded SHACL or equivalent
check may detect that an already identified input is absent; neither mechanism
determines the primitive sequence.

## Ontology role and publication boundary

The ICRA paper uses a lightweight PPR ontology as the architecturally required
formal backbone joining the product specification to primitive composition. It
is the authoritative semantic interface through which PA and RA share grounded
product facts, requested-process meaning, broad Workcell capabilities, and the
system-selected process execution. Hash-pinned typed records carry provenance,
numeric state, detailed resource configuration, and primitive bindings that the
loaded vocabulary does not express. Removing the ontology would change the
proposed semantic join and assignment boundary rather than merely remove an
optional implementation aid.
Ontology schema design nevertheless remains outside the standalone ICRA novelty;
the contribution is the integrated neuro-symbolic grounding,
composition-validation, and revision structure.

The implemented Phase 4.4 path is:

```text
exact requirement → approved PA investigation → proposal
→ structural/provenance and required planning-input checks
→ budgeted PA corrections from generic contract feedback
→ accepted feature/states and pairwise association ABox
→ all bound coordinate references → PA capable-arm reachability/selection
→ four processExecution/resource assertions → completion
→ selected-RA envelope and paired context → unbound structural draft
```

Completion pins the proposal, its pre-commit context and complete evidence manifest, presentations, capability snapshots, reachability, assignment delta and final context without duplicating the target feature. Recovery revalidates the chain and live MoveIt reachability. Phase 5.2A reconstructs the target and bounded typed values from those authorities. `PrimitiveProgramDraft` remains unchanged and contains no bound program or execution claim.

Historical records remain untouched; incompatible contracts require “Start a fresh interaction” and cannot authorize new allocation, RA context or drafts. Current assignment requires live MoveIt position planning; grasp and insertion remain unvalidated.

One source per cycle is an auditable retrieval operation, not a limit of one
source for the interaction. PA chooses the number and order of relevant
document, CAD, and RGB-D requests from the evolving goal and ABox state together
with the approved source types. It does not retrieve the entire
approved corpus by default.

Approved catalog refs are intentionally visible evidence and may share lexical
tokens with the requirement. The native tool exposes only prompt-local handles,
but this implementation does not claim learned, graph-ranked, or
lexically unbiased source discovery. Retrieval evaluation must therefore report
catalog-label and distractor ablations separately from the constrained grounding
result.

For ICRA:

- PA performs schema-constrained, evidence-backed ABox instance grounding under
  the supplied PPR TBox. It does not discover or revise the ontology schema.
  The active proposal contains one overall target feature with multiple
  pairwise relationships when supported; unresolved required relations block completion.
- The immutable TBox supplies the exact `product`, `feature`, `process`,
  `resource`, `capability`, and `processExecution` vocabulary before the first
  PA source decision.
- One independent interaction ABox contains the exact requirement as an
  explicitly unresolved request. The host adds the generated root feature/states, accepted pairwise
  association individuals and the accepted PA allocation, each paired with persisted provenance.
- The TBox contains generic classes, properties, and restrictions. PA selects
  an authorized process and authors target semantics; the host creates the
  feature individual and exact projection. Workcell-profile supplies the
  authorized process catalog, resource catalog, and explicit capability facts.
  The production profile currently materializes `xarm6 capableOf assembly` and
  `ur5e capableOf assembly`. Only the system may add the selected
  `processExecution`; PA does not author resource or execution facts.
- Document and scene tools are controlled evidence producers. They return
  typed evidence records; only the late ontology mapper may propose assertions,
  and the orchestrator validates every proposed ABox change.
- `GroundingProducerDescriptor` records map a missing semantic or typed output
  to an authorized producer and evidence policy. Neither the TBox nor PA uses a
  product-specific slot template, fixed source count, fixed modality set, or
  fixed retrieval order.
- PA-to-RA target-feature records and resource-owned primitive contracts reference stable
  ontology identities for auditable primitive-input binding. The complete
  primitive catalog remains a typed, selected-RA-owned record and no
  primitive receives `ppr:capableOf`.
- The requested configured process may be linked through `realizes` to its
  grounded product outcome. It must not be linked to primitive offerings
  through `requires`, `precedes`, `capableOf`, or another task-to-primitive
  mapping. The missing mapping is intentional: the RA LLM
  reasons from the requested process and outcome to a candidate sequence over
  the available lower-level interfaces.
- A Spec2Primitives-owned read-only projection adapter dynamically joins only
  the relevant PA and Workcell ABox assertions, the system assignment, every
  offering in the selected-resource typed catalog snapshot, authorized typed
  context refs, and their source fingerprints for the current RA turn. The RA
  LLM consumes this bounded projection plus the executable interface cards and
  fresh state. The shared RobotAgent does not depend on RDFLib, mutate an ABox,
  or receive an ontology-inferred decomposition.

The minimal Phase 4.4 relationship is illustrated below. It records broad
Workcell eligibility and the evidence-backed selected execution, but contains
no primitive offering or sequence. Phase 5 supplies the selected RA's complete
typed catalog separately and preserves every symbol exactly:

```turtle
# PA interaction ABox
ctx:specification_1 ppr:defines ctx:feature_0001 .
ctx:feature_0001 a ppr:feature .
process:assembly ppr:realizes ctx:feature_0001 .

# Predefined Workcell ABox
process:assembly a ppr:process .
resource:xarm6 a ppr:resource ;
    ppr:capableOf process:assembly .
resource:ur5e a ppr:resource ;
    ppr:capableOf process:assembly .

# System-authored selection after typed evidence and reach validation
ctx:specification_1
    ppr:hasProcessExecution ctx:process_execution_0001 .
ctx:process_execution_0001 a ppr:processExecution ;
    ppr:runsProcess process:assembly ;
    ppr:runsOnResource resource:xarm6 .
```

The graph intentionally contains no primitive `ppr:capableOf`, `ppr:requires`,
or `ppr:precedes` assertion and no claim that a primitive by itself realizes the
feature. Consequently, TBox plus ABoxes do not entail a sequence. The RA LLM
proposes the missing connection as a candidate runtime record, and deterministic
validators return acceptance, rejection, or `unmodeled` without converting that
hypothesis into evidence-backed PA facts.

The remaining ontology boundaries are:

- Locations, poses, RGB-D data, calibration, tolerances, and other numeric payloads remain
  in the existing typed context records when the loaded TBox cannot express
  them.
- CAD correspondence, observation association, and context-record links also
  remain in typed records when the fixed PPR vocabulary cannot express them.
  The runtime never invents an ontology predicate.
- Initial PA handoff occurs only after one final target feature and one final
  resource/location selection pass their declared Phase 4 checks. If a later primitive
  contract exposes missing product or scene bindings, RA returns one
  `MissingContextBatch`; PA resumes the same controlled producer loop and
  returns a `CompositionContextBundle`. The validated post-assignment
  ontology projection and typed grounding state are revalidated or revised
  before RA binds and authors another candidate. Robot-side gaps remain
  RA-owned.
- Composition readiness means RA can bind every required primitive input and
  pass every applicable candidate-schema, backward modeled-condition, forward
  resource-state/data-dependency, physical, and projected-outcome check over
  the represented contract surface. PA handoff readiness is therefore a
  product-level handoff decision, not a guarantee of composition or execution
  readiness.
- The ontology join identifies only the configured resources broadly capable of
  the accepted feature's selected process; it does not choose among them. PA
  proposes a resource with a required check of every bound coordinate
  reference. Live MoveIt position planning establishes the limited Phase 4
  reachability scope; live MoveIt position validation is required. These checks are
  supporting evidence, not an optimal allocator or manufacturing execution.
  Recovery input
  already carries its assigned `resource_jid`; both paths later unicast to the
  selected RA.
- In the recovery transfer case, grounded ontology identifiers may describe the
  desired product outcome, but the fault, custody, live resource state, selected
  recovery event, and feasibility records remain typed runtime context. The
  work does not introduce ontology-based fault diagnosis, recovery planning,
  event generation, or allocation.

The ICRA contribution remains RA-owned agentic selection, ordering, and
parameterization of `primitive_steps` within a non-synthesizing
validation-feedback revision loop. ICRA uses agent-requested,
one-source-at-a-time exact-ref retrieval only inside auditable PA producer
operations; cross-agent product/scene needs and PA responses are batched. RA
additionally receives the reconstructed PA-authored `target_feature` plus a
bounded deterministic projection over the current feature and selected-resource
assignment. ICRA does not claim ontology design, SHACL
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

The central gap is not manual-to-action generation; Manual2Skill already
addresses that problem. It is also not dynamic selection and ordering from a
fixed primitive library or closed-loop correction; the cited multi-robot
framework already generates Python primitive sequences and replans after visual
verification. Spec2Primitives must not use either capability alone as its gap.

The gap is also not the requirement-to-capability matching problem addressed in
the prior RCIM work. That workflow can adaptively select and coordinate
predefined composite functions. The narrower investigation begins when the
product requirement and live evidence must be joined to a predefined Workcell,
one resource must be selected without encoding the answer in the ontology, and
the selected RA must compose outside its composite-function coverage from its
own complete, pinned primitive interface catalog.

A paper-ready gap statement is:

> Prior work covers manual-derived assembly structure, target-pose generation,
> predefined-function resource coordination, and VLM-generated action sequences
> over a fixed primitive library with visual recovery. It does not establish the
> specific end-to-end boundary studied here: evidence-backed ontology grounding
> of an unforeseen product request, explicit broad-capability versus current-
> reach resource assignment, a complete selected-resource-authoritative and
> hash-pinned typed primitive catalog, and structured candidate revision driven
> by deterministic validators that never synthesize repair steps. This leaves an
> empirical question about whether that combined authority, provenance, and
> validation boundary improves feasible robot-specific primitive programs over
> equal-input alternatives without a task-specific primitive recipe.

The corresponding research question is:

> Given an ontology-grounded task and selected-resource assignment, fresh
> selected-resource context, and the
> complete selected-RA-authoritative primitive-only catalog with partial local
> executable contracts, where no valid selection, ordering, and
> parameterization of the selected RA's predefined composite functions
> satisfies that task, can one RA-owned LLM composer author and revise
> validator-accepted `primitive_steps` for nominal and recovery-origin tasks
> without a task-specific recipe or complete PDDL domain/problem?

The concise contribution statement is:

> Spec2Primitives combines an architecturally required ontology-backed symbolic
> context and resource assignment with a selected-RA-authoritative, pinned
> primitive catalog, RA-owned structured candidate synthesis, and a
> non-synthesizing symbolic and physical validation-feedback loop.

The architectural hypothesis is that the ontology-backed context provides the
stable formal interface through which dynamically retrieved product knowledge,
selected-resource capabilities, and primitive bindings enter composition. The
primary evaluated hypothesis is coverage of the admitted transitions outside
the predefined composite-function interface. The feasibility hypothesis is that
validation-guided neural revision improves accepted and executed primitive
programs over equal-input primitive composers. The recovery hypothesis is
limited to transfer of the same composition mechanism.

The predefined Workcell ABox deliberately provides only broad resource matches
for the grounded high-level process. It does not entail which arm can reach the
current product; Phase 4.4 makes that decision from typed location evidence and
pinned manifest limits, then the system records one selected execution. Neither
the broad match nor the final assignment entails an ordered primitive
realization. The semantic primitive interface cards supply callable behavior
descriptions, and the evaluated hypothesis is that the RA LLM can bridge this
deliberately unmodeled task-to-primitive relation by proposing and revising
candidates that the non-synthesizing validators can accept or reject. This
motivates the agentic composer without claiming that no alternative composer
could bridge the same relation.

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
- the selected RA's primitive contracts form a complete classical planning
  domain merely because the catalog snapshot is operationally complete;
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
- dynamic selection, ordering, and parameterization from a fixed action-
  primitive library is unique to Spec2Primitives;
- moving a predefined composite function's internal sequence into an LLM prompt
  constitutes dynamic primitive composition; or
- the inherited recovery event-generation, selection, safety, or
  approval framework is an ICRA contribution.

The post-assignment ontology projection and typed evidence provide a grounded
goal and binding input rather than a PDDL problem or primitive decomposition;
there is no separate task-transition contract. Acceptance over the partial
interface surface proves neither global symbolic plan soundness or completeness
nor the realized assembly outcome. Required conditions outside all declared
contracts, invariants, evaluator endpoints, and projected-outcome checks must be
reported as `unmodeled` and fail closed.

PA and RA being separate agents is an architectural mechanism, not sufficient
novelty by itself. Dynamic action-primitive sequencing from a fixed library is
also established prior art. The contribution depends on showing that the
ontology-grounded and provenance-backed resource assignment, selected-RA catalog
authority, RA-owned structured candidate generation, and non-synthesizing
validation-driven revision improve primitive-program feasibility. The
comparison and reporting requirements under `Why an RA LLM composer is
justified relative to symbolic planning` define how that hypothesis is
evaluated. Monolithic, no-retrieval, stale-context, and no-revision ablations may
further isolate the nominal pipeline. Recovery results support composer transfer
only and must not be used to re-claim the inherited recovery stack.

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
