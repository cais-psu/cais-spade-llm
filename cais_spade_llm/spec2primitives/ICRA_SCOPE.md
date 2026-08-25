# ICRA 2027 Spec2Primitives Scope

## Research title

**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**

## Fundamental research challenge

The central research question is:

> Can an underspecified product requirement and heterogeneous product evidence
> be transformed at runtime into a validated, robot-specific primitive program
> without a product-specific assembly program or hidden ground truth?

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
validated executable behavior
```

The research contribution is not that PA and RA exchange messages. It is the
combination of:

1. PA grounding an incomplete product requirement from documents, CAD, and
   RGB-D evidence into provenance-backed `target_feature`, target pose,
   insertion axis, tolerances, required outcome, and unresolved evidence.
2. RA, where RA means RobotAgent, grounding that assembly task against fresh
   robot state and its resource-owned primitive catalog before authoring
   robot-specific `primitive_steps`.
3. Robot-local validation returning concrete rejection evidence that causes RA
   to revise an infeasible primitive program instead of regenerating blindly.

Exact-ref retrieval, PA and RA live cards, ordered interaction records, and
simulation are supporting infrastructure and evaluation evidence. They are not
the fundamental contribution by themselves.

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

> Separating product grounding from robot embodiment grounding, while
> connecting them through a provenance-backed assembly contract and
> validation-driven revision loop, improves synthesis of feasible primitive
> programs for previously unseen robotic assembly requirements.

The controlled Medium Gear case is the development starting point, not
sufficient evidence of general primitive composition. The final evaluation must
withhold product-specific assembly programs, provide only atomic robot
primitives, include CAD distractors, vary observations and object placements,
exercise vague or incomplete requirements, and require different primitive
composition structures beyond Small, Medium, and Large variants of one gear
sequence.

Evaluation must report product grounding, primitive-program validity, revision
convergence, robot-local feasibility, and simulation execution separately. It
must compare the complete PA-RA method against at least a monolithic method, a
no-retrieval condition, and a no-validation-revision condition. Retrieval,
schema validation, replay, simulation, and physical execution remain distinct
claims.

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
product requirement: assemble Medium Gear
            ↓
PA retrieves manual/specification/CAD
            ↓
PA grounds target_feature, target pose, insertion axis, tolerances
            ↓
RA retrieves fresh resource state and resource-owned primitive catalog
            ↓
RA authors primitive_steps
            ↓
state checks + IK/collision/trajectory validation
       ↙ rejected                         accepted ↘
concrete feedback → RA revision       RA execution
```

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
