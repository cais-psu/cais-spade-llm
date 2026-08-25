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
robot-independent assembly plan
        ↓
robot-specific primitive_steps
        ↓
validated executable behavior
```

The composition contribution occurs at a finer-grained execution abstraction
than product-level assembly planning. It is not itself low-level robot control.
Trajectory generation, force control, velocity commands, and joint commands
remain below `primitive_steps`.

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

The primitive contract should describe:

- the operation attempted;
- accepted inputs and parameters;
- termination status;
- direct evidence or results; and
- the resource-owned evaluator used to determine feasibility.

The contract does not need to assert exhaustive symbolic preconditions and
effects or claim the complete task-level outcome. The combined result of several
primitives may establish an assembly outcome that no individual primitive can
establish alone.

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

> Dynamic primitive composition is the runtime selection, parameterization,
> ordering, validation, and revision of resource-owned primitive calls into
> robot-specific `primitive_steps`, using the grounded assembly task, fresh
> robot state, and the currently retrieved primitive catalog.

The word `dynamic` requires more than replacing parameters in a fixed template.
The evaluation must demonstrate that:

1. No product-specific assembly program or completed `primitive_steps` is
   supplied.
2. RA receives only the available atomic robot primitives and fresh resource
   state.
3. Primitive selection, ordering, bindings, and parameters are authored at
   runtime.
4. Changes in robot state or available capabilities can change the resulting
   primitive program.
5. Concrete state, IK, collision, and trajectory rejection evidence can cause a
   structural revision rather than only a blind retry or parameter perturbation.
6. Different assembly requirements require different composition structures,
   rather than variants of one stored sequence.

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
> product intent can be dynamically transformed into a feasible and revisable
> primitive program without providing a product-specific assembly program.

The corresponding research question is:

> Can separating product grounding by PA from robot embodiment grounding by RA
> improve the validity and feasibility of dynamically composed
> `primitive_steps` for previously unseen industrial assembly requirements?

The concise contribution statement is:

> Manual2Skill grounds manuals into assembly structure; Spec2Primitives grounds
> assembly requirements into dynamically composed and validator-revised robot
> primitive programs.

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
- `primitive_steps` schema validity;
- robot-local feasibility validation;
- validation-driven revision convergence;
- simulation execution; and
- physical robot execution.

PA and RA being separate agents is an architectural mechanism, not sufficient
novelty by itself. The contribution depends on showing that the separation,
provenance-backed assembly contract, fresh resource grounding, and
validation-driven revision improve primitive-program feasibility relative to a
monolithic method, a no-retrieval condition, and a no-validation-revision
condition.
