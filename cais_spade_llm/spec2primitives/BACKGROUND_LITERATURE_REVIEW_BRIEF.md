# Spec2Primitives Background Literature Review

Current implementation: PA-owned product grounding with deterministic contract and evidence checks, budgeted PA corrections, and MoveIt-validated arm assignment, followed by selected-RA context and unbound structural drafting. The later binding, executable-validation and outcome workflow below is research methodology, not completed runtime behavior. See [implementation status](IMPLEMENTATION_PLAN.md) and [bias experiments](BIAS_VALIDATION.md).

## Goal

Prepare a concise Background/Related Work section for:

> **Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition
> in Industrial Robotic Assembly**

Focus on whether prior work already lets an LLM dynamically select, order,
repeat, bind, validate, and revise robot primitives at runtime.

## High-level summary

### Introduction and motivation

- Flexible manufacturing must handle new or incomplete product requirements
  without manually programming every product variant. For example, a new gear,
  target location, or assembly instruction can otherwise require an engineer to
  rewrite the robot program.
- Industrial robots already provide reusable operations such as sensing,
  moving, grasping, placing, and releasing. However, those operations do not
  explain how they should be combined for a new product requirement.
- A shared ontology connects the product, its required process, and available
  robot capabilities so that different parts of the system use the same
  meaning.
- The goal is to connect product intent to a safe robot-specific program while
  preserving clear checks before physical execution.

### Research gap

- Existing work covers manual understanding, robot skill selection, planning,
  and execution. Many approaches, however, assume a prepared task recipe, a
  complete planning model, or a fixed library shared by the whole system.
- Ontologies can represent product requirements, processes, resources, and
  capabilities, but that shared knowledge does not itself provide the new
  task-specific primitive sequence.
- The open question is whether an LLM can create a new program after the product
  goal and robot have been selected, using only that robot's available
  primitives and current state.
- A plausible-looking program may still skip a necessary movement, use missing
  information, or fail physically. Independent checks must therefore evaluate
  the proposal without silently rewriting it.

### Methodology

- First, the system interprets the product request and identifies what evidence
  is still needed. It retrieves only relevant instructions, drawings, part
  models, or camera observations to describe the current product state and the
  required final state.
- The accepted meaning is recorded in an ontology that links the product goal,
  required process, and available robot capabilities. The ontology constrains
  what the task means, but it does not contain the primitive recipe.
- The grounded requirement is matched to available robots. Capability,
  live MoveIt position plans support the current arm
  assignment. IK and collision-free motion checks belong to later executable validation.
- The selected robot then reports its current state and its available primitive
  operations, including their inputs, outputs, limits, and known conditions.
- The LLM uses this information to propose the program structure. For example,
  it may choose to sense the parts, calculate pick and placement targets,
  approach, grasp, transport, place, and release. No product-specific sequence
  is supplied in advance.
- The selected operations reveal the exact values needed for execution. The
  system resolves robot-owned values locally and retrieves additional product
  or scene evidence only when an operation requires it.
- The LLM then produces a fully parameterized program. Independent checks test
  whether every operation exists, required values are connected correctly, the
  order is logically consistent, robot-state conditions hold, motion is
  collision-free and reachable, and the predicted result satisfies the product
  goal.
- A failed check returns a specific finding, such as a missing target pose,
  unsupported operation, broken dependency, or unreachable motion. The LLM—not
  the checker—creates the revised program.
- Only an accepted program is executed. New observations are then compared with
  the requested product state to determine whether the assembly actually
  succeeded.

```text
Instructions + drawings + part models + camera observations
                          ↓
          Ontology-grounded product goal and process
                          ↓
             Suitable robot selected
                          ↓
        Current state + available primitives
                          ↓
              LLM-generated program
                          ↓
             Independent validation
                 ├── failure → findings returned to LLM → revised program ↺
                 └── accepted
                          ↓
                       Execution
                          ↓
                Observed product outcome
```

### Case study intent

- Use a dual-robot industrial assembly taskboard with gear components. The
  product-specific program is withheld so the method must interpret the request
  and compose the required operations at runtime.
- The nominal case tests whether a selected robot can generate and execute the
  assembly program. A recovery case tests whether the same method can respond
  to changed physical conditions, such as a slipped part, without a separate
  recovery-specific recipe.
- Report generated-program validity, physical feasibility, execution success,
  and observed assembly outcome separately.

## Papers to read first

1. [*Adaptive task planning and coordination in multi-agent manufacturing
   systems using large language models*](https://doi.org/10.1016/j.rcim.2026.103245)
   — our prior framework. Clearly identify what Spec2Primitives inherits and
   where predefined composite functions stop.

2. [*A Closed-Loop Multi-Agent Framework for Robust Multi-Robot
   Manipulation*](https://arxiv.org/abs/2607.06990) — the closest external work.
   Compare its fixed action-primitive library, Manipulation Agent, Verification
   Agent, parameter grounding, and recovery loop.

3. [*Trust the PRoC3S: Solving Long-Horizon Robotics Problems with LLMs and
   Constraint Satisfaction*](https://proceedings.mlr.press/v270/curtis25a.html)
   — compare LLM-proposed skill sequences, parameter solving, physical
   constraints, and revision after failure.

4. [*Code as Policies: Language Model Programs for Embodied
   Control*](https://arxiv.org/abs/2209.07753) — compare LLM-generated programs
   over predefined robot APIs.

5. [*Manual2Skill*](https://arxiv.org/abs/2502.10090) and
   [*Manual2Skill++*](https://arxiv.org/abs/2510.16344) — compare manual
   grounding, assembly graphs, connector reasoning, target poses, action
   generation, and execution.

6. [*A formal framework for the specification and verification of robotic
   skills composition for autonomous
   behaviors*](https://doi.org/10.1016/j.robot.2025.105041) — compare formal
   skill composition, Skill Petri nets, model checking, and code generation.

7. [*Capability-based Frameworks for Industrial Robot Skills: a
   Survey*](https://doi.org/10.1109/CASE49997.2022.9926648) — use for the
   industrial task-skill-primitive definitions.

8. [*SkiROS2: A skill-based Robot Control Platform for
   ROS*](https://arxiv.org/abs/2306.17030) — compare knowledge-backed skill
   descriptions, pre-/post-conditions, planning, and execution.

9. [*PDDLStream*](https://arxiv.org/abs/1802.08705) and
   [*LLMs Can't Plan, But Can Help Planning in LLM-Modulo
   Frameworks*](https://arxiv.org/abs/2402.01817) — use for a fair comparison
   with symbolic planning and generator-verifier systems.

Also search recent ICRA, IROS, RSS, CoRL, CASE, and ICAPS papers using:

- `LLM robot primitive composition`
- `LLM robot skill composition validation`
- `robot program synthesis primitive library`
- `industrial robot skills ontology composition`
- `LLM assembly planning primitive sequence`
