# Spec2Primitives Background Literature Review

## Goal

Prepare a concise Background/Related Work section for:

> **Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition
> in Industrial Robotic Assembly**

Focus on whether prior work already lets an LLM dynamically select, order,
repeat, bind, validate, and revise robot primitives at runtime.

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

