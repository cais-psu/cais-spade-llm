# Journal Paper 2: Recovery Framework

## Working Thesis

CAIS-SPADE-LLM provides a layered recovery framework for multi-agent manufacturing
systems where formal recovery, LLM-assisted bridge generation, safety validation,
and resource-agent execution work together after runtime failures.

This paper should focus on the recovery architecture itself. Keep Ontology RAG as
future work or as a short discussion pointer to Journal Paper 3.

## Core Idea

The system starts from a monitored production plan and reacts when execution
deviates from the expected product/resource state. Recovery is handled in layers:

1. DES-style recovery checks whether an enabled recovery path already exists.
2. LLM bridge generation proposes recovery actions when the compiled model is
   too limited.
3. Validation checks generated recovery actions before runtime execution.
4. Resource agents execute the accepted recovery task or primitive program.
5. Operator surfaces and the digital twin help inspect, preview, replay, and
   validate recovery behavior.

## Main Contributions

- A layered recovery framework that separates formal recovery, LLM-assisted
  proposal, validation-stage checks, runtime gating, and execution.
- A bridge validation pipeline that rejects unsafe or infeasible recovery actions
  before they become runtime tasks.
- Resource-agent primitive execution for recovery actions, including dry_run,
  Gazebo, digital twin, and hardware-facing paths where available.
- A practical manufacturing case study with xArm6, UR5e, task execution,
  safety constraints, and recovery after failure.

## Current System Strengths

- Recovery is tied to product/resource state rather than free-form text.
- DES enabledness and validation-stage checks are separate from downstream
  runtime gating.
- The system already has bridge-visible primitive programs and resource-owned
  execution surfaces.
- Monitor, Teach, Preview in Gazebo, Replay in Twin, and digital twin provide
  useful operator-facing recovery workflow pieces.

## Main Limitation To Admit

The current recovery framework has dynamic context retrieval, but the retrieval
surface is mostly procedural and ref-based. The LLM can request useful context,
but the framework does not yet use a first-class PPR graph to proactively decide
which Product, Process, Resource, primitive, fact, and constraint context should
be retrieved before generation.

This limitation becomes the motivation for Journal Paper 3.

## Paper Outline

1. Introduction
   - Flexible manufacturing needs recovery after runtime failure.
   - Pure formal recovery is safe but limited.
   - Pure LLM recovery is flexible but hard to trust.
   - This paper combines layered recovery with validation and execution.

2. Related Work
   - Multi-agent manufacturing systems.
   - Product agents and resource agents.
   - DES and supervisory control for manufacturing.
   - LLM planning and robot task generation.
   - Runtime safety validation and recovery.

3. System Architecture
   - Product agent, resource agents, and central controller.
   - Task planning, execution monitoring, and failure detection.
   - Resource-agent execution modes.
   - ROS2/Gazebo/MoveIt and digital twin context.

4. Layered Recovery Framework
   - Failure context.
   - DES recovery path.
   - LLM bridge proposal path.
   - Validation-stage checks.
   - Runtime dispatch and resource-agent execution.

5. Primitive Recovery Execution
   - Bridge events and primitive_steps.
   - Resource-owned primitive catalogs.
   - Primitive validation, event_facts, and projected state.
   - Recovery macro execution.

6. Experiments
   - Normal execution vs failure execution.
   - DES-only recovery vs LLM bridge recovery vs full validated recovery.
   - dry_run, Gazebo, digital twin, and hardware-facing demonstrations where
     available.

7. Discussion
   - What the framework handles well.
   - Limits of ref-based dynamic retrieval.
   - Why PPR Ontology RAG is the next step.

8. Conclusion
   - Layered recovery improves flexibility while preserving validation.

## Experiment Plan

Use small, repeatable failure cases:

- Resource enters failed state during a task.
- Part is held by the wrong resource.
- Destination or expected state does not match the plan.
- Primitive proposal misses required event_facts.
- Recovery action is not enabled from the current symbolic state.

Compare:

- No recovery.
- DES-only recovery.
- LLM bridge without full validation.
- Full layered recovery framework.

Measure:

- Recovery success rate.
- Invalid proposal rejection rate.
- Safety violation count.
- Recovery latency.
- Number of LLM turns.
- Primitive program validity.

## Figures To Produce

- Layered recovery architecture.
- Validation-stage vs runtime-gating diagram.
- Recovery sequence example.
- Primitive recovery execution trace.
- Experiment result table or bar chart.

## Immediate Writing Tasks

- Freeze the title and contribution list.
- Extract one representative recovery scenario for the running example.
- Write the before/after recovery example.
- Decide which experiments can be repeated reliably before submission.
