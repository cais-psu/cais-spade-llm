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

## Validation-Authority Boundary

The recovery outline uses explicit agent ownership rather than treating the PA
as a proxy validator.

| Recovery operation | Authoritative component | Paper interpretation |
| --- | --- | --- |
| Proposal generation | LLM called by PA | `pure_llm` produces three candidate events and selects one index; `neurosymbolic` produces one to `candidate_proposal_budget` next-event candidates without selecting. |
| Syntax and grounding | PA | Verifies response structure and exact formal-system bindings. |
| Transition feasibility | PA | Verifies expected-start agreement and a meaningful projected state or blocker effect. |
| Physical feasibility | Responsible live RA | Uses a fresh RA snapshot and resource-specific feasibility checks. |
| Local plant model | Responsible live RA | Privately supplies `G_r` as a task-level extended finite automaton with exact variables, finite domains, valuation, local alphabet, guards, updates, event classifications, marked conditions, and descriptor fingerprint. The LLM does not receive this descriptor. |
| Candidate safety | Live CCA | Uses current CCA safety rules and `validate_outline_macro_recovery_safety()`. |
| Model-based selection | PA | In `neurosymbolic`, compares PA/RA/CCA-valid successors through exact open recovery obligations and recovery-relevant admissible enabled events from RA-owned models. |
| Candidate commit | PA | Rechecks the PA fingerprint. `pure_llm` preserves valid LLM selection; `neurosymbolic` commits only a unique nondominated successor. |
| Post-outline safety generation | CCA | Produces the recovery safety bundle after outline completion. |
| Runtime safety gating | CCA | Supervises dispatch and execution; this is not the outline-time candidate check. |
| Primitive execution | Responsible RA | Generates and executes the resource-owned primitive program. |

For each candidate, the PA first performs its own checks. It then sends a
correlated `recovery_outline_physical_validate` request to the exact RA named by
`resource_jid`. The RA refreshes `get_recovery_snapshot()`, validates physical
feasibility, and replies with its snapshot, snapshot fingerprint, local recovery
DES descriptor, and descriptor fingerprint. RA-valid candidates
are sent to the CCA through `recovery_outline_safety_validate`. The CCA returns
the safety findings, active rule identifiers, cleared and remaining condition
identifiers, and its safety-rule fingerprint. Request id, recovery session id,
turn index, PA state fingerprint, and sender JID are checked before PA accepts a
reply. Missing, malformed, stale, incorrectly routed, or timed-out validation
fails closed.

This wording is important for the paper: the PA orchestrates validation but does
not claim RA physical authority or CCA safety authority. Outline validation is
also snapshot-specific. Passing it does not guarantee primitive synthesis,
inverse kinematics, collision-free MoveIt planning, trajectory execution,
Gazebo behavior, or hardware execution.

## Neurosymbolic Selection Method

The experimental comparison has two explicit modes. `pure_llm` is the baseline:
the LLM authors exactly three one-action candidates and returns
`selected_candidate_index`. The selected candidate remains authoritative only
when PA, the responsible RA, and CCA all accept it. `neurosymbolic` keeps LLM
proposal generation mandatory because the recovery event alphabet is incomplete,
but the LLM does not rank or select its one to `candidate_proposal_budget`
proposals.

Use conventional DES notation for the plant:

`G = G_P || (||_{r in R} G_r)`.

`G_P` is the product automaton and each RA owns its local extended finite
automaton `G_r`. Its descriptor declares the exact local variables and finite
domains, current valuation, local event alphabet, controllable and observable
classifications, guards, updates, and local marked conditions. The CCA induces
the supervised plant `S/G`. PA stores and composes these descriptors but does
not infer their semantics from event names, descriptions, resource types, or
Case 3 tokens.

Robot descriptors contain task-level `RobotTaskProgram` transitions rather
than controller primitives, sensing functions, or geometry-computation
functions. Continuous pose and workspace evidence remain outside the finite DES
state and are used by the RA physical validator. `RobotAgent` retains
`gripper_state` because robot-task enabledness and physical feasibility use it
with `held_part`. `PrintingAgent` represents `pause_job`, `resume_job`, and
`cancel_job` through `resource_state` only. Printer snapshots and primitive
execution still retain `job_state` and `active_job` as runtime evidence, but
they are not PrintingAgent DES variables. Another ResourceAgent declares only
the exact finite variables needed by its own enabledness, marked conditions,
safety checks, or physical validator. It must implement the same private
descriptor interface or provide an explicit initialization descriptor; a
missing or invalid descriptor fails closed.

The LLM receives only compact live symbolic state, observations needed for
grounding, reachable locations and named poses, recovery goals, active safety
rules, and exact current-turn validation findings. Event alphabets, guards,
updates, marked conditions, descriptor fingerprints, primitive catalogs, and
accepted-transition stacks are not prompt material. Complete private models
remain auditable in internal session state and outline result artifacts.

The response schema uses the union of RA-declared state-field names so one LLM
response can contain candidates for different resources. PA nevertheless
authorizes each candidate against only the descriptor owned by its exact
`resource_jid` and enforces each field's declared `resource` or `part` scope.
Thus `gripper_state` is a valid RobotAgent variable only when that RobotAgent
declares it. `job_state` is not accepted for that robot or for the current
PrintingAgent private DES descriptor. Resource-specific values must belong to
the responsible RA's exact finite domain. Adding another ResourceAgent changes
its descriptor, not PA validation code. Accepted declared values are preserved
in PA state and the RA/CCA projections.

The LLM can author any nonempty exact `event_name` and may optionally introduce
a new nonempty `resource_state` or `part_state` in the candidate end state.
These state values are uninterpreted labels: they do not extend or mutate the
live RA descriptor. A committed label is retained only in the recovery
session's projected state and must be matched exactly by the following
candidate. A label alone supplies neither transition semantics nor selection
progress. PA accepts it only with another concrete declared effect or when the
exact state value clears a supplied recovery obligation. No validation or
selection stage derives motion, acquisition, release, safety, or progress from
wording inside an LLM-authored event or state name.

For current composed state `q`, let `O(q)` be the exact open recovery
obligations: PA Recovery Goals, continuation blockers, reentry requirements,
and CCA safety-condition identifiers. Let `Q_m^R` be the set of
recovery-compatible marked states. Let `Gamma^R_{S/G}(q)` denote the
recovery-relevant controllable events enabled by exact RA guards at the
CCA-accepted projected state, after backward relevance from `O(q)` and `Q_m^R`.
Every PA/RA/CCA-valid LLM event `e_i` yields `q'_i = delta(q, e_i)`.

The event progresses when it strictly reduces `O(q)` without introducing
another obligation, or preserves `O(q)` while enabling at least one previously
disabled recovery-relevant event in the successor. The evidence also records
the exact RA event whose update realizes and is consumed by the candidate.
Literal enabled-set inclusion is not used as the progress test because an
ordinary DES event often disables itself while enabling its successor.

The partial order is:

`e_i` dominates `e_j` iff `O(q'_i) ⊆ O(q'_j)` and
`Gamma^R_{S/G}(q'_i) ⊇ Gamma^R_{S/G}(q'_j)`, with at least one strict relation.

One unique nondominated event is appended and projected before the next LLM
turn. No valid/progressing event returns `need_revision`. Equivalent or
incomparable nondominated events return `selection_ambiguous`, append nothing,
and expose only compact obligation and enabled-event comparison evidence to the
next LLM request. Descriptor and supervisor fingerprints remain private in the
result artifact. The next request asks for at most one
representative per equivalent successor class. Three unresolved revisions under
the same plant and supervisor fingerprints terminate as `selection_unresolved`.
If a representative is rejected, its exact current PA/RA/CCA finding is shown
alongside the still-active one-representative constraint until a transition is
committed or an authoritative fingerprint changes.
No LLM preference, candidate position,
`event_name`, `rationale`, resource name, token ordering, or random tie breaker
is used. PA, RA, and CCA never invent a symbolic fallback event.

CCA owns safety projection and returns `safety_dfa_states_before` and
`safety_dfa_states_after`. The first candidate uses the live CCA DFA state;
later unexecuted outline transitions use the previously selected projected DFA
state. Temporary validation monitors do not mutate the live monitor. Stale rule
or live-state fingerprints fail closed.

The method should be described as one-step receding-horizon successor dominance,
not BFS, global shortest-path recovery, or a formal nonblocking supervisor over
a complete event alphabet. Its explicit limitation is that incomparable
successors cannot be resolved without a separately declared objective. Outline
validation remains snapshot-specific and does not guarantee primitive,
trajectory, Gazebo, or hardware execution.

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
