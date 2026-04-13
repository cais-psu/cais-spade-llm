# Hybrid DES Bridge v1

## Purpose

`hybrid_des_v1` is the recovery mode intended to act as the digital-twin
recovery planner for disruption handling in the two-robot assembly cell.

The target use case is a synchronized physical + virtual recovery workflow:

1. The physical cell encounters a disruption.
2. The digital twin is synchronized to the disrupted robot/part/workspace state.
3. Hybrid mode generates a constrained recovery space from that state.
4. The DES solver searches for a safe recovery trace.
5. The solved trace is validated against physical and safety constraints.
6. A validated recovery proposal is returned for execution.

This mode is the right fit for scenarios that involve deadlock resolution,
cross-robot handoff, constrained resequencing, and "what-if" recovery
evaluation in the twin before physical execution.

## Current Loop

The current implementation already has the core control flow:

`evaluate_grounding -> domain_generation -> compose_and_solve -> validate_plan -> finalize`

High-level role of each phase:

- `evaluate_grounding`: detect whether the disrupted state is sufficiently
  observed to plan recovery.
- `domain_generation`: ask the LLM to author a recovery plant automaton.
- `compose_and_solve`: compose the plant with safety DFAs and search for a
  valid recovery trace.
- `validate_plan`: check the solved trace against physical feasibility and
  ordering constraints.
- `finalize`: emit the accepted recovery proposal.

## Why Hybrid Needs More Work

The current implementation is close to the intended architecture, but it still
leans too heavily on static prompt context and does not expose enough runtime
recovery structure to the LLM.

Current limitations:

- Recovery blockers in prompt construction are pulled too directly from static
  `llm_input`, instead of being recomputed from synchronized symbolic state.
- The domain-generation prompt shows current state and blockers, but it does
  not clearly state what must be restored for nominal execution to resume.
- Revision prompts show raw plant, solver, and feasibility feedback, but they
  do not summarize persistent bad patterns that should be ruled out.
- Final output is still mostly an outline task list rather than a full
  digital-twin recovery result with revision and validation metadata.
- There is no dedicated design note explaining hybrid mode's intended role,
  interfaces, and evaluation contract.

## Design Goal

Hybrid mode should behave like a digital-twin recovery synthesizer, not just a
one-shot plant generator.

That means the mode must:

- derive active blockers from synchronized symbolic runtime state
- expose deadlock and continuation-resume context in the prompt
- preserve revision memory across failed plant attempts
- summarize what previous failed revisions already ruled out
- emit recovery proposals with enough metadata for evaluation in the twin and
  later execution in the physical cell

## Required Modifications

### 1. Dynamic Runtime Blockers

Hybrid prompt building should derive blockers from the current symbolic state,
not only from static `llm_input`.

The blocker computation should include at least:

- observation-required blockers
- resource terminal-state blockers
- assembly-order blockers
- deadlock-style shared-workspace blockers
- persistent reachability blockers proven by prior validation failures

These blockers should be recomputed for every domain-generation revision.

### 2. Continuation / Resume Context

The domain-generation prompt should explicitly show what must be restored for
nominal execution to resume.

Add prompt sections for:

- `Continuation / Resume Conditions`
- `Current Deadlock Conditions`
- `Observed Part Poses And Workspace Facts`
- `What Must Be True For Recovery To Count As Complete`

The prompt should state that a valid recovery plant must:

- clear the immediate disruption
- restore a state from which nominal execution can continue
- respect assembly-order and shared-workspace constraints
- allow cross-robot handoff or resequencing when required by reachability

### 3. Actionable Revision Feedback

Revision prompts should not only repeat raw failures. They should also explain
what the failures rule out for the next revision.

Add compact summaries for:

- persistent infeasible action patterns
- ruled-out resource/part pairings
- solver unsat or safety-blocked outcomes
- continuation conditions that remain unmet after a solved trace

Examples of useful guidance:

- do not assign a part to a resource whose workspace already failed on the
  current observed pose
- do not model direct relocation without acquisition/holder transfer
- do not stop at a recovery state that leaves the nominal suffix blocked

### 4. Revision-State Memory

Hybrid mode should preserve revision progress across failed attempts instead of
starting from scratch each time.

Track in session state:

- `last_rejected_plant`
- `last_solver_trace`
- `last_feasibility_findings`
- `persistent_constraint_summary`
- `revision_history`

These should be used in the feedback prompt so the next plant is a deliberate
revision, not a fresh attempt that repeats already disproven branches.

### 5. Stronger Output Contract

The final hybrid proposal should contain enough metadata to support digital-twin
evaluation and later physical execution.

Required final fields:

- `outline_tasks`
- `engine = "hybrid_des_v1"`
- solver status
- domain revision count
- accepted recovery trace length
- validation status
- summary of cleared blockers / resume conditions

Useful optional fields:

- solver explored-state count
- feasibility finding count
- grounding observation count
- revision-history summary

### 6. Observation-First Recovery

Observation handling should remain an explicit part of the documented hybrid
workflow.

The design contract should say:

- hybrid may pause to request real observation execution
- observed part poses are written back into symbolic recovery state
- domain generation only starts once grounding is sufficient

This is important because the hybrid method is supposed to operate on the
current synchronized digital-twin state, not on stale nominal assumptions.

### 7. Hybrid-Specific Tests

Hybrid mode needs tests that validate it as a runtime recovery method, not only
as an alternative branch in the dry-run harness.

Add or strengthen tests for:

- dynamic blocker derivation in the hybrid prompt
- continuation / deadlock context visibility
- actionable feedback prompt contents after failed revisions
- finalize output metadata
- case-3 style cross-robot recovery traces

## Prompt Expectations

The LLM should not be asked to "guess a plausible repair."

It should be asked to generate a recovery plant that is:

- grounded in the current synchronized state
- expressive enough for cross-resource recovery
- structurally capable of clearing the active blockers
- shaped so the solver can find a safe accepting trace

The prompt should therefore emphasize:

- this is a recovery-space generation task
- the solver will choose a trace, but the LLM defines what is possible
- marked states must represent both recovery completion and resumability
- the recovery space may include observation, handoff, resequencing, and
  resource-reset actions when justified by the disrupted state

## Revision Prompt Expectations

When a plant is rejected, the next revision prompt should answer:

- What was structurally wrong with the plant?
- What solver outcome showed it was incomplete or unsafe?
- What physical constraints invalidated the solved trace?
- What resource/part/action combinations should not be repeated?
- What continuation conditions still need to be cleared?

The feedback prompt should make it easy for the LLM to modify the recovery
space rather than regenerate an unrelated one.

## Output Semantics

The finalized proposal should be interpretable as a twin-side recovery result:

- the accepted recovery trace found in the plant
- the executable recovery outline derived from that trace
- the reason the trace is valid
- the conditions it clears
- the revision effort required to obtain it

This is the object that can be compared across disruption scenarios and used to
measure recovery decision time, trace length, success rate, and physical
execution outcomes.

## Acceptance Criteria

Hybrid DES Bridge v1 should be considered aligned with the intended testbed
once all of the following are true:

- recovery blockers are derived from synchronized symbolic state at each turn
- domain-generation prompts explicitly expose resumability and deadlock context
- repeated failed revisions receive feedback that rules out persistent bad
  recovery branches
- final proposals include both executable tasks and recovery metadata
- case-3 style disruption scenarios can be evaluated in hybrid mode with
  grounding, revision, solver search, feasibility validation, and final output

## Planned File Touchpoints

Primary implementation files:

- `cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/modes/hybrid_des_v1.py`
- `cais_spade_llm/agents/intelligent_product/replanner/llm_bridge/prompts/hybrid_des_v1.py`
- `test/test_case3_bridge_dryrun.py`

This document exists to define the intended behavior before code changes are
made.

## Scope Assumptions

- Hybrid remains the global recovery-space + solver method.
- Multi-turn remains the incremental comparator, not the target of this design
  document.
- The digital twin is represented here by synchronized symbolic state plus
  solver/validator what-if evaluation, not by embedding a physics simulator
  inside hybrid mode itself.
