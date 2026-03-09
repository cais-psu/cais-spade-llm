# Online Safety Supervisor

## Purpose

The runtime safety monitor already blocks immediate safety violations on task
start. That is not enough for response and precedence-style constraints,
because the current execution can remain locally legal while becoming
unrecoverable. A typical example is a completed `place_insert` that now
requires a later `move_home`, or a task that must wait until another resource
finishes a prerequisite.

This module adds a supervisory layer over:

- the plan FSA state `x`
- the vector of safety DFA states `q_vec`
- the AP-relevant resource state signature `sig`

The combined runtime state is a product state `(x, q_vec, sig)`.

## Formal Model

- Plan automaton:
  `P = (X, E, T, x0, Xm)`
- Safety automata:
  one DFA per safety rule
- Product state:
  `(x, q_vec, sig)`

The supervisor precomputes a winning set `W`:

- `W` contains all reachable product states from which there exists at least
  one continuation to a marked plan state while all safety DFAs remain
  satisfiable
- terminal acceptance is checked with the standard final empty-step evaluation
  used by the offline safety validator

This is a supervisory-control-style nonblocking policy over the plan/spec
product automaton.

## Runtime Statuses

- `safe`
  current product state is in `W` and all current DFA states are accepting
- `pending_obligation`
  current product state is in `W`, but one or more DFA states are non-accepting
  because an obligation is still open
- `inevitable_violation`
  current product state is not in `W`; no safe completion exists in the current
  modeled plan
- `violated`
  some DFA is already in its violation state
- `blocked_candidate`
  the current state is still recoverable, but a specific candidate start would
  leave the winning set

## Policy

### Candidate task start

Before allowing `task.start`:

1. run the existing immediate safety check
2. check whether the product successor is still in `W`

If the successor is outside `W`, the task is blocked even if it does not
trigger an immediate DFA sink transition.

This lets the system distinguish:

- wait: current state is recoverable, but this task is too early
- replan: current state is no longer recoverable at all

### After task completion or failure

After every `.done` and `.fail` event:

1. update the plan FSA monitor
2. update the safety monitor
3. classify the current product state

If the status is `inevitable_violation` or `violated`, the controller triggers
replanning immediately.

If the status is `pending_obligation`, the controller exposes the diagnosis but
does not replan yet.

## Waiting vs Replanning

This distinction is the main reason for the supervisor.

Example:

- `insert_pin` requires `clamp.done`
- `clamp` is assigned to another resource and is still reachable later

Then:

- starting `insert_pin` now is blocked as `blocked_candidate`
- current state remains in `W`
- no replanning is needed

If `clamp` becomes unreachable, the current state leaves `W` and the diagnosis
becomes `inevitable_violation`, which triggers replanning.

## Repair Hints

When the current state is still in `W`, the supervisor can compute a shortest
safe suffix in the precomputed product graph. The first steps of that suffix are
returned as `safe_suffix_hint`.

This is advisory only. The supervisor does not insert or dispatch new tasks by
itself in v1.

If the current state is outside `W`, no safe suffix exists in the current model.
In that case the supervisor returns diagnosis only and leaves structural repair
to the replanner.

## Implementation Split

- `PlanSafetyValidator.compute_winning_set(...)`
  builds the reachable joint product graph and computes `W`
- `OnlineSafetySupervisor`
  performs runtime classification and candidate checks using O(1) membership in
  `W`
- `CentralControllerAgent`
  decides when to notify, block, or replan based on the supervisor diagnosis

## Why This Generalizes

The supervisor does not special-case:

- `move_home`
- precedence rules
- mutex rules
- specific robots or stations

It only relies on:

- the plan automaton
- the safety DFA semantics
- the AP-relevant resource state tracked by the existing monitor

That makes the same mechanism work for response, precedence, exclusion, and
state-based rules without rule-family-specific runtime hacks.
