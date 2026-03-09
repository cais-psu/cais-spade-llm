# `ap_state` / `ap_event` Strategy for Mutex Safety

## Purpose

This document describes the safety abstraction used to solve shared-workspace
mutex problems without hard-coding specific action names into the safety logic.

The main issue with action-only APs is that they are too short-lived:

- an action AP is active only while the action runs
- spatial risk often continues between actions and after an action completes

For example, a rule such as:

- "both arms should not be at the assembly board at the same time"

cannot be enforced correctly with only:

- `place_approach`
- `place_insert`

because the unsafe condition is not "both are executing the same function"
but rather "both are in the same constrained state slice of the workspace."

## Two AP Kinds

The implementation uses two real AP kinds in compiled safety formulas.

### `ap_event`

Transient AP active during task execution:

```text
ap_event/<process>/<product>/<resource>/<function>/<context>
```

Example:

```text
ap_event/assembly/any/ur5e/place_approach/destination=assembly_board-v1
```

Use this for:

- execution overlap
- entry into a constrained state slice
- action-level ordering rules

### `ap_state`

Persistent AP active while a resource remains in a state:

```text
ap_state/<process>/<product>/<resource>/<state>/<context>
```

Example:

```text
ap_state/assembly/any/ur5e/positioned/destination=assembly_board-v1
```

Use this for:

- persistent spatial occupancy
- state-dependent exclusion
- safety obligations that remain active between actions

## Why Both Are Needed

For a shared-board mutex rule, `ap_state` alone is not enough.

If only these APs were used:

```text
ap_state/.../positioned@board
ap_state/.../placed@board
```

then the monitor would see the first arm occupying the board after
`place_approach.done`, but it could still miss the second arm entering at the
start of its own `place_approach`.

That is why the compiled mutex uses:

- `ap_event` for the entry action
- `ap_state` for the persistent occupied states

This combination covers:

- entering the board
- remaining at the board between actions
- still occupying the board after placement

## Compile-Time Expansion

The LLM does not generate the final AP list directly.

Instead, safety compilation may use `ap_selector`, a compile-time-only macro
that expands into the required `ap_event` and `ap_state` terms using the tools
catalog and the tool/state graph.

Example selector:

```json
{
  "type": "ap_selector",
  "resource": "ur5e",
  "match": {
    "context": { "destination": "assembly_board-v1" }
  },
  "include_entry_events": true,
  "include_state_aps": true
}
```

For the current assembly workflow, that selector expands to:

```text
ap_event/assembly/any/ur5e/place_approach/destination=assembly_board-v1
ap_state/assembly/any/ur5e/positioned/destination=assembly_board-v1
ap_state/assembly/any/ur5e/placed/destination=assembly_board-v1
```

This is derived from tool metadata:

- `function`
- `in_state`
- `out_state`
- `context_mapping`

It is not hard-coded as:

- "if board, then use place_approach + positioned + placed"

The same expansion mechanism can be used for other resources and processes as
long as the tool catalog provides the same semantic fields.

## Mutex Encoding

For two resources, the shared-board rule becomes:

```text
G !((u_enter | u_positioned | u_placed) & (x_enter | x_positioned | x_placed))
```

This is still a standard mutex formula.

The difference is that each side of the mutex is a disjunction of:

- one or more entry `ap_event`s
- one or more persistent `ap_state`s

So the logic is:

- if resource A has entered or is still occupying the constrained state slice
- and resource B has entered or is still occupying that same slice
- then the execution is unsafe

## Online Semantics

At runtime:

- `ap_event` is active while the task runs
- `ap_state` is stored per resource and remains active until that resource
  changes state

Example:

1. `ur5e.place_approach(board)` finishes
2. monitor records:

```text
ap_state/assembly/any/ur5e/positioned/destination=assembly_board-v1
```

3. `xarm6.place_approach(board)` requests start
4. the online monitor evaluates:
   - candidate `ap_event` for `xarm6.place_approach`
   - predicted `ap_state` for `xarm6.positioned`
   - current persistent `ap_state` for `ur5e.positioned`
5. the mutex becomes true, so the start is blocked

This closes the gap where one arm was already at the board but the other arm
had not yet finished its own approach.

## Predicted State APs

The monitor also computes predicted next-state APs during `safety_check`.

This is important because a candidate start may be unsafe even before the new
state is committed.

For a task with:

- `function = place_approach`
- `out_state = positioned`

the monitor adds temporary predicted `ap_state` terms for the candidate check.

This means the online check reasons over:

- currently running `ap_event`
- currently active persistent `ap_state`
- candidate `ap_event`
- predicted next `ap_state`

This is why the system can block an unsafe entry before the second arm actually
finishes entering the board.

## Offline / Online Consistency

The same semantics are used in both places:

- offline validator:
  reconstructs persistent state and predicted next-state APs from the compiled
  plan FSA
- online monitor:
  reconstructs them from runtime events and reported current state

This avoids the mismatch where:

- offline says a plan is safe
- online later blocks it because the runtime monitor knows more than the
  validator

## Why This Generalizes

This strategy is not specific to:

- UR5e
- xArm6
- assembly boards
- the state names `positioned` or `placed`

It only assumes:

- tools expose state transitions
- tools expose enough context to identify the constrained state slice
- the compiler can expand selectors from the tools/state graph

So the same pattern can model:

- machine occupancy
- printer active regions
- conveyor transfer states
- handoff zones
- any other mutex rule where the unsafe condition is state persistence, not just
  action overlap

## Current Files

Main implementation points:

- [`safety_logic.py`](/home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/safety_logic.py)
  compiles selectors into `ap_event` / `ap_state`
- [`base_safety_checker.py`](/home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/base_safety_checker.py)
  maps runtime tasks and states to AP labels
- [`online_safety_monitor.py`](/home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/online_safety_monitor.py)
  tracks persistent state APs and predicted next-state APs
- [`offline_safety_validator.py`](/home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/offline_safety_validator.py)
  uses the same AP semantics for offline checking

## Summary

The mutex strategy is:

- compile spatial/state-slice rules into a mix of `ap_event` and `ap_state`
- use `ap_event` to capture entry
- use `ap_state` to capture persistence
- use predicted `ap_state` during start-time checks
- keep the final safety layer in standard LTLf/DFA form

This preserves the formal temporal-logic pipeline while fixing the main failure
mode of action-only APs for spatial mutex constraints.
