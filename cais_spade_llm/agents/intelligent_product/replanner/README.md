# Replanner Recovery Flow

This folder contains the runtime recovery logic used when the product plan can no longer continue safely.

## What Was Implemented

The recovery path was changed from a generic suffix-search plus synthetic LLM tool insertion into a safety-driven workflow:

1. The Central Controller Agent (CCA) now derives formal `obligation_targets` from the active structured safety rules.
2. DES recovery uses those obligation targets to search for real catalog tools that satisfy the required APs.
3. `"any"` remains only a tool applicability wildcard for `in_state`; it is not used to guess the missing recovery action.
4. If DES cannot find a catalog-valid obligation-satisfying path, the LLM bridge is used only to propose a high-level recovery macro.
5. Bridge proposals are approval-gated. They are not executed directly as invented tool names.
6. On approval, the proposal is compiled into normal runtime recovery tasks using existing catalog functions, and execution resumes automatically.
7. On rejection, operator feedback is recorded and immediately fed back into bridge regeneration.
8. If no modeled path exists and the bridge cannot produce a compilable proposal, runtime recovery ends in `human_required`.

## Main Files

- `resource_bidding.py`
  DES forward search over modeled resource states and tool transitions.
- `environment_model.py`
  LLM bridge proposal parsing and normalization.
- `../process_planner.py`
  Recovery orchestration, obligation-driven DES selection, and bridge macro materialization.
- `../../central_controller/central_controller_agent.py`
  Builds formal `obligation_targets` from structured safety rules and current diagnosis.
- `../product_agent.py`
  Owns runtime recovery session state, approval/rejection actions, and validation/resume flow.

## Runtime Flow

### 1. Safety diagnosis

When the CCA detects `violated` or `inevitable_violation`, it includes `safety_ctx.obligation_targets` in the replan request.

Each obligation target contains:

- `rule_id`
- `resource_jid`
- `required_event_aps`
- `required_state_aps`
- `candidate_tools`
- `required_in_states`
- `required_out_states`
- `current_resource_state`

`candidate_tools` is derived generically from rule APs and the shared tools catalog. There is no hard-coded `move_home` or `place_insert` special case in the selector.

### 2. Obligation-driven DES search

`ProcessPlanner.replan_with_feedback_des()` collects the obligation targets and tries modeled recovery first.

For each target:

- DES computes the current modeled resource state.
- It extracts the catalog tool signatures from `candidate_tools`.
- `compute_bid(..., goal_event_signatures=...)` searches for the shortest modeled path whose final event matches one of those exact catalog signatures.

This is why a rule such as `place_insert -> F move_home` now resolves to the real `move_home` tool instead of a generic suffix heuristic.

### 3. Non-hard-coded AP/state matching

The recovery selector is driven by formal APs:

- Event APs match catalog functions.
- State APs match catalog `out_state`.
- Resource matching is done through the rule AP resource token and catalog owner agent.
- Process matching is checked against the catalog `process`.

This supports both:

- Event obligations such as `move_home after place_insert`
- State-sensitive obligations where the satisfying tool must have the correct modeled transition

### 4. Bridge fallback

If no catalog-valid modeled path exists for the active obligation:

- the planner calls the LLM bridge
- the bridge must return exactly one proposal object
- the outer `function_name` may be new
- `macro_steps` must use exact existing catalog functions for the same resource

This means the bridge is allowed to invent a new high-level recovery macro name, but it is not allowed to invent a new low-level executable tool.

### 5. Approval and resume

Runtime recovery stores:

- `bridge_proposal`
- `bridge_approval_state`
- `bridge_feedback_history`

When the operator approves:

1. `ProductAgent.approve_runtime_bridge_proposal()` compiles the proposal into ordinary recovery task nodes.
2. Those nodes are inserted into the runtime plan as normal tasks.
3. The plan FSA is rebuilt and revalidated with the CCA.
4. Execution resumes automatically after approval because the approved macro has been materialized into executable catalog-backed tasks.

Implementation detail:

- The current runtime executor does not execute bridge macros directly.
- Instead, approval materializes the macro into standard recovery task nodes.
- This preserves existing resource-agent execution semantics and avoids adding a second execution engine.

### 6. Rejection and regeneration

When the operator rejects a bridge proposal:

1. rejection feedback is appended to `bridge_feedback_history`
2. the active proposal is cleared
3. DES recovery is re-entered with that feedback
4. the bridge regenerates immediately within the same runtime recovery session

No manual retry is required after rejection.

### 7. Terminal failure

Recovery transitions to `human_required` when:

- DES finds no catalog-valid obligation path, and
- the bridge returns nothing, invalid JSON, a wrong-resource proposal, or macro steps that do not compile to exact existing catalog functions

## Verified-Bundle Runtime Recovery

The verified bundle itself remains immutable under `user_verified_plan/bundles/...`.

At startup:

- the verified bundle plan and FSA are copied into `cais_spade_llm/monitor/plan/...`
- runtime recovery edits only those live monitor artifacts
- a new run overwrites `monitor/plan` again from the verified bundle

This means runtime recovery is always working on the live execution copy, not patching the stored verified bundle.

### Why the `move_home` recovery now executes

The working path for the bundle `case1_move_home_not_included` is:

1. Startup loads the verified bundle and skips offline revalidation.
2. Because the run started from a verified bundle, runtime supervision is forced to stay in `reactive` mode.
3. After `place_insert.done`, the CCA opens the formal obligation and sends `obligation_targets`.
4. DES matches the obligation target to the real catalog tool `move_home` and inserts a runtime recovery task.
5. The product recompiles the repaired global FSA and sends it back to the CCA.
6. The CCA restores runtime execution progress into the repaired FSA monitor instead of resetting to `x0`.
7. The repaired runtime monitor remains `reactive`, so the modeled recovery task is allowed to run.

This is what makes the successful log sequence possible:

- `Modeled obligation recovery matched rule ... with 1 step(s)`
- `Applied fix ... move_home`
- `Plan FSA Validation: OK ... supervisor_mode=reactive`
- `SUPERVISOR DEFERRED ... mode=reactive`
- `ACK ... status='completed'`

### Regressions that were fixed for this path

- Startup from a verified bundle was incorrectly drifting into `preventive` mode.
- Repaired-plan validation was resetting the plan FSA monitor to the initial state instead of preserving the completed runtime prefix.
- The DES planner was previously reading a robot-specific coordination key instead of a generic resource-state view.
- Prompt-file changes were incorrectly deactivating otherwise valid verified bundles before startup.

## Tool Modeling Notes

### `"any"` semantics

`in_state: "any"` means:

- the tool is applicable from any current resource state

It does not mean:

- the tool should be preferred
- the tool satisfies the obligation by default
- the tool is the recovery goal

Goal selection comes from the formal obligation target, not from wildcard applicability.

### Pure robot-state actions

`resource_bidding.py` now supports tools that do not manipulate parts or locations, such as `move_home`.

This matters for suffix recovery cases where:

- all parts are already assembled
- the only missing obligation is a robot state cleanup action

## Current UI/Operator Semantics

The dashboard recovery panel now shows bridge proposals when runtime recovery enters `llm_bridge`.

The operator can:

- approve and resume
- reject and regenerate with feedback

The UI displays:

- proposal macro name
- target resource
- description
- rationale
- compiled macro steps

## Verified Behavior

The implementation was verified with targeted checks for the bundle `case1_move_home_not_included`:

- the CCA obligation target resolves to the real `move_home` tool for the affected resource
- DES bidding from `resource_state='placed'` finds a modeled path ending in `move_home`
- repaired-plan validation preserves runtime progress when swapping in the updated FSA
- verified-bundle runtime supervision stays `reactive`, which allows the modeled recovery task to execute

Syntax checks were also run with `py_compile` and `git diff --check`.

## Known Design Boundary

Bridge proposals currently compile into existing catalog-backed task nodes. They do not create new low-level robot controller capabilities at runtime.

That is intentional. It keeps runtime execution inside the existing resource-agent tool model while still allowing the LLM bridge to propose new high-level recovery macros for operator approval.
