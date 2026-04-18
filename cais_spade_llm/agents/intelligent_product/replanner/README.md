# Replanner Recovery Flow

This folder contains the runtime recovery logic used when the product plan can no longer continue safely.

Runtime replanning is DES-driven. There is no separate top-level runtime
`replan_mode`; the LLM is used only inside the DES bridge fallback. The older
offline LLM plan-repair path remains separate.

## What Was Implemented

The recovery path was changed from a generic suffix-search plus synthetic LLM tool insertion into a safety-driven workflow:

1. The Central Controller Agent (CCA) now derives formal `obligation_targets` from the active structured safety rules.
2. DES recovery uses those obligation targets to search for real catalog tools that satisfy the required APs.
3. `"any"` remains only a tool applicability wildcard for `in_state`; it is not used to guess the missing recovery action.
4. If DES cannot find a catalog-valid obligation-satisfying path, the LLM bridge is used only to propose a high-level recovery macro.
5. Bridge proposals are approval-gated. They are not executed directly as invented tool names.
6. On approval, the proposal is compiled into ordered `execute_recovery_macro` runtime recovery tasks, and execution resumes automatically.
7. On rejection, operator feedback is recorded and immediately fed back into bridge regeneration.
8. If no modeled path exists and the bridge cannot produce a compilable proposal, runtime recovery ends in `human_required`.

## Main Files

- `des_search/`
  DES forward search and resource bidding over modeled resource states and tool
  transitions.
- `llm_bridge/`
  LLM bridge parsing, primitive semantics, bridge safety, deterministic bridge
  compilation, and the multi-turn DES-guided bridge loop.
- `classical_vs_llm_bridge.md`
  Design note on why the current bridge layer is under-modeled for classical planning alone and why `LLM + ReAct + DES` fits the existing task/primitive split.
- `tss_based_llm_bridge_v4.md`
  Design-intent note for the new bridge: current prepare-trace/`llm_input`
  state and the target validated proposal flow.
- `../process_planner.py`
  Top-level DES recovery orchestration, obligation-driven DES selection, and
  delegation into the extracted bridge replanner module.
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
- the proposal contains one `primary_obligation` and ordered `macro_tasks[]`
- each `macro_task` chooses a `resource_jid`, carries node-level task metadata,
  and executes controller-level `primitive_steps`
- the proposal is normalized and semantically validated before approval

This means the bridge is allowed to invent a recovery macro structure, but it
is not allowed to invent a new runtime execution surface outside the approved
controller primitive set.

### 5. Approval and resume

Runtime recovery stores:

- `bridge_proposal`
- `bridge_approval_state`
- `bridge_feedback_history`

When the operator approves:

1. `ProductAgent.approve_runtime_bridge_proposal()` compiles the proposal into
   ordered `execute_recovery_macro` recovery task nodes.
2. Those nodes are inserted into the runtime plan as a serial bridge sequence.
3. The plan FSA is rebuilt and revalidated with the CCA.
4. Execution resumes automatically after approval.
5. After each completed bridge macro, Product refreshes the runtime snapshot and
   either hands control back to DES, trims the remaining bridge tail, or
   continues the approved bridge sequence.

Implementation detail:

- Approved bridge nodes execute directly through `RobotAgent.execute_recovery_macro`.
- The bridge remains approval-gated and isolated from the shared DES tools
  catalog.

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
- the bridge returns nothing, invalid JSON, a wrong-resource proposal, or a
  normalized proposal that still fails validation or compilation
- an approved bridge macro fails runtime semantic validation or diverges from
  its approved projected post-state
- the approved bridge tail finishes but DES still has no valid continuation

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
7. If the repaired FSA still contains the current live runtime state, the CCA preserves that exact live monitor state rather than rebuilding only from a coarse completed/running snapshot.
8. The repaired runtime monitor remains `reactive`, so the modeled recovery task is allowed to run.

This is what makes the successful log sequence possible:

- `Modeled obligation recovery matched rule ... with 1 step(s)`
- `Applied fix ... move_home`
- `Plan FSA Validation: OK ... supervisor_mode=reactive`
- `SUPERVISOR DEFERRED ... mode=reactive`
- `ACK ... status='completed'`

### Repaired-FSA synchronization

Revalidating a repaired runtime plan is not just an offline check. The repaired FSA must stay synchronized with the execution that is already in flight.

The current behavior is:

1. Product recompiles the repaired FSA and sends `plan_safety_check`.
2. CCA reuses the existing plan monitor if the FSA is unchanged.
3. If the FSA changed, CCA creates a fresh `OnlineFsaMonitor` for the repaired FSA.
4. If the old live monitor's exact `current_state` still exists in the repaired FSA, that state is preserved directly.
5. Otherwise, CCA falls back to rebuilding progress from runtime context plus any prior completed/running/failed task knowledge that can still be replayed safely.

This matters when recovery is inserted while a task is already running. For example, if `REQ_1_T3.start` happened before the repair but `REQ_1_T3.done` arrives after the repaired FSA is installed, the repaired monitor must still know that `REQ_1_T3` is running. Otherwise the later `.done` event lands on the wrong state and downstream tasks such as `REQ_1_T4` can look falsely "not enabled".

### Event guard during resynchronization

`OnlineFsaMonitor` now treats unmatched runtime completion events conservatively.

- If a `.done` or `.fail` event is not enabled from the current repaired FSA state, the monitor leaves the state unchanged.
- An unmatched `.done` event is not allowed to silently mark the task as completed.

This prevents stale or out-of-order runtime events from corrupting the repaired execution prefix after FSA replacement.

### Regressions that were fixed for this path

- Startup from a verified bundle was incorrectly drifting into `preventive` mode.
- Repaired-plan validation was resetting the plan FSA monitor to the initial state instead of preserving the completed runtime prefix.
- Repaired-plan validation could lose an in-flight start event when swapping monitors, which later made the next real task appear "not enabled" in the repaired FSA.
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

`des_search/resource_bidding.py` now supports tools that do not manipulate parts or locations, such as `move_home`.

This matters for suffix recovery cases where:

- all parts are already assembled
- the only missing obligation is a robot state cleanup action

## Current UI/Operator Semantics

The dashboard recovery panel now shows bridge proposals when runtime recovery enters `llm_bridge`.

The operator can:

- approve and resume
- reject and regenerate with feedback

The UI displays:

- proposal macro tasks
- target resource(s)
- description and rationale
- compiled bridge tasks
- temporary LLM bridge debug trace for prompt/input/output inspection

## Verified Behavior

The implementation was verified with targeted checks for the bundle `case1_move_home_not_included`:

- the CCA obligation target resolves to the real `move_home` tool for the affected resource
- DES bidding from `resource_state='placed'` finds a modeled path ending in `move_home`
- repaired-plan validation preserves runtime progress when swapping in the updated FSA
- verified-bundle runtime supervision stays `reactive`, which allows the modeled recovery task to execute

Syntax checks were also run with `py_compile` and `git diff --check`.

## Forward Simulation and DFA-Guided Recovery

Two generalized mechanisms were added to make safety-block recovery smarter and avoid unnecessary LLM bridge fallbacks.

### Forward simulation: wait vs. replan

When a safety violation blocks a task, the CCA no longer immediately triggers a replan. Instead, it forward-simulates the composite plan FSA to determine if the violation will naturally resolve.

**Method:** `CentralControllerAgent._will_violation_resolve()`

**Algorithm:**

1. BFS through all reachable FSA transitions from the current composite state, excluding the blocked task's own transitions.
2. At each transition (`.start` or `.done`), simulate AP changes:
   - `.start` events add event APs to the running set.
   - `.done` events remove event APs, update resource state APs (from `out_state`), and advance the safety DFA.
3. At each reached state, test whether the blocked task would now be allowed (non-mutating safety DFA check).
4. If any reachable interleaving clears the violation, the CCA waits instead of replanning.

The existing `_retry_blocked_tasks()` mechanism handles the actual unblocking when the real safety DFA reaches a safe state.

**Example — mutex with `move_home` in the plan:**

If ur5e's plan includes `place_approach → place_insert → move_home`, the forward simulation shows that after `move_home.done`, ur5e exits the assembly zone and the mutex clears. CCA logs `"Safety block on task=... will resolve naturally"` and skips the replan.

**Example — mutex without `move_home` in the plan:**

If ur5e's plan ends at `place_insert`, the simulation shows ur5e remains in `placed` state. The mutex persists across all interleavings, so a replan is needed.

This forward simulation only answers whether the current modeled plan can resolve the block naturally. It does not synthesize new recovery steps.

### DFA-guided recovery tool discovery

When a replan IS needed, the obligation target builder now uses DFA-guided recovery as a fallback when direct AP matching produces no candidate tools.

**Method:** `CentralControllerAgent._find_dfa_recovery_tools()`

**Algorithm:**

1. Get the safety DFA for the violated rule and its current state.
2. Enumerate subsets of currently active AP symbols (the AP symbol set per rule is small — typically 2–8 symbols, so 2^N is feasible).
3. For each subset removed, check if the DFA transitions to a safe (non-violation) state.
4. Map the "removed APs" to state AP descriptors to identify which resource states need to be exited.
5. Find tools in the catalog whose `in_state` matches a violating state and `out_state` does NOT match any violating state.

This is fully general — it works for any constraint type (mutex, ordering, liveness) because it reasons about DFA transitions rather than rule structure. No tool names, rule IDs, or robot names are hardcoded.

**Example:**

For mutex rule `SAFE_1` with APs `{place_approach, positioned, placed}` for ur5e:

- Removing `ap_state/.../ur5e/placed/...` leads to a safe DFA state.
- The system finds `move_home` because `in_state=placed` (violating) → `out_state=idle` (not violating).
- DES receives `candidate_tools=[move_home]` with a valid signature, projects the full modeled suffix snapshot for the target resource, finds the catalog-valid recovery path, and avoids the LLM bridge.

The suffix projection step matters. Obligation recovery now searches from a fully consistent modeled resource snapshot:

- `resource_state`
- `current_part`
- `current_location`
- `part_states`
- `part_locations`

That avoids hybrid search states such as a projected `resource_state='placed'` combined with a live `current_part='MCP'`, which can incorrectly hide pure robot-state recovery actions and force an unnecessary bridge fallback.

If the live product snapshot is stale but the next modeled suffix task provides enough semantic information to reconcile it, the planner aligns the modeled snapshot with that task's catalog preconditions before projecting the suffix. This uses only generic catalog fields such as:

- `part_in_state`
- `context_mapping.location_type`
- `context_mapping.location_param`
- `params.part_name`

No tool-name or rule-name special cases are needed.

### Residual recovery insertion

When suffix projection succeeds, the DES recovery path is computed from the end of the already-modeled suffix, not from the current live start state. That means the inserted repair is only the residual tail that is still missing.

**Example:**

- Existing modeled suffix: `ur5e.place_approach -> ur5e.place_insert`
- Projected end state after that suffix: `resource_state='placed'`
- Recovery candidate found by DES: `move_home`

The inserted repair is:

- `ur5e.move_home`

and not:

- `ur5e.place_approach -> ur5e.place_insert -> ur5e.move_home`

### Recovery boundary gating

Recovery tasks are also spliced into the graph at the blocking boundary.

- The recovery path is anchored after the target resource's projected pending suffix.
- The blocked task that triggered replanning is rewritten to depend on the inserted recovery tail.

This preserves the intended ordering in the plan FSA. In the mutex example, the repaired shape is:

- `ur5e.place_approach`
- `ur5e.place_insert`
- `ur5e.move_home`
- `xarm6.place_approach`

If a resource still has a pending/running suffix but that suffix cannot be projected consistently, the planner does not fall back to a live-state DES search for that same resource. Doing so would replay already-modeled suffix tasks as duplicate "recovery" work. In that case the planner continues searching other modeled options and only falls through to bridge if no catalog-valid residual recovery exists.

### Transient blocked-task retry behavior

When a blocked task becomes safe later, the system now performs a true retry as a new task attempt.

This is important because a ResourceAgent block is terminal for that specific attempt:

- the resource already sent `status='blocked'`
- that attempt will never spontaneously resume
- the resource will only ask CCA again if Product dispatches the task again

The runtime behavior is now:

1. CCA keeps a queue of temporarily blocked tasks.
2. After each real runtime finish/fail event, CCA re-evaluates those blocked tasks against the current safety monitor and online supervisor state.
3. If a blocked task is now safe, CCA clears its internal blocked-queue entry and notifies Product that the task is `retry_ready`.
4. Product changes the blocked DAG node back to `pending`.
5. The normal plan executor dispatches the task again.
6. The ResourceAgent sends a fresh `safety_check`.
7. CCA returns a fresh `allow` or `block` for this new attempt.

This is generalized behavior. It does not special-case `xarm6`, `move_home`, or mutex rules. The decision is always based on the current live AP/supervisor state for the exact blocked task.

**Example:**

- `xarm6.place_approach` is blocked because `ur5e` is still occupying the assembly-zone slice.
- DES inserts `ur5e.move_home` and gates the blocked `xarm6` task after it.
- After `ur5e.move_home.done`, CCA re-checks the previously blocked `xarm6.place_approach`.
- Because the safety DFA no longer predicts overlap, CCA marks the task retry-ready.
- Product requeues `REQ_2_T3` to `pending`, dispatches it again, and the resource performs a fresh `safety_check`.

### Integration

The DFA-guided fallback is invoked in `_build_obligation_targets()`: if `_candidate_tools_for_obligation()` returns an empty list (no direct AP match), `_find_dfa_recovery_tools()` is called as a second pass.

The forward simulation is invoked in `_SafetyCheckInbox._handle_safety_check()`: after blocking the task, the CCA calls `_will_violation_resolve()` and only sends a replan request if the violation cannot naturally resolve.

## Known Design Boundary

> **Superseded.** The original constraint below has been removed. See
> [`llm_bridge_construction.md`](llm_bridge_construction.md) for the
> current design and [`llm_bridge_todo.md`](llm_bridge_todo.md) for the
> current implementation status / known issues.

Bridge proposals now embed **controller-level primitive sequences** that
execute through `RobotAgent.execute_recovery_macro`. This allows the LLM
bridge to compose novel recovery actions from low-level motion, gripper,
and perception primitives without being limited to existing catalog
functions.

The shared `tools.json` catalog and DES planning surface remain
unchanged — only the bridge path gains access to controller primitives
via a private in-memory catalog.
