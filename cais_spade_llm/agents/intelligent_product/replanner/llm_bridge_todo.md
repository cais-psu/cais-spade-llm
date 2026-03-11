# TODO: Remaining LLM Bridge Implementation

This document tracks the remaining work after the primitive `preconditions/effects`
layer was added.

Current status:
- Done: bridge-visible controller primitives now expose explicit `preconditions/effects`
- Done: private in-memory primitive catalog builder
- Done: primitive-level bridge snapshot
- Done: semantic validation of `primitive_steps` before approval
- Done: `execute_recovery_macro` start-snapshot check and runtime state syncing

What is still missing is the higher-level bridge orchestration needed for robust
system-level recovery.

## TODO 1. Ground Primitive Parameters

Goal:
- prevent the LLM from inventing arbitrary coordinates
- force bridge params to come from current observed context

Required work:
- [ ] Define a bridge param reference format
  - Recommended shape:
    ```json
    {"context_ref": "parts.LCP.pose.x"}
    ```
- [ ] Add a resolver module that converts `context_ref` values into literal params
- [ ] Extend bridge context to expose all valid grounded values
  - observed part poses
  - named poses
  - known origin/destination locations
  - board target poses
- [ ] Reject unresolved or unknown references before approval
- [ ] Keep the executed primitive calls unchanged after resolution

Files:
- `cais_spade_llm/agents/intelligent_product/replanner/primitive_semantics.py`
- `cais_spade_llm/agents/intelligent_product/replanner/environment_model.py`
- `cais_spade_llm/prompts.py`

Acceptance checks:
- bridge proposal using a valid `context_ref` is accepted
- bridge proposal using an invalid `context_ref` is rejected
- no raw arbitrary Cartesian coordinates are needed in prompts for standard recovery flows

## TODO 2. Support Observation-to-Action Binding

Goal:
- let a bridge macro use observation results from `detect_parts()` or `get_current_pose()`
- enable flows like `detect_parts -> move to detected pose`

Required work:
- [ ] Define step-output storage in the bridge macro format
  - Example:
    ```json
    {"primitive": "detect_parts", "params": {"part_name": "LCP"}, "store_as": "detected_lcp"}
    ```
- [ ] Extend the resolver to allow later params to reference stored step outputs
- [ ] Extend semantic validation to understand observational outputs
- [ ] Decide the v1 allowed shape for multi-result observations
  - Recommended: only allow one filtered part result for `detect_parts(part_name=...)`
- [ ] Add runtime extraction of stored outputs inside `execute_recovery_macro`

Files:
- `cais_spade_llm/agents/intelligent_product/replanner/environment_model.py`
- `cais_spade_llm/agents/intelligent_product/replanner/primitive_semantics.py`
- `cais_spade_llm/agents/resource_agent/robot_agent.py`
- `cais_spade_llm/prompts.py`

Acceptance checks:
- a bridge macro can detect one part and reuse that pose in a later `move_cartesian`
- invalid output references are rejected before execution

## TODO 3. Upgrade From Single Macro to System-Level Recovery Patch

Goal:
- allow the bridge to return an ordered patch across multiple resources, not just one macro

Required work:
- [ ] Change bridge output schema from:
  - one `macro_name` + one `primitive_steps`
- [ ] To:
  - ordered `macro_tasks[]`
- [ ] Each `macro_task` should contain:
  - `resource_jid`
  - `macro_name`
  - `expected_snapshot`
  - `task_metadata`
  - `primitive_steps`
  - `predecessors`
- [ ] Update prompt instructions to request `macro_tasks[]`
- [ ] Update proposal normalization to validate every macro task independently
- [ ] Update plan compilation to generate one `execute_recovery_macro` node per macro task
- [ ] Preserve predecessor ordering across generated nodes

Files:
- `cais_spade_llm/prompts.py`
- `cais_spade_llm/agents/intelligent_product/replanner/environment_model.py`
- `cais_spade_llm/agents/intelligent_product/process_planner.py`

Acceptance checks:
- bridge can propose `UR5e` macro followed by `xArm6` macro, or vice versa
- compiled plan contains ordered `execute_recovery_macro` nodes with predecessors

## TODO 4. Build System-Level Bridge Context

Goal:
- let the LLM reason over the whole disrupted system, not just one robot snapshot

Required work:
- [ ] Extend `_bridge_primitive_context(...)` to gather per-resource snapshots for all robot resources
- [ ] Build a system-level context that includes:
  - per-resource primitive snapshot
  - per-resource primitive catalog
  - part tracker state and observed poses
  - current unfinished goals
  - active safety obligation targets
- [ ] Keep the bridge prompt centered on the stuck resource while still showing global context
- [ ] Avoid scenario-specific derived facts such as deadlock labels or custom bridge booleans

Files:
- `cais_spade_llm/agents/intelligent_product/process_planner.py`
- `cais_spade_llm/prompts.py`

Acceptance checks:
- prompt contains enough cross-resource state for a bridge patch to decide which robot should act first
- no hand-authored recovery operator labels are needed

## TODO 5. Recheck World State After Each Bridge Macro

Goal:
- stop bridge execution as soon as the normal DES planner has a valid continuation again
- fail closed if the world diverges from the expected post-state

Required work:
- [ ] After each executed bridge macro, refresh the affected robot snapshot
- [ ] Re-run perception for touched parts when the macro includes observational or part-manipulation primitives
- [ ] Compare actual post-state with projected post-state
- [ ] If mismatch:
  - stop
  - mark bridge macro as failed
  - either regenerate bridge proposal or escalate to `human_required`
- [ ] If post-state is valid:
  - ask DES again whether a catalog-valid continuation now exists
- [ ] Stop bridge execution immediately when DES can continue normally

Files:
- `cais_spade_llm/agents/intelligent_product/process_planner.py`
- `cais_spade_llm/agents/resource_agent/robot_agent.py`
- possibly controller/perception helpers if a part refresh helper is needed

Acceptance checks:
- bridge executes only as many macros as needed to restore DES solvability
- stale or mismatched world state causes fail-closed behavior

## TODO 6. Finish Bridge Part Tracking

Goal:
- make bridge-generated recovery tasks update the product part tracker correctly

Current gap:
- part tracking only runs when `task_node["params"]["part_name"]` exists
- primitive bridge nodes currently do not always carry enough explicit part identity

Required work:
- [ ] Decide how bridge macros declare touched parts
  - Recommended: add `part_name` or `touched_part` at macro level
- [ ] Pass that field into the compiled `execute_recovery_macro` task params
- [ ] Update product tracking to use node-level `part_transition` together with the macro’s touched part
- [ ] Ensure bridge macros that manipulate one part can update the tracker without relying on shared `tools.json`

Files:
- `cais_spade_llm/agents/intelligent_product/process_planner.py`
- `cais_spade_llm/agents/intelligent_product/product_agent.py`

Acceptance checks:
- a bridge macro that stashes a held part updates the part tracker
- a bridge macro that recovers and places a failed part updates the part tracker

## TODO 7. Strengthen Runtime Macro Validation

Goal:
- ensure the runtime executor is not weaker than prompt-time semantic validation

Required work:
- [ ] Re-validate primitive sequence against the latest runtime snapshot just before execution
- [ ] Add runtime support for resolved param references and stored observation outputs
- [ ] Decide which runtime violations are:
  - hard reject before first step
  - immediate stop during execution
- [ ] Record detailed bridge failure observations:
  - step index
  - primitive
  - expected snapshot
  - actual snapshot
  - projected snapshot if available

Files:
- `cais_spade_llm/agents/resource_agent/robot_agent.py`
- `cais_spade_llm/agents/intelligent_product/replanner/primitive_semantics.py`

Acceptance checks:
- a bridge proposal that was valid at approval time but invalid at runtime is rejected cleanly
- failure logs clearly explain why execution stopped

## TODO 8. End-to-End Bridge Tests

Goal:
- cover the actual recovery loop, not just unit semantics

Required tests:
- [ ] single-resource primitive bridge:
  - DES fails
  - bridge proposes valid primitive macro
  - macro compiles and executes
- [ ] invalid semantic sequence:
  - proposal rejected before approval
- [ ] invalid reference:
  - proposal rejected before approval
- [ ] observation binding:
  - `detect_parts -> move_cartesian` using stored output
- [ ] post-macro recheck:
  - world mismatch forces fail-closed
- [ ] bridge-to-DES handoff:
  - bridge macro restores a valid DES continuation and no extra bridge macro runs
- [ ] multi-resource patch:
  - ordered `macro_tasks[]` compile to multiple bridge nodes in sequence

Suggested scenario test:
- xArm6 fails placing prerequisite part
- failed part becomes observable in UR5e region
- UR5e is holding a later-priority part
- bridge patch clears the held part, recovers the prerequisite, then DES resumes nominal completion

## Recommended Order For Tomorrow

1. Implement `context_ref` grounding
2. Implement step-output storage and observation binding
3. Upgrade proposal schema to ordered `macro_tasks[]`
4. Add part identity plumbing for bridge tracking
5. Add post-macro recheck + DES handoff loop
6. Add end-to-end tests

## Done / Not Done Boundary

Done now:
- primitive semantics metadata
- semantic prompt grounding
- semantic proposal validation
- runtime bridge snapshot syncing

Not done yet:
- grounded param references
- observation binding
- multi-resource patch schema
- post-macro replan loop
- complete bridge part tracking
- end-to-end recovery tests
