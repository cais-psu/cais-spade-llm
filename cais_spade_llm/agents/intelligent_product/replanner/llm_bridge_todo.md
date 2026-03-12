# LLM Bridge Status

This file is the current implementation/status snapshot for the primitive
LLM bridge. It replaces the old raw TODO list.

Design reference:
- `llm_bridge_construction.md` describes the intended bridge architecture.

Current runtime scope:
- DES remains the primary recovery planner.
- The bridge is invoked only when DES cannot find a modeled continuation.
- Approved bridge proposals compile into ordered `execute_recovery_macro`
  tasks and run inside the normal runtime recovery session.

## Implemented

- Grounded `context_ref` parameter resolution exists in
  `primitive_semantics.py` and is covered by `test/test_primitive_semantics.py`.
- `store_as` and later step-output references are implemented for
  `detect_parts(part_name=...)` and `get_current_pose()`, including validation
  and runtime extraction.
- Ordered `macro_tasks[]` are implemented. The bridge no longer depends on a
  single top-level macro shape.
- Whole-system bridge context is implemented. The prompt includes per-resource
  bridge snapshots, per-resource primitive catalogs, pending tasks, part
  tracker state, active obligation targets, and grounded context paths.
- Bridge proposals are normalized and semantically validated before approval.
  Each macro task is checked against the projected bridge snapshot and can be
  rejected before execution.
- Bridge part tracking is implemented. Approved macro tasks can carry
  `part_name`, `task_metadata.part_transition`, and `projected_part_entry`, and
  Product tracking applies those updates after completion.
- Runtime semantic validation is implemented in `RobotAgent.execute_recovery_macro`.
  The executor rechecks `expected_start_state`, `expected_snapshot`, primitive
  semantics, and runtime param refs before running the sequence.
- Post-macro DES recheck / bridge tail trimming is implemented. After each
  completed bridge macro, Product refreshes the resource snapshot, compares it
  against the projected post-state, and either:
  - hands control back to DES immediately,
  - trims the remaining bridge tail and validates the repaired plan, or
  - continues the approved bridge tail if DES still has no continuation yet.
- Temporary dashboard bridge-debug visibility is implemented in Dashboard >
  Replan / Recovery > `LLM Bridge Debug (temporary)`. The panel exposes the
  bridge request, prompt grounding, prompt text, raw model output, normalized
  proposal, compiled bridge tasks, and active bridge execution state.

## Working But Fragile

- Prompt compliance with `task_metadata` and `task_params` is still brittle.
  The system can validate and reject bad bridge output, but the LLM does not
  yet reliably emit compilable task context for recovery macros.
- Runtime grounding quality for `recovery_required` situations still depends on
  how well the prompt captures the disrupted world state. The architecture now
  supports whole-system recovery, but the prompt examples are not yet strong
  enough to make those recoveries reliable.
- World/perception refresh is only as good as the live bridge snapshot and part
  tracker. The fail-closed checks are in place, but successful handoff still
  depends on accurate runtime state refresh and part identity plumbing.
- Bridge observability is good enough for debugging, but not yet a stable
  operator-facing product surface. The dashboard panel is intentionally marked
  temporary.

## Known Broken Behavior

Concrete failing scenario from the latest `xarm6` LCP-slippage run on
March 12, 2026:

1. Nominal execution advanced far enough for `xarm6` to reach
   `REQ_2_T4 -> place_insert`.
2. The injected LCP slippage path fired correctly:
   - `place_insert` failed,
   - the LCP part was dropped into the UR5e region,
   - `xarm6` transitioned to `current_state='recovery_required'`.
3. CCA correctly classified the result as `inevitable_violation` and sent the
   online replan request to Product.
4. Product correctly entered DES runtime recovery and the planner correctly
   reached bridge fallback:
   - `DES replanning triggered`
   - `DES found no modeled continuation; requesting bridge proposal`
5. The bridge call path itself is therefore working:
   - `CentralControllerAgent._handle_runtime_event()`
   - `ProductAgent._handle_runtime_des_replan_request()`
   - `ProductAgent._run_des_runtime_recovery_attempt()`
   - `ProcessPlanner.replan_with_feedback_des()`
   - `ProcessPlanner._request_bridge_proposal()`
   - `environment_model.llm_explore_states_and_events()`
6. The current failure is at proposal normalization / compilability, not at
   bridge entry. The latest rejected output failed with:

   `Bridge macro_task 1 missing required task_params keys: ['destination']`

7. Because the bridge output was invalid, runtime recovery ended in
   `human_required`.
8. Duplicate replan requests are still emitted after the same runtime failure.
   Product suppresses the duplicate work with:

   `Runtime recovery already active for REQ_2_T4; ignoring duplicate replan request.`

What this means:
- The recovery pipeline from runtime failure into bridge fallback is now in place.
- The current blocker is that the LLM still does not reliably emit a
  normalization-safe bridge proposal for the LCP-slippage case.

## Next Work

1. Make bridge outputs reliably satisfy `required_context_keys` and
   `part_transition` param requirements so proposals do not fail during
   normalization.
2. Add a stronger bridge prompt/example for assembly recovery that uses the
   implemented canonical task param names such as `destination_location`.
3. Add end-to-end tests for the LCP-slippage bridge path:
   - `place_insert` failure,
   - DES fallback,
   - bridge proposal normalization,
   - approval,
   - execution,
   - DES handoff.
4. Investigate duplicate failure / replan notifications after a single runtime
   failure so the bridge path only triggers once per event.
5. Tighten post-bridge runtime validation where perception refresh is required,
   especially when the bridge proposal depends on observation-heavy recovery.
