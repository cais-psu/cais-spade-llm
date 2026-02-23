# TODO: Use DES Bidding for Offline Initial Planning

## Background

Currently the offline planning pipeline is:
```
NL → (LLM) → Structured Requirements → (LLM) → DAG → (FSA) → Verify → (LLM) → Fix violations
```
The LLM is called **3 times**: parsing, planning, and repairing.

## Proposed Change

Replace the second LLM call (Requirements → DAG) with DES bidding:
```
NL → (LLM) → Structured Requirements → (DES) → compute_bid per resource → compile_environment_model → BFS → DAG
```

## Why

- Plans are correct by construction (no hallucinated transitions)
- Removes one expensive LLM call
- Resource assignment is automatic via `function_owner_agent` in `tools.json`
- FSA verification still needed for ordering safety constraints (e.g. "SG before MCP"), but not for state transitions

## What Still Needs LLM

- Step 1: `NL → Structured Requirements` (DES cannot read NL)
- Step 3: LLM repair if FSA detects ordering violations

## Implementation Hints

- Translate structured requirements into `P_id` + `goal_state` + initial `x_c` (requirements parser already produces fields close to this)
- Call `compute_bid()` per resource agent using its tools from `tools.json`
- Call `compile_environment_model(bids)` to build M_e
- Call `plan_on_environment_model(M_e, x_c, P_id, goal_state)` to get event path
- Convert path events to DAG tasks (similar to what `replan_with_feedback_des` already does)

## Files To Modify

- `cais_spade_llm/agents/intelligent_product/process_planner.py` — add offline DES planning path
- `cais_spade_llm/agents/intelligent_product/replanner/resource_bidding.py` — reuse `compute_bid`
- `cais_spade_llm/agents/intelligent_product/replanner/environment_model.py` — reuse `compile_environment_model` + `plan_on_environment_model`
