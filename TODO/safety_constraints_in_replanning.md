# TODO: Wire DFA Safety Constraints into DES Replanning BFS

## Background

Safety constraints are already defined and compiled into DFAs in:
- `cais_spade_llm/safety/cca_safety_logic.json` — structured rules with LTLf formulas and atomic propositions
- `cais_spade_llm/safety/SAFE_N_dfa.dot` — pre-compiled DFAs (one per safety rule)

Example existing constraint:
- **SAFE_1** (ordering): `(!ap001) U ap002` = "SG must be placed before MCP" 
- **case2-mutual** (mutex): "xarm6 and ur5e should not pick_parts at the same time"

The current `plan_on_environment_model()` BFS in `environment_model.py` ignores these completely.

## Proposed Change

Augment the BFS state to include DFA states, and prune transitions that violate safety:

```python
# BFS node becomes:
(M_e_state_key, dfa_1_state, dfa_2_state, ...)

# For each event applied:
# 1. Evaluate which atomic propositions it satisfies (part, function, resource)
# 2. Advance each DFA state
# 3. Prune if any DFA enters a reject/trap state
# 4. Only accept goal states where all DFAs are in accepting states
```

## Implementation Steps

1. **Parse DFA `.dot` files** → dict `{state: {label_expr: next_state}}`, mark accepting vs trap states
2. **Map events to APs** — given an event dict `{function_name, ra_jid, part_name}`, evaluate boolean for each AP label (e.g. `ap/assembly/sg/any/place_part/any`)
3. **Augment BFS** in `plan_on_environment_model()` — add DFA state tuple to BFS node
4. **Handle mutex constraints** — requires tracking simultaneous resource activity (may need to restructure BFS into parallel-step model instead of sequential events)

## Files to Modify

- `environment_model.py` — augment `plan_on_environment_model()` and `_find_start_state()`
- `process_planner.py` — load `cca_safety_logic.json` + DFA `.dot` files and pass into BFS

## Key Complexity

Mutex constraints ("no simultaneous pick_grasp") require the BFS to reason about what **all** resources are doing at the same step simultaneously — the current sequential event BFS doesn't model this. This may require a parallel-step BFS model.

---

# TODO: Validation Pipeline for LLM Bridge Events (Layer 2)

## Background

When Layer 1 (compiled DES BFS) finds no recovery path, the system invokes `llm_explore_states_and_events()` to generate synthetic bridge events that extend the environment model M_e. These LLM-generated events have **no formal guarantees** — the LLM may hallucinate infeasible transitions, violate safety constraints, or reference states/locations that don't exist.

Current code (`environment_model.py:119-156`) injects LLM output directly into M_e with no validation.

## Validation Tiers (Ordered by Priority)

### Tier 1: DFA Safety Pre-check (HIGHEST PRIORITY)

Apply existing safety DFAs to the bridge event sequence **before** injecting into M_e.

```python
# For each bridge event in sequence:
#   1. Compute atomic propositions from event metadata (function_name, part_name, ra_jid)
#   2. Advance each DFA: q_next = delta(q_current, AP_set)
#   3. If any DFA reaches violation_state → REJECT event sequence
```

This reuses the same DFA infrastructure from offline validation (`offline_safety_validator.py`). It guarantees that LLM-generated recovery plans satisfy the same safety specifications as Layer 1 plans.

**Files to modify:**
- `environment_model.py` — add validation before bridge event injection
- Reuse DFA parsing/evaluation from `offline_safety_validator.py`

### Tier 2: Schema and DES Consistency (FREE — milliseconds)

Validate structural correctness of LLM output before any further processing:

- **Schema validation**: `function_name` exists in tools catalog or is a recognizable composition of primitives. All required parameters present and correctly typed.
- **State chain consistency**: Bridge event N's `out_state` must equal event N+1's `in_state`. First event's `in_state` must match the actual stuck state.
- **Workspace reachability**: Target locations within `workspace_boundaries` of the assigned resource. Handoff locations in `staging_areas` reachable by both involved resources.
- **Part existence and state consistency**: Referenced parts exist in part tracker. Assumed part state/location matches what part tracker reports.

**Files to modify:**
- `environment_model.py` — add validation in `llm_explore_states_and_events()` before returning

### Tier 3: Re-query Loop with Rejection Feedback

When any validation tier rejects bridge events, feed the specific rejection reason back to the LLM:

```
Your proposed recovery event was rejected:
  Event: pick_part_from_floor(sg, location=floor_zone_3)
  Rejection: location floor_zone_3 is outside ur5e workspace boundaries.
  Workspace boundaries for ur5e: [assembly-board-v1, staging-area-1, ...]

  Generate an alternative recovery that respects these constraints.
```

Bound the loop: **max 2-3 re-queries**, then escalate to Layer 3 (human intervention).

**Files to modify:**
- `environment_model.py` — wrap `llm_explore_states_and_events()` in retry loop
- `prompts.py` — add rejection feedback prompt template

### Tier 4: Runtime Monitoring with Checkpointing

For bridge events that pass all pre-execution checks but still carry uncertainty (the LLM's world model may be wrong):

- **Vision verification**: Before picking a displaced part, confirm part is visible at expected location via camera.
- **Force/torque monitoring**: During pick, if force exceeds threshold, abort — part may not be there or is jammed.
- **State checkpointing**: After each bridge event executes, verify the expected post-state was actually achieved before proceeding to next event. If mismatch, re-enter replanning from actual state.

This catches cases where formal checks pass but the physical world doesn't match the LLM's assumptions.

**Files to modify:**
- `resource_agent.py` — add post-action state verification for bridge-originated tasks
- `central_controller_agent.py` — handle checkpoint failure as new replan trigger

### Tier 5: Kinematic/Physics Simulation (Optional)

Query ROS MoveIt motion planner for bridge events involving unusual locations:

- **Inverse kinematics**: Can the robot reach the target pose?
- **Collision checking**: Does a collision-free trajectory exist?
- **Grasp feasibility**: Can the gripper access the part at the displaced location?

Cost: 1-5 seconds per event. Acceptable since Layer 2 is already the slow path (LLM call takes 2-10s).

**Files to modify:**
- New validation module or integration with ROS MoveIt planning scene

### Tier 6: LLM Ensemble / Consistency Check (Optional — for evaluation)

Query the LLM multiple times (or multiple models) with the same stuck state. If responses converge on the same recovery strategy, confidence is higher. If they diverge, the situation is ambiguous — escalate to human.

Cost: 2-3x LLM latency (parallelizable). Primarily useful for paper evaluation section to provide statistical reliability argument.

## Implementation Order

| Step | Tier | Effort | Impact |
|------|------|--------|--------|
| 1 | Tier 2 (schema + DES consistency) | Low | Catches obvious hallucinations |
| 2 | Tier 1 (DFA safety pre-check) | Medium | Formal safety guarantee for Layer 2 |
| 3 | Tier 3 (re-query loop) | Low | Improves LLM success rate |
| 4 | Tier 4 (runtime checkpointing) | Medium | Catches world-model errors |
| 5 | Tier 5 (simulation) | High | Physics feasibility |
| 6 | Tier 6 (ensemble) | Low | Evaluation metric |
