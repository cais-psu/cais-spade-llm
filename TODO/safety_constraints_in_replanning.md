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

Mutex constraints ("no simultaneous pick_part") require the BFS to reason about what **all** resources are doing at the same step simultaneously — the current sequential event BFS doesn't model this. This may require a parallel-step BFS model.
