# TODO: Validate LLM Bridge Output Before Injection

## Background

When `llm_explore_states_and_events()` generates recovery steps, they are currently injected into `M_e` with minimal validation (only JSON format and list type check). The LLM can hallucinate physically impossible actions, out-of-bounds coordinates, or safety-violating sequences.

## Validation Layers (in order of complexity)

### Layer 1: Schema Validation (easy)
Reject bridge tools missing required keys:
```python
required = {"function_name", "in_state", "out_state", "part_effect"}
for tool in bridge_tools:
    if not required.issubset(tool.keys()):
        reject(tool)
```

### Layer 2: Coordinate Bounds Checking (medium)
Validate XYZ coordinates in `params` fall within the resource's reachable workspace:
```python
for param, val in tool["params"].items():
    if param in ("x", "y", "z") and not workspace_bounds.contains(val):
        reject(tool)
```
Workspace bounds can be derived from `resource_infos` staging areas and reachability.

### Layer 3: State Compatibility (medium)
Validate that `out_state` connects back to a known `in_state` in the tools catalog, ensuring the second BFS pass can actually continue from where the bridge lands:
```python
valid_in_states = {t["in_state"] for t in tools_catalog}
if tool["out_state"] not in valid_in_states:
    reject(tool)
```

### Layer 4: DFA Safety Check (depends on `safety_constraints_in_replanning.md` TODO)
Before injecting bridge into `M_e`, simulate the bridge event sequence through the existing DFAs (`SAFE_N_dfa.dot`). If any DFA enters a reject/trap state, discard the bridge. This comes for free once safety DFAs are wired into the BFS.

### Layer 5: Human-in-the-Loop (optional, UX decision)
Log the LLM's proposed bridge and optionally require operator approval before execution in safety-critical scenarios.

## Where to Implement

- `environment_model.py` → `llm_explore_states_and_events()` — add validation between `json.loads(raw)` and `return bridge_tools`
- Layers 1-3 can be implemented immediately
- Layer 4 depends on the safety DFA integration TODO
