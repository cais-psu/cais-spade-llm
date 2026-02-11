# LLM + PDDL Online Replanning

## What this is

PDDL-based online replanning for **all resource failures** in the manufacturing system. The LLM understands the failure and builds a focused PDDL problem; the PDDL solver finds a guaranteed constraint-correct recovery plan.

### Why not pure-LLM replanning

The previous approach (`replan_with_feedback_online()`) sent a raw JSON dump to the LLM and asked it to produce DAG patches directly. Results were poor:

- LLM hallucinated action sequences that violate safety constraints
- No formal guarantee that the recovery plan is valid
- Especially bad for multi-resource deadlocks with ordering constraints

### Why LLM + PDDL, not PDDL alone

- PDDL has no language understanding -- cannot read NL requirements or safety text
- PDDL alone finds technically valid plans but semantically wrong ones (e.g. places part at random staging area instead of origin printer)
- LLM focuses the search space (filters to relevant resources, locations, constraints)
- PDDL guarantees the plan satisfies all preconditions

### Why not replace DAG with PDDL

- `compile_global_fsa()` needs explicit predecessor/successor structure for parallel composition
- PDDL plans are flat ordered lists -- causal links are implicit in preconditions/effects, not stored as data
- Rewriting FSA compilation is high risk for a safety-critical component that already works
- PDDL patches the DAG after replanning -- DAG feeds FSA as before

---

## Current State (v1 -- Robot-Specific)

The initial PDDL module (`cais_spade_llm/pddl/`) handles **robot pick-and-place deadlocks only**.

### What works

- Domain auto-generated from `tools.json` via `pddl_domain.py`
- Problem built from system state via `pddl_problem.py`
- Solved with pyperplan (BFS STRIPS) via `pddl_planner.py`
- Translated back to DAG patches via `pddl_translator.py`
- LLM extracts pending goals via `pddl_goals.py`
- Smoke test passes for the SG/MCP deadlock scenario

### What's robot-specific (the problem)

**Domain (`pddl_domain.py`)** -- 4 hard-coded action patterns, all robotic manipulation:

| Pattern | Transition | Meaning |
|---|---|---|
| 1 | `resource_only -> resource_only` | Robot movement |
| 2 | `part_location -> resource_part` | Robot picks part |
| 3 | `resource_part -> resource_part` | Robot carries part |
| 4 | `resource_part -> part_location` | Robot places part |

No patterns for: print job rerouting, CNC re-machining, general task reassignment.

**Problem builder (`pddl_problem.py`)** -- assumes spatial reachability:

- `workspace_boundaries` with x/y/z ranges
- `_in_workspace(bounds, x, y, z)` for coordinate checking
- `staging_areas` as physical zones

For printers, "reachability" means material compatibility, not workspace bounds.

**Translator (`pddl_translator.py`)** -- hard-coded action expansions:

- `pick-part` -> `move_to_pick_location` + `pick_part`
- `place-part` / `stage-part` -> `move_loaded_to_destination` + `place_part`

Nothing for printer retries, CNC operations, or resource reassignment.

### Test scenario (robot deadlock)

```
Assembly:   SG from prusa-mk4-1 -> assembly-board-v1
            MCP from prusa-mk4-2 -> assembly-board-v1
Safety:     SG must be placed before MCP

Failure:    xArm6 fails to place SG (slippage)
            SG drops to UR5e workspace region
            UR5e is holding MCP, positioned at assembly board
            Deadlock -- no valid next action exists

Recovery (found by PDDL automatically):
  1. UR5e places MCP back at prusa-mk4-2 (origin)
  2. UR5e picks SG from ur5e-region
  3. UR5e places SG on assembly-board       <- ordering satisfied
  4. UR5e picks MCP from prusa-mk4-2
  5. UR5e places MCP on assembly-board      <- done
```

---

## Generalized Architecture (v2 -- All Resource Types)

### Pipeline

```
Failure event (from CCA)
      |
      v
+-----------------------------------+
|  LLM (understands the failure)    |
|                                   |
|  Inputs:                          |
|   - failure context from CCA      |
|   - full system state             |
|   - domain action catalog         |
|   - safety rules                  |
|                                   |
|  Outputs:                         |
|   - relevant objects (filter)     |
|   - init facts (current state)    |
|   - goal facts (what to achieve)  |
|                                   |
|  = a focused PDDL problem         |
+-----------------------------------+
      |
      v
+-----------------------------------+
|  PDDL solver (finds the plan)     |
|   - constraint-correct             |
|   - deterministic                  |
|   - fast                           |
+-----------------------------------+
      |
      v
+-----------------------------------+
|  Translator (data-driven)          |
|   - expansion rules from config    |
|   - not hard-coded per resource    |
+-----------------------------------+
      |
      v
DAG patches -> recompile FSA -> resume
      |
      v (if PDDL unsolvable)
LLM fallback (existing replan_with_feedback)
```

### Key design principle

**One unified PDDL domain** that covers all resource types. The LLM's job is to build a **focused problem** -- selecting the relevant subset of objects, facts, and goals for the specific failure. A printer failure doesn't need robot workspace facts; a gripper failure doesn't need printer queue facts. The LLM filters to what matters.

### What needs to change

#### 1. Domain: add action patterns for all resource types

Current `STATE_SEMANTICS` already includes non-robot states (`ready`, `printed`, `machined`, `processing`) but they never produce usable actions because the 4 transition patterns are all pick-and-place.

New patterns needed:

| Resource type | Failure example | Recovery actions |
|---|---|---|
| Robot | Gripper slippage, collision | stage, re-pick, re-place (already works) |
| Printer | Print fails mid-job | cancel-job, reroute-to-printer, restart-print |
| CNC | Tool break, material issue | cancel-machining, reassign-machine |
| Any resource | Resource goes offline | reassign-task to different resource of same type |

The domain generator should derive these patterns from `tools.json` the same way it currently derives robot patterns -- by reading `in_state`/`out_state` transitions.

#### 2. Problem builder: resource-type-aware modeling

Instead of only spatial reachability (`_in_workspace`), support different reachability semantics per resource type:

| Resource type | "Reachable" means |
|---|---|
| Robot | Physical workspace bounds (x/y/z) + named locations |
| Printer | Material compatibility + part size constraints |
| CNC | Tool availability + material compatibility |

The `resource_infos` structure already supports this -- each resource has `static_capabilities` which can contain type-specific fields. The problem builder just needs to not assume everything is spatial.

#### 3. Translator: data-driven action expansion

Replace the hard-coded `if action_name == "pick-part"` blocks with a config-driven expansion map:

```json
{
  "pick-part": [
    {"function": "move_to_pick_location", "param_map": {"origin_resource_location": "$loc"}},
    {"function": "pick_part", "param_map": {"origin_resource_location": "$loc"}}
  ],
  "place-part": [
    {"function": "move_loaded_to_destination", "param_map": {"destination_location": "$loc"}},
    {"function": "place_part", "param_map": {"destination_location": "$loc"}}
  ],
  "stage-part": [
    {"function": "move_loaded_to_destination", "param_map": {"destination_location": "$loc"}},
    {"function": "place_part", "param_map": {"destination_location": "$loc"}}
  ],
  "reroute-print": [
    {"function": "cancel_print_job", "param_map": {"printer_jid": "$from_resource"}},
    {"function": "start_print_job", "param_map": {"printer_jid": "$to_resource", "part_name": "$part"}}
  ],
  "reassign-task": [
    {"function": "cancel_task", "param_map": {"resource_jid": "$from_resource"}},
    {"function": "assign_task", "param_map": {"resource_jid": "$to_resource"}}
  ]
}
```

The translator reads this config and expands PDDL actions generically, instead of having per-action-name branches.

---

## File Structure

```
cais_spade_llm/
  pddl/
    __init__.py              <- exports replan()
    replanner.py             <- orchestrator: prompt -> LLM -> solver -> translate
    solver.py                <- pyperplan BFS wrapper
    test_pddl_replanner.py   <- deterministic tests (no LLM needed)
```

Integration point: `process_planner.py` `_replan_with_feedback()` tries PDDL first (online only), falls back to pure-LLM replan.

---

## Implementation History

### v1 (replaced)

Robot-specific module with 6 files: `pddl_domain.py`, `pddl_problem.py`, `pddl_planner.py`, `pddl_goals.py`, `pddl_translator.py`, `test_pddl_smoke.py`. Hard-coded action patterns, spatial reachability, robot action expansions. Deleted and replaced with v2.

### v2 (current)

1. `solver.py` -- pyperplan wrapper (from v1, cleaned up)
2. `replanner.py` -- LLM generates both PDDL domain + problem; generic translator driven by tools.json
3. `__init__.py` -- public API
4. Wired into `process_planner.py` via `_apply_replan_patch()` helper
5. Tests pass: parse, solve, translate
