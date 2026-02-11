# Deterministic Failure-Context + DAG -> PDDL Checklist

This document is an implementation checklist for compiling replanning PDDL
deterministically from:
- runtime `failure_context`
- current DAG/FSA execution context
- robot/part runtime state
- `tools.json`/resource capabilities

No LLM is required for this path.

## 1. Required Inputs

1. Failure payload:
   - `violations[*].failure_context.failure_mode`
   - `violations[*].failure_context.affected_entities`
   - `violations[*].failure_context.observations.state_before`
   - `violations[*].failure_context.observations.state_after`
2. Runtime state:
   - `system_state.robots`
   - `system_state.parts`
3. Current plan graph:
   - task nodes with `id`, `status`, `function_name`, `params`, dependencies
4. Action catalog:
   - `tools_catalog` from `tools.json`
5. Resource capabilities:
   - `resource_infos` (`reachability`, `staging_areas`, etc.)
6. Safety/ordering source:
   - DAG precedence and/or compiled safety constraints

## 2. Domain Compiler (Deterministic)

1. Build domain actions only from `tools_catalog` function names.
2. Keep a fixed allowed predicate set (example):
   - `idle`, `carrying`, `resource-positioned`
   - `part-at`, `part-placed`
   - `reachable`, `resource-available`
   - `placed`, `can-place` (for ordering)
3. Reject any action outside the tools-derived allowlist.
4. Keep planner-compatible requirements:
   - `:strips :typing :negative-preconditions`

## 3. Objects Compiler

1. Resources:
   - include operational resources
   - include resources referenced by relevant pending/blocked tasks
2. Parts:
   - include affected entities
   - include parts referenced by relevant pending/blocked descendants
3. Locations:
   - include origin/destination locations from relevant tasks
   - include known part locations (or synthetic failed locations)
   - include reachable/staging locations for selected resources
4. Normalize all names to PDDL-safe tokens with a reversible map.

## 4. Init Compiler

1. Resource facts:
   - add `(resource-available r)` unless policy marks robot unavailable
   - if robot holds a part: add `(carrying r p)` or `(resource-positioned r p)`
   - if free: add `(idle r)`
2. Part facts:
   - placed/verified -> `(part-placed p l)`
   - known location -> `(part-at p l)`
   - unknown label but known coordinates/last-known -> map to `failed-loc-*` and add `(part-at p failed-loc-*)`
3. Reachability:
   - add `(reachable r l)` only from declared capabilities
4. Failure delta application:
   - compare `state_before` and `state_after`
   - remove contradicted facts
   - add implied facts from failure observations

## 5. Goal Compiler

1. Determine remaining required outcomes from pending/blocked descendants.
2. Map terminal task intent to goal predicates (example):
   - `place_part(part_name, destination_location)` -> `(part-placed part destination)`
3. Exclude already satisfied goals from completed work/current state.

## 6. Constraint Compiler (DAG -> PDDL)

1. Compile precedence with control predicates:
   - effect of predecessor adds marker (`placed sg`, `done-<task>`)
   - dependent action precondition requires marker (`can-place mcp`, `done-<task>`)
2. Initialize frontier enable facts in `:init`.
3. If using unlock actions:
   - generate them deterministically from precedence edges
   - do not rely on invented action names
4. Alternative:
   - use multi-shot planning per frontier layer (no synthetic unlock actions)

## 7. Validation Gates (Fail Fast)

1. Problem integrity:
   - all predicates in allowlist
   - all objects typed/declared
   - no dangling symbols
2. Plan integrity:
   - every solved action must be in tools-derived allowlist
   - fail on unknown actions (do not skip unknown steps)
3. Translation integrity:
   - all plan actions map to executable functions
   - all required params resolved
4. Safety gate:
   - if validation fails, return no patch instead of partial patch

## 8. Unknown Failure Policy (No LLM)

1. Ignore unknown label; trust structured evidence.
2. Primary evidence order:
   - `state_after`
   - part tracker current location/state
   - failure observations (`affected_entities`, coordinates, last-known)
3. If essential facts are missing:
   - apply conservative defaults
   - mark for observation/human intervention instead of fabricating facts

## 9. Minimum Test Suite

1. `failed:misplaced` with known coordinates.
2. Unknown failure mode with valid before/after snapshots.
3. Resource operational-but-occupied (holding part).
4. Resource unavailable with alternate resource reroute.
5. Conflicting state inputs (define and assert source precedence policy).
6. UNSAT case returns cleanly with no partial patch application.

## 10. Suggested Implementation Order (Tomorrow)

1. Add `problem_compiler.py`:
   - `build_objects(...)`
   - `build_init(...)`
   - `build_goal(...)`
2. Add `constraint_compiler.py`:
   - DAG precedence -> control predicates/facts
3. Wire deterministic path into `pddl/replanner.py` before LLM path.
4. Add strict validators and fail-fast behavior.
5. Add tests for all scenarios in section 9.

## Reference Mapping (Current Runtime Example)

Given:
- `SG` failed on `place_part` and is misplaced
- `UR5e` is holding `MCP`
- downstream task depends on failed `SG` placement

Expected deterministic compile outcome:
1. Objects include:
   - resources: `ur5e`, `xarm6` (if operational policy allows)
   - parts: `sg`, `mcp`
   - locations: failed location + destination + relevant reachable/staging locations
2. Init includes:
   - `UR5e` occupied (`carrying` or `resource-positioned`), not `idle`
   - `SG` at failed symbolic location
   - reachability facts from capabilities
3. Goals include unmet placements:
   - `(part-placed sg assembly-board-v1)`
   - `(part-placed mcp assembly-board-v1)`

