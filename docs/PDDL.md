# LLM + PDDL Online Replanning

## What this is

Adding PDDL **only for online replanning**. Everything else (initial planning, DAG, FSA, safety monitors) stays unchanged.

### Why

The current `replan_with_feedback_online()` sends a raw JSON dump to the LLM and hopes it generates a valid recovery plan. No formal guarantees. No constraint enforcement. For deadlocks involving multiple robots and safety ordering constraints (e.g. SG before MCP), this fails.

### What changes

| Component | Before | After |
|---|---|---|
| Initial planning | LLM → DAG | unchanged |
| FSA compilation | DAG → FSA | unchanged |
| Safety monitoring | OnlineSafetyMonitor | unchanged |
| **Online replanning** | **LLM direct patch** | **LLM builds PDDL problem → PDDL planner solves it → patch DAG** |
| Unknown failures | LLM fallback | LLM fallback (unchanged) |

---

## The Scenario That Motivates This

```
Assembly:   SG from prusa-mk4-1 → assembly-board-v1
            MCP from prusa-mk4-2 → assembly-board-v1
Safety:     SG must be placed before MCP

Failure:    xArm6 fails to place SG
            SG drops to UR5e region
            UR5e is holding MCP
            Deadlock — no valid next action exists

Recovery:   UR5e places MCP back to prusa-mk4-2
            UR5e picks SG from ur5e-region
            UR5e places SG on assembly-board       ← sg-assembled = true
            UR5e picks MCP from prusa-mk4-2
            UR5e places MCP on assembly-board      ← constraint satisfied
```

The PDDL planner finds this recovery automatically because placing MCP back
is the only valid first action from the deadlock state given the preconditions.

---

## Architecture

```
Failure occurs
      ↓
CCA sends replan_request (failure_context, part_tracker, robot states)
      ↓
Step 1: LLM call — build focused PDDL problem
  - filter to available robots only
  - filter to relevant locations only
  - add part origin facts
  - add cost hints (prefer returning part to origin)
      ↓
Step 2: PDDL planner — solve it
  - Fast Downward or similar
  - guaranteed constraint-correct plan
  - deterministic, fast
      ↓ (if unsolvable)
Step 3: LLM fallback — existing replan_with_feedback_online()
      ↓
Patch DAG → recompile FSA → resume execution
```

### Why LLM + PDDL, not PDDL alone

- PDDL has no language understanding — cannot read NL requirements or safety text
- PDDL alone finds technically valid plans but semantically wrong (e.g. places MCP at random staging area instead of origin printer)
- LLM focuses the search space (relevant robots, locations, costs)
- PDDL guarantees the plan satisfies all safety preconditions

### Why not replace DAG with PDDL

- `compile_global_fsa()` needs explicit predecessor/successor structure for parallel composition
- PDDL plans are flat ordered lists — causal links are implicit in preconditions/effects, not stored as data
- Rewriting FSA compilation is high risk for a safety-critical component that already works
- PDDL patches the DAG after replanning — DAG feeds FSA as before

---

## What to Build

### 1. PDDL Domain File

**File:** `cais_spade_llm/pddl/assembly_domain.pddl`

Extract from existing `in_state`/`out_state` docstrings in `robot_agent.py`.

```pddl
(define (domain assembly)
  (:requirements :strips :equality :conditional-effects :action-costs)

  (:types robot part location)

  (:predicates
    (robot-idle ?r - robot)
    (holding ?r - robot ?p - part)
    (part-at ?p - part ?l - location)
    (robot-at ?r - robot ?l - location)
    (part-origin ?p - part ?l - location)
    (sg-assembled)
    (mcp-assembled))

  (:functions (total-cost))

  (:action pick_part
    :parameters (?r - robot ?p - part ?l - location)
    :precondition (and (robot-idle ?r)
                       (part-at ?p ?l))
    :effect (and (holding ?r ?p)
                 (not (robot-idle ?r))
                 (not (part-at ?p ?l))
                 (increase (total-cost) 1)))

  (:action place_part
    :parameters (?r - robot ?p - part ?l - location)
    :precondition (and (holding ?r ?p)
                       (when (= ?p MCP) (sg-assembled)))
    :effect (and (robot-idle ?r)
                 (part-at ?p ?l)
                 (not (holding ?r ?p))
                 (when (= ?p SG) (sg-assembled))
                 (when (= ?p MCP) (mcp-assembled))
                 (increase (total-cost) 1)))

  (:action move_to_location
    :parameters (?r - robot ?l - location)
    :precondition (robot-idle ?r)
    :effect (and (robot-at ?r ?l)
                 (increase (total-cost) 1)))
)
```

**Note:** Expand with `move_to_pick_location`, `move_loaded_to_destination`, `move_home`
from the existing robot docstrings as the system grows.

---

### 2. LLM PDDL Problem Builder Prompt

**File:** `cais_spade_llm/prompts.py` — add `build_pddl_problem_prompt()`

The LLM receives the failure context and outputs a PDDL problem file as a string.

```python
def build_pddl_problem_prompt(
    *,
    violations: list,
    part_tracker: dict,
    system_state: dict,
    domain_summary: str,
    safety_text: str,
) -> str:
```

The prompt must instruct the LLM to:
- Include only available robots (exclude failed ones)
- Include only relevant locations (part origins, drop sites, goal)
- Add `(part-origin ?part ?location)` facts from `part_tracker`
- Add action costs: lower cost for returning part to its origin
- Output a valid PDDL problem string (parseable by Fast Downward)

Example LLM output for the deadlock scenario:

```pddl
(define (problem deadlock-recovery)
  (:domain assembly)
  (:objects
    ur5e - robot
    SG MCP - part
    prusa-mk4-1 prusa-mk4-2 ur5e-region assembly-board-v1 - location)
  (:init
    (holding ur5e MCP)
    (part-at SG ur5e-region)
    (robot-idle ur5e)
    (part-origin MCP prusa-mk4-2)
    (part-origin SG prusa-mk4-1)
    (= (total-cost) 0))
  (:goal
    (and (sg-assembled) (mcp-assembled)))
  (:metric minimize (total-cost))
)
```

---

### 3. PDDL Planner Integration

**File:** `cais_spade_llm/pddl/planner.py`

```python
def call_planner(domain_path: str, problem_str: str) -> list[dict] | None:
    """
    Write problem to temp file, call Fast Downward, parse output.
    Returns list of grounded actions or None if unsolvable.

    Each action dict:
    {
        "action_name": "pick_part",
        "params": {"robot": "ur5e", "part": "SG", "location": "ur5e-region"}
    }
    """
```

Install Fast Downward:
```bash
pip install downward   # or build from source: https://github.com/aibasel/downward
```

---

### 4. PDDL Plan → DAG Patch Converter

**File:** `cais_spade_llm/agents/intelligent_product/process_planner.py`
— add `_patch_dag_from_pddl_plan()`

```python
def _patch_dag_from_pddl_plan(self, pddl_actions: list[dict]) -> None:
    """
    Convert PDDL grounded actions into DAG node updates.

    For each action in the plan:
    - Find matching node in self.nodes by function_name + resource_jid
    - Update status to "pending"
    - Update params if resource reassigned
    - Insert new nodes for actions not in original DAG (e.g. place_part_back)
    - Rebuild predecessor/successor links based on action sequence per robot
    """
```

---

### 5. Wire into replan_with_feedback_online()

**File:** `cais_spade_llm/agents/intelligent_product/process_planner.py`

```python
async def replan_with_feedback_online(
    self,
    violations: list[dict],
    system_coordination_state: dict | None = None
) -> None:

    # Step 1: LLM builds focused PDDL problem
    pddl_problem_str = await self.product_agent.ask_llm(
        prompt=build_pddl_problem_prompt(
            violations=violations,
            part_tracker=self.product_agent.part_tracker,
            system_state=system_coordination_state,
            domain_summary=PDDL_DOMAIN_SUMMARY,
            safety_text=self.product_agent.safety_text,
        ),
        with_functions=False,
        temperature=0.0,
    )

    # Step 2: PDDL planner solves it
    plan = call_planner(PDDL_DOMAIN_PATH, pddl_problem_str)

    if plan:
        self._patch_dag_from_pddl_plan(plan)
        self.logger.info("[Planner] PDDL recovery plan applied (%d actions).", len(plan))
        return

    # Step 3: LLM fallback for unknown failures
    self.logger.warning("[Planner] PDDL unsolvable — falling back to LLM replan.")
    await self._replan_with_feedback(violations, source="online",
                                     system_coordination_state=system_coordination_state)
```

---

## File Structure

```
cais_spade_llm/
  pddl/
    assembly_domain.pddl     ← domain file (written once)
    planner.py               ← Fast Downward integration
  agents/
    intelligent_product/
      process_planner.py     ← add _patch_dag_from_pddl_plan()
                             ← modify replan_with_feedback_online()
  prompts.py                 ← add build_pddl_problem_prompt()
```

---

## Implementation Order

1. `assembly_domain.pddl` — extract from robot docstrings
2. `planner.py` — Fast Downward integration + action parser
3. `build_pddl_problem_prompt()` in `prompts.py`
4. `_patch_dag_from_pddl_plan()` in `process_planner.py`
5. Wire into `replan_with_feedback_online()`
6. Test with deadlock scenario (xArm6 slippage, UR5e holding MCP)

---

## Test Scenario

Use the existing `sg_slippage_mode` injection in `robot_agent.py`.
It already simulates the exact deadlock described above.

Expected result after implementation:
- PDDL planner finds 5-step recovery automatically
- No LLM replan call needed for this failure type
- SG-before-MCP constraint preserved in recovery plan
- UR5e returns MCP to prusa-mk4-2 (not a random location)