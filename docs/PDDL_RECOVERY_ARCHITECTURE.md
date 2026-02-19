# PDDL-Based Recovery Architecture

## Three-Layer Failure Recovery: Compiled PDDL, LLM Reasoning, Human Intervention

This document explains the recovery planning architecture for the CAIS-SPADE-LLM manufacturing system using the 6W+1H framework. It covers what PDDL is, how it maps to the tools catalog, and why three layers — compiled PDDL, LLM extension, and human intervention — are each necessary.

---

## Scenario (used throughout this document)

**Assembly requirement:**
- Assemble part SG (small gear) from picking location prusa-mk4-1 to assembly_board-v1
- Assemble part MCP (medium circular pin) from picking location prusa-mk4-2 to assembly_board-v1

**Safety constraint:** SG must be placed before MCP.

**Resources:**

| Resource | Capabilities | Reachable locations |
|---|---|---|
| xarm6@localhost | gripper, camera, arm mobility | prusa-mk4-1, assembly-board-v1 |
| ur5e@localhost | gripper, camera, arm mobility | prusa-mk4-1, prusa-mk4-2, ur5e-workspace, assembly-board-v1 |

**The action catalog** (`tools.json`) defines only the happy path — no failure states declared:

| # | Function | in_state | out_state | Key params |
|---|----------|----------|-----------|------------|
| 1 | `move_to_pick_location` | `idle` | `at_pick` | `origin_resource_location`, `part_name` |
| 2 | `pick_part` | `printed` | `picked` | `origin_resource_location`, `part_name` |
| 3 | `move_loaded_to_destination` | `picked` | `positioned` | `destination_location`, `part_name` |
| 4 | `place_part` | `positioned` | `placed` | `destination_location`, `part_name` |

**Unexpected disruption:**
- xarm6 fails to place SG at assembly-board-v1
- SG is now physically at ur5e-workspace (misplaced)
- ur5e is in state `positioned`, holding MCP
- ur5e cannot place MCP (safety: SG must go first)
- **System deadlock — no agent can make progress**

**Assumption:** failure is observable (sensors report data, but no pre-labeled failure states).

**Ideal recovery:**
1. ur5e puts MCP back at prusa-mk4-2
2. ur5e locates the dropped SG at ur5e-workspace
3. ur5e picks and places SG at assembly-board-v1 (safety satisfied)
4. ur5e resumes MCP assembly

---

## 1. What — What is PDDL? What are the three layers?

### PDDL

PDDL (Planning Domain Definition Language) is a formal language that describes a planning problem in two parts:

**Domain** — the rules of the world (what CAN happen):
- **Types**: categories of objects (`resource`, `part`, `location`)
- **Predicates**: boolean facts about the world (`holding ur5e sg`, `robot-state-idle xarm6`, `can-reach ur5e prusa-mk4-2`)
- **Actions**: operations with preconditions (what must be true before) and effects (what changes after)

**Problem** — the current situation (what IS true, what you WANT):
- **Objects**: specific entities that exist (ur5e, xarm6, SG, MCP, prusa-mk4-1, ...)
- **Init**: currently true facts (ur5e is positioned, holding MCP, ...)
- **Goal**: desired facts (SG placed at assembly-board-v1, MCP placed at assembly-board-v1)

A **classical planner** (pyperplan BFS in this system) searches for an ordered sequence of actions that transforms init into goal. The plan is **provably valid** — every action's preconditions are guaranteed satisfied before it executes. No hallucinated steps, no skipped prerequisites, no safety violations.

### PDDL as a Discrete Event System (DES)

The `tools.json` catalog defines an automaton G = (Q, Σ, δ, q₀, Qₘ):

| DES concept | PDDL mapping | tools.json source |
|---|---|---|
| State set Q | Predicate combinations | `in_state`/`out_state` values: {idle, at_pick, picked, positioned, placed, printed} |
| Event set Σ | Actions | `function` names: {move_to_pick_location, pick_part, move_loaded_to_destination, place_part} |
| Transition δ | Precondition → effect | `in_state` → precondition, `out_state` → effect |
| Initial state q₀ | Problem `:init` | System state at runtime |
| Marked states Qₘ | Problem `:goal` | Successful assembly |

The forward-path automaton:

```
Robot:   idle ──► at_pick ──► picked ──► positioned ──► idle (cycle)
Part:  printed ──► picked ──► in_transit ──────────► placed (goal)
```

With only happy-path declarations, the automaton has no representation for failure states. When a failure occurs, the system enters a state q? ∉ Q — the automaton has no vocabulary for it, no predicate to describe it, and no transition out of it.

### The three layers

Because failures can produce states outside the compiled automaton, recovery requires three layers:

| Layer | What it does | DES interpretation |
|---|---|---|
| **Compiled PDDL** | Finds alternative paths through existing events | Navigate within G = (Q, Σ, δ, q₀, Qₘ) |
| **LLM** | Interprets unknown states, extends vocabulary and events | Extend to G' = (Q∪Q', Σ∪Σ', δ∪δ', q_current, Qₘ) |
| **Human** | Physically intervenes when system capability is exceeded | External supervisor restores system to q ∈ Q where G' is live |

---

## 2. Why — Why PDDL? Why LLM? Why Human?

### Why PDDL

When a task fails mid-execution, the system has **structured state** (robot positions, part locations, what is held) and a **defined goal** (complete the assembly). This is exactly what PDDL solves.

Compared to ad-hoc recovery logic:
- PDDL **cannot skip prerequisites** — `place_part` requires `positioned`; the planner cannot jump from `picked` to `placed`
- PDDL **cannot violate safety** — if MCP placement is blocked by a safety predicate, no valid plan includes it before SG
- PDDL **cannot hallucinate actions** — only actions defined in the domain exist
- The planner either finds a **provably valid** sequence, or reports no solution

Compared to pure LLM replanning:
- An LLM might propose "pick SG" while ur5e is holding MCP — physically impossible
- An LLM might place MCP before SG — violating safety
- PDDL makes these errors structurally impossible through precondition enforcement

PDDL is used for **recovery planning only**, not initial planning, because:
- Initial planning starts from ambiguous natural language — the problem is not well-defined enough for a formal planner
- Recovery has structured, well-defined state and goals — exactly what PDDL needs
- Formal guarantees matter more during recovery (system is mid-execution with robots in motion)

### Why LLM

**The scalability argument.**

The `tools.json` catalog defines only the happy path. It does not declare failure states because exhaustive failure declaration is infeasible at scale.

For a system with n actions each having k possible failure modes:

```
Failure states to declare:         O(n·k)
Pairwise failure interactions:     O(n²·k²)
Recovery actions to pre-design:    O(n·k) minimum, O(n²·k²) for interactions
```

A realistic manufacturing system with 15 resource types averaging 8 actions each has ~120 actions. Each action can have 10+ failure modes (misplaced, misoriented, dropped in transit, dropped at destination, partial insertion, part damaged, gripper fault, collision with fixture, collision with other part, wrong part, ...). Declaring all failure states requires a designer to anticipate every possible failure mode, name it, and design a recovery action at design time. Pairwise interactions between simultaneous failures are combinatorially explosive.

The LLM replaces this enumeration burden:

```
With exhaustive failure declarations:        With LLM:
  Designer specifies:                          Designer specifies:
    what can go RIGHT                            what can go RIGHT
    what can go WRONG (all of it)              LLM figures out:
  Scales: O(n·k) to O(n²·k²)                    what to do when things go wrong
  Requires: omniscient designer                Scales: O(n)
  Breaks: unanticipated failure                Breaks: failure beyond LLM reasoning
          → system is blind                            → falls to human
```

Specifically, the LLM does three things no compiler can when failure states are not declared:

**1. Interpret raw observations into a state description.** Without declared failure states, the system receives raw sensor data (gripper force, camera detections, confidence scores). The LLM reasons about the physical meaning: "force=0 means released, camera found part upright and undamaged means graspable."

**2. Extend the state vocabulary.** The compiled domain has no predicate for what SG is after a failed placement. The LLM creates one (e.g., `part-state-displaced`) — defining a new element of Q that did not exist in the automaton.

**3. Design recovery actions grounded in physical capabilities.** The LLM proposes actions using known resource capabilities (camera, gripper) that bridge the new failure state back to the forward path. It decides preconditions, effects, and target states based on physical reasoning about the specific situation.

The LLM is the system's ability to observe something it has no name for, give it a name, and figure out what to do about it.

### Why Human

The LLM can only reason about failures within the observability and capability envelope of the system. Human intervention is needed when that envelope is exceeded:

| Trigger | Why the system cannot handle it |
|---|---|
| **Physical capability gap** | No resource can physically perform the needed action (e.g., part fell on floor, no robot reaches it) |
| **Observability gap** | Sensors cannot determine the actual state (e.g., camera cannot find part, contradictory readings) |
| **Safety unverifiable** | CCA cannot prove the recovery plan satisfies LTLf spec (e.g., other robot's position uncertain, collision risk unverifiable) |
| **Cascading failures** | Multiple simultaneous failures exhaust all resources (e.g., ur5e gripper also faults while holding MCP) |
| **Goal unachievable** | Physical state makes the original goal impossible (e.g., part is damaged, needs reprint — a business/quality decision) |
| **Cost/time judgment** | Recovery is feasible but may not be worth it (e.g., 30-step automated plan takes 45 min vs. manual reset in 10 min) |

The human is the ultimate supervisor — they can observe what sensors cannot, manipulate what no robot reaches, confirm states the CCA cannot verify, and redefine goals when the original objective is no longer achievable.

---

## 3. Who — Who is involved?

| Role | Actor | Layer |
|---|---|---|
| **Reports failure** | Resource agent (xarm6/ur5e) — raw sensor data | — |
| **Detects deadlock** | CCA (safety agent) — no agent can progress | — |
| **Triggers replan** | ProductAgent → `_replan_with_feedback()` | — |
| **Compiles forward domain** | Python compiler (from `tools.json` happy path) | Layer 1 |
| **Compiles problem** | Python compiler (from system state) | Layer 1 |
| **Identifies gaps** | Compiler (blocking state analysis) | Layer 1 → 2 handoff |
| **Solves** | pyperplan BFS | Layer 1 and 2 |
| **Interprets failure** | LLM (raw sensors → state description) | Layer 2 |
| **Extends vocabulary** | LLM (new predicates for unnamed failure states) | Layer 2 |
| **Designs recovery actions** | LLM (new events grounded in resource capabilities) | Layer 2 |
| **Verifies safety** | CCA (DFA from LTLf spec checks proposed plan) | Layer 2 |
| **Translates plan → DAG** | `_translate_plan()` (PDDL actions → task DAG patch) | Layer 1 and 2 |
| **Merges into live DAG** | `_apply_replan_patch()` | Layer 1 and 2 |
| **Physically intervenes** | Human operator | Layer 3 |
| **Redefines goals** | Human operator | Layer 3 |
| **Executes recovery** | Assigned resource agent | Post-recovery |

---

## 4. When — When does each layer trigger?

### Timeline

```
INITIAL PLANNING (no PDDL)
├─ NL requirements ──LLM──► structured requirements ──LLM──► task DAG ──► FSA
│
PARALLEL EXECUTION (FSA monitors)
├─ xarm6: T1 ✓ → T2 ✓ → T3 ✓ → T4 ✗ FAILED (place SG)
├─ ur5e:  T5 ✓ → T6 ✓ → T7 ✓ → T8 ○ BLOCKED (safety: SG first)
│
DEADLOCK DETECTED (CCA: no agent can progress)
│
├── LAYER 1: Compiled PDDL ──► pyperplan ──► no solution
│   (SG has no representable state in compiled vocabulary)
│
├── LAYER 2: LLM extends domain ──► pyperplan ──► 10-step plan ✓
│   (LLM interprets failure, adds predicates and recovery actions)
│   CCA safety check ──► PASS
│
│   If Layer 2 also fails:
├── LAYER 3: Human intervention
│   (retrieve part, confirm state, redefine goal)
│
RECOVERY EXECUTION
└─ ur5e executes 10 recovery tasks → assembly complete
```

### Trigger conditions

**Layer 1 (Compiled PDDL) — always, immediately, every replan starts here:**
- Cost: ~0ms, free, deterministic, works offline
- Succeeds when the failure only requires rerouting through existing events (e.g., resource substitution)
- Fails when the system is in a state not in the compiled vocabulary

**Layer 2 (LLM Extension) — only when Layer 1 reports no solution:**
- Cost: ~2-3s, one LLM API call
- Succeeds when the failure is observable and recoverable with existing resource capabilities
- Fails when no feasible recovery event exists, state is unobservable, or safety cannot be verified

**Layer 3 (Human) — only when Layer 2 cannot produce a safe, feasible plan:**
- Cost: variable (human response time)
- Always available as final fallback
- After human action, system re-enters Layer 1 with updated state

---

## 5. Where — Where does each layer sit in the architecture?

```
                    INITIAL PLANNING (no PDDL)
                    ══════════════════════════
     NL ──LLM──► Requirements ──LLM──► Task DAG ──► FSA
                                            │
                    EXECUTION               │
                    ═════════               ▼
                                     FSA monitors
                                            │
                                       task fails
                                            │
                    RECOVERY                ▼
                    ════════           CCA violation
                                            │
              ┌─────────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ LAYER 1: Compiled PDDL                                    │
   │                                                            │
   │  tools.json (happy path) ──► Domain                        │
   │    4 catalog actions, forward-path predicates only         │
   │    safety constraint encoding                              │
   │                                                            │
   │  system_state ──► Problem                                  │
   │    robot states, known part locations, goal                │
   │    NOTE: failed part has no representable state            │
   │                                                            │
   │  pyperplan ──► solution? ──YES──► translate ──► merge      │
   │                    │                              │        │
   │                   NO                           execute     │
   │                    │                                       │
   │  Gap analysis:                                             │
   │    "SG not representable — no predicate for its state.     │
   │     ur5e stuck in positioned+holding — only exit is        │
   │     place-part but safety-blocked."                        │
   └────────────────────┬───────────────────────────────────────┘
                        │ gap analysis + raw failure observations
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ LAYER 2: LLM Reasoning                                    │
   │                                                            │
   │  INPUT:                                                    │
   │    compiled domain + gap analysis + raw sensor data        │
   │    resource capabilities (camera, gripper, mobility)       │
   │                                                            │
   │  LLM STEP 1: Interpret raw observations → state            │
   │  LLM STEP 2: Extend vocabulary (new predicates)            │
   │  LLM STEP 3: Design recovery actions (new events)          │
   │                                                            │
   │  pyperplan (extended) ──► solution?                         │
   │         │                                                  │
   │    ┌────┴────┐                                             │
   │  YES        NO ───────────────────────────┐                │
   │    │                                      │                │
   │    ▼                                      │                │
   │  CCA safety check                         │                │
   │    │                                      │                │
   │  ┌─┴──┐                                   │                │
   │ SAFE  UNSAFE ─────────────────────────────┤                │
   │  │                                        │                │
   │  ▼                                        │                │
   │ translate ──► merge ──► execute            │                │
   └────────────────────────────────────────────┼───────────────┘
                                                │
                                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │ LAYER 3: Human Intervention                               │
   │                                                            │
   │  System reports:                                           │
   │    what failed, current state, why Layers 1-2 failed       │
   │                                                            │
   │  Human can:                                                │
   │    observe what sensors cannot                             │
   │    manipulate what no robot reaches                        │
   │    confirm states CCA cannot verify                        │
   │    redefine goals (accept partial, reprint, abort)         │
   │                                                            │
   │  After human action ──► re-enter Layer 1 with new state    │
   └──────────────────────────────────────────────────────────┘
```

### Code locations

- Domain/problem compiler: alongside `cais_spade_llm/pddl/replanner.py`
- LLM extension prompt: `cais_spade_llm/pddl/replanner.py:_build_pddl_prompt()`
- Solver: `cais_spade_llm/pddl/solver.py` (pyperplan BFS wrapper)
- Plan translator: `cais_spade_llm/pddl/replanner.py:_translate_plan()`
- Recovery trigger: `cais_spade_llm/agents/intelligent_product/process_planner.py:_replan_with_feedback()`

---

## 6. Which — Which PDDL subset? Which failures does each layer handle?

### PDDL subset

```
(:requirements :strips :typing :negative-preconditions)
```

The minimal subset pyperplan supports. Types: `resource`, `part`, `location`. Actions map 1:1 to `tools.json` with hyphens (`move_to_pick_location` → `move-to-pick-location`).

Excluded features and why:

| Feature | Why excluded |
|---|---|
| Conditional effects (`when`) | pyperplan does not support; not needed for linear state chains |
| Action costs / `:functions` | All actions equally weighted; BFS finds shortest plan |
| Equality (`=`) | Resources distinguished by name, not compared |
| Quantifiers (`forall`, `exists`) | Small object sets; can enumerate explicitly |
| Temporal planning | Actions sequential per resource; DAG handles parallelism at execution layer |

### Safety encoding

- Predicate: `(placement-allowed ?p - part)` — is this part cleared for final placement?
- Init: `(placement-allowed sg)` — SG can be placed. MCP is NOT listed.
- Unlock action: `unlock-mcp-placement` fires when `(part-state-placed sg)` becomes true, setting `(placement-allowed mcp)`
- Standard STRIPS pattern for encoding ordering constraints without conditional effects.

### Which failures each layer handles

**Layer 1 — Compiled PDDL (alternative paths through existing events):**

Handles failures where all entities remain in cataloged states and only resource availability or reachability changes.

Example: xarm6 goes offline before picking SG. SG is still `printed` at prusa-mk4-1. ur5e is `idle`. The planner reassigns all SG tasks to ur5e by omitting `(resource-available xarm6-localhost)` from the problem. All states are in Q, all events are in Σ, the planner finds a different path through the same automaton.

**Layer 2 — LLM extension (new events for states outside the vocabulary):**

Handles failures where entities enter states not modeled in the forward path.

Example: the deadlock scenario. SG is in an unnamed state (misplaced, not in any cataloged state). The LLM interprets sensor data, creates a new predicate (`part-state-displaced`), and designs recovery actions (`release-to-storage`, `locate-and-recover`) that bridge the unnamed state back to the forward path.

**Layer 3 — Human intervention (beyond system capability):**

Handles failures where the system's physical capabilities or observability are exceeded.

Examples:
- Part fell on the floor (no robot can reach it)
- Camera cannot find part (state unobservable)
- Robot position uncertain after fault (safety unverifiable by CCA)
- Both robots faulted simultaneously (no resource available)
- Part is physically damaged (goal needs redefinition)

---

## 7. How — End-to-end walkthrough

### Phase A: Initial plan (before failure)

The LLM produces this task DAG from the NL requirements:

```
xarm6 handles SG (from prusa-mk4-1):
  T1: move_to_pick_location(xarm6, SG, prusa-mk4-1)              idle → at_pick
  T2: pick_part(xarm6, SG, prusa-mk4-1)                          printed → picked
  T3: move_loaded_to_destination(xarm6, SG, assembly-board-v1)    picked → positioned
  T4: place_part(xarm6, SG, assembly-board-v1)                    positioned → placed

ur5e handles MCP (from prusa-mk4-2), parallel start:
  T5: move_to_pick_location(ur5e, MCP, prusa-mk4-2)              idle → at_pick
  T6: pick_part(ur5e, MCP, prusa-mk4-2)                          printed → picked
  T7: move_loaded_to_destination(ur5e, MCP, assembly-board-v1)    picked → positioned
  T8: place_part(ur5e, MCP, assembly-board-v1)                    positioned → placed

DAG: T1→T2→T3→T4, T5→T6→T7→T8, T4→T8 (safety: SG before MCP)
```

T1-T3 and T5-T7 execute in parallel. T8 waits for T4.

### Phase B: Failure and deadlock

```
T1-T3: ✓ completed     T5-T7: ✓ completed
T4:    ✗ FAILED         T8:    ○ BLOCKED (safety)
```

System state after failure:

| Entity | State | Detail |
|---|---|---|
| xarm6 | idle | Gripper released after failed placement |
| ur5e | positioned | Holding MCP, waiting to place |
| SG | ??? | Raw sensor data only — no labeled state |
| MCP | in ur5e gripper | Last successful: move_loaded_to_destination |

Raw failure report from xarm6:
```json
{
  "task_id": "T4",
  "function": "place_part",
  "status": "failed",
  "error": "placement verification failed",
  "observations": {
    "gripper_force": 0.0,
    "camera_detection": {
      "object_found": true,
      "location": [0.25, -0.15, 0.02],
      "zone": "ur5e-workspace",
      "shape_match_confidence": 0.85,
      "orientation": "upright",
      "visible_damage": false
    }
  }
}
```

Why it is a deadlock (in DES terms): the system is in a state where no event in Σ is enabled. ur5e cannot place MCP (safety). No agent can pick SG (SG has no representable state — it is not `printed`, not `picked`, not `placed`). The automaton has entered a state q? ∉ Q.

### Phase C: Layer 1 — Compiled PDDL (no LLM)

The compiler produces a domain from `tools.json` (4 actions, forward-path predicates only, safety encoding) and a problem from system state.

**Domain** (compiled mechanically from `tools.json`):

```pddl
(define (domain manufacturing)
  (:requirements :strips :typing :negative-preconditions)
  (:types resource part location - object)

  (:predicates
    (robot-state-idle ?r - resource)
    (robot-state-at-pick ?r - resource)
    (robot-state-picked ?r - resource)
    (robot-state-positioned ?r - resource)
    (part-state-printed ?p - part)
    (part-state-picked ?p - part)
    (part-state-placed ?p - part)
    (holding ?r - resource ?p - part)
    (part-at ?p - part ?l - location)
    (can-reach ?r - resource ?l - location)
    (resource-available ?r - resource)
    (placement-allowed ?p - part)
  )

  (:action move-to-pick-location
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-idle ?r) (resource-available ?r)
      (can-reach ?r ?l) (part-at ?p ?l))
    :effect (and
      (robot-state-at-pick ?r) (not (robot-state-idle ?r))))

  (:action pick-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-at-pick ?r) (can-reach ?r ?l)
      (part-at ?p ?l) (part-state-printed ?p))
    :effect (and
      (robot-state-picked ?r) (holding ?r ?p) (part-state-picked ?p)
      (not (robot-state-at-pick ?r)) (not (part-state-printed ?p))
      (not (part-at ?p ?l))))

  (:action move-loaded-to-destination
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-picked ?r) (holding ?r ?p) (can-reach ?r ?l))
    :effect (and
      (robot-state-positioned ?r) (not (robot-state-picked ?r))))

  (:action place-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-positioned ?r) (holding ?r ?p)
      (can-reach ?r ?l) (placement-allowed ?p))
    :effect (and
      (robot-state-idle ?r) (resource-available ?r)
      (part-state-placed ?p) (part-at ?p ?l)
      (not (robot-state-positioned ?r)) (not (holding ?r ?p))))

  (:action unlock-mcp-placement
    :parameters ()
    :precondition (part-state-placed sg)
    :effect (placement-allowed mcp))
)
```

**Problem** (compiled from runtime state):

```pddl
(define (problem recover-assembly-deadlock)
  (:domain manufacturing)
  (:objects
    ur5e-localhost xarm6-localhost - resource
    sg mcp - part
    prusa-mk4-1 prusa-mk4-2 assembly-board-v1 ur5e-workspace - location)
  (:init
    (robot-state-idle xarm6-localhost)
    (resource-available xarm6-localhost)
    (can-reach xarm6-localhost prusa-mk4-1)
    (can-reach xarm6-localhost assembly-board-v1)

    (robot-state-positioned ur5e-localhost)
    (resource-available ur5e-localhost)
    (holding ur5e-localhost mcp)
    (can-reach ur5e-localhost prusa-mk4-1)
    (can-reach ur5e-localhost prusa-mk4-2)
    (can-reach ur5e-localhost assembly-board-v1)
    (can-reach ur5e-localhost ur5e-workspace)

    ;; SG: no representable state — not printed, not picked, not placed
    ;; MCP: in ur5e gripper
    (part-state-picked mcp)

    (placement-allowed sg))
  (:goal (and
    (part-state-placed sg) (part-at sg assembly-board-v1)
    (part-state-placed mcp) (part-at mcp assembly-board-v1)))
)
```

**Result: NO SOLUTION.** SG is absent from the model — no predicate describes its state, so no action can produce `(part-state-placed sg)`.

**Gap analysis:** "SG has no representable state. ur5e is in positioned+holding with only exit being place-part, which is safety-blocked. Raw failure data available for Layer 2."

### Phase D: Layer 2 — LLM reasoning

The LLM receives: compiled domain + gap analysis + raw failure observations + resource capabilities.

**LLM Step 1 — Interpret:** "gripper_force=0 means released. Camera found SG at ur5e-workspace, upright, undamaged. Conclusion: SG is displaced but graspable."

**LLM Step 2 — Extend vocabulary:**
```pddl
;; New predicate — does not exist in compiled domain
(part-state-displaced ?p - part)

;; Updated init
(part-state-displaced sg)
(part-at sg ur5e-workspace)
```

**LLM Step 3 — Design recovery actions:**

```pddl
;; RECOVERY ACTION 1: release-to-storage
;; Robot opens gripper to place a held part as temporary storage.
;; Part returns to "printed" (re-pickable). No placement-allowed
;; check — this is not final assembly.
;; Physical basis: same gripper-open action as place_part.
(:action release-to-storage
  :parameters (?r - resource ?p - part ?l - location)
  :precondition (and
    (robot-state-positioned ?r) (holding ?r ?p) (can-reach ?r ?l))
  :effect (and
    (robot-state-idle ?r) (resource-available ?r)
    (part-at ?p ?l) (part-state-printed ?p)
    (not (robot-state-positioned ?r)) (not (holding ?r ?p))
    (not (part-state-picked ?p))))

;; RECOVERY ACTION 2: locate-and-recover
;; Robot uses camera to visually locate a displaced part and
;; confirms it is graspable. Transitions to "printed" (pickable).
;; Physical basis: camera perception (declared in resource capabilities).
(:action locate-and-recover
  :parameters (?r - resource ?p - part ?l - location)
  :precondition (and
    (robot-state-idle ?r) (resource-available ?r)
    (part-state-displaced ?p) (part-at ?p ?l) (can-reach ?r ?l))
  :effect (and
    (part-state-printed ?p) (not (part-state-displaced ?p))))
```

In DES terms: Σ is extended with two new events. New transitions connect the `displaced` dead-end and the `positioned+holding` trap back to the live portion of the automaton.

**pyperplan solves the extended domain:**

```
Step  Action                                               Resource  Part
────  ───────────────────────────────────────────────────── ──────── ────
 1    release-to-storage(ur5e, mcp, prusa-mk4-2)           ur5e      MCP
 2    locate-and-recover(ur5e, sg, ur5e-workspace)          ur5e      SG
 3    move-to-pick-location(ur5e, sg, ur5e-workspace)       ur5e      SG
 4    pick-part(ur5e, sg, ur5e-workspace)                   ur5e      SG
 5    move-loaded-to-destination(ur5e, sg, assembly-board-v1) ur5e    SG
 6    place-part(ur5e, sg, assembly-board-v1)               ur5e      SG
      unlock-mcp-placement()
 7    move-to-pick-location(ur5e, mcp, prusa-mk4-2)        ur5e      MCP
 8    pick-part(ur5e, mcp, prusa-mk4-2)                    ur5e      MCP
 9    move-loaded-to-destination(ur5e, mcp, assembly-board-v1) ur5e   MCP
10    place-part(ur5e, mcp, assembly-board-v1)              ur5e      MCP
```

This matches the ideal recovery:
1. ur5e puts MCP back at printer (step 1 — LLM action)
2. ur5e locates dropped SG (step 2 — LLM action)
3. ur5e picks and places SG (steps 3-6 — catalog actions, safety satisfied at step 6)
4. ur5e resumes MCP assembly (steps 7-10 — catalog actions)

**Validation chain before execution:**

```
LLM proposals
     │
CHECK 1: PDDL syntax (pyperplan parser)
     │    Catches: undefined predicates, type mismatches
CHECK 2: Plan validity (pyperplan BFS)
     │    Catches: impossible precondition chains, unreachable goals
CHECK 3: Safety (CCA DFA from LTLf spec)
     │    Catches: safety violations (e.g., MCP placed before SG)
CHECK 4: Catalog mapping (_translate_plan via tools.json)
     │    Catches: actions with no physical basis in resource capabilities
CHECK 5: Execution (resource agent)
         Catches: physically infeasible actions → re-enter recovery loop
```

### Phase E: Layer 3 — Human intervention examples

**SG fell on the floor (physical capability gap):**
Camera detects nothing in any robot workspace. LLM cannot propose any recovery — no resource can reach the floor. Human retrieves part, places on printer. System re-enters Layer 1.

**Camera returns contradictory data (observability gap):**
Object detected at two locations simultaneously (sensor fault). LLM cannot determine SG state with confidence. Human visually inspects, confirms actual location. System re-enters Layer 2.

**ur5e gripper also faults (cascading failure):**
ur5e cannot open gripper. LLM's `release-to-storage` is physically infeasible. xarm6 cannot reach ur5e-workspace. Zero feasible recovery events. Human manually releases gripper and retrieves parts. System reset.

**SG is damaged (goal unachievable):**
Camera detects SG but reports visible damage (teeth broken). Placing it produces a defective assembly. LLM reports options: reprint (2hr), use spare, partial assembly, abort. Human makes business decision. System receives new goal.

**xarm6 position uncertain (safety unverifiable):**
xarm6 encoder faulted after failure, position unknown. LLM proposes recovery plan for ur5e. CCA cannot verify xarm6 is clear of ur5e-workspace — collision risk unverifiable. Plan rejected. Human confirms xarm6 position. CCA can now verify safety. System re-enters Layer 2.

### Layer 1 success example (no LLM needed)

Not all failures require LLM intervention. When xarm6 goes offline **before** picking SG (e.g., e-stop during T1), all entities remain in cataloged states:

```
SG: printed at prusa-mk4-1 (untouched)
MCP: printed at prusa-mk4-2 (untouched, or picked by ur5e)
ur5e: idle (or in normal forward-path state)
```

The compiled PDDL simply omits `(resource-available xarm6-localhost)` and pyperplan reassigns everything to ur5e:

```
1. move-to-pick-location(ur5e, sg, prusa-mk4-1)
2. pick-part(ur5e, sg, prusa-mk4-1)
3. move-loaded-to-destination(ur5e, sg, assembly-board-v1)
4. place-part(ur5e, sg, assembly-board-v1)
   unlock-mcp-placement()
5. move-to-pick-location(ur5e, mcp, prusa-mk4-2)
6. pick-part(ur5e, mcp, prusa-mk4-2)
7. move-loaded-to-destination(ur5e, mcp, assembly-board-v1)
8. place-part(ur5e, mcp, assembly-board-v1)
```

No new predicates, no new actions, no LLM. The failure only changed which resources are available — the automaton vocabulary was sufficient.

---

## Summary

```
Failure complexity ──────────────────────────────────────────────────────────►

  Known rerouting         Unknown but recoverable         Beyond system capability
  ────────────────        ──────────────────────          ──────────────────────
  │                │      │                      │        │                      │
  │   LAYER 1      │      │      LAYER 2          │        │      LAYER 3          │
  │   Compiled PDDL│      │   LLM Reasoning       │        │   Human Intervention  │
  │                │      │                      │        │                      │
  │ Domain: tools  │      │ Interprets raw        │        │ Observes what sensors │
  │   .json happy  │      │   sensor data         │        │   cannot              │
  │   path         │      │ Extends vocabulary    │        │ Manipulates what no   │
  │ Problem: system│      │   (new predicates)    │        │   robot reaches       │
  │   state        │      │ Designs recovery      │        │ Confirms what CCA     │
  │ Solve: pyper-  │      │   actions (new events)│        │   cannot verify       │
  │   plan BFS     │      │ Validated by planner  │        │ Redefines goals       │
  │                │      │   + CCA + translator  │        │                      │
  │ ~0ms, free     │      │ ~2-3s, 1 API call     │        │ Variable              │
  │ deterministic  │      │ Scales: O(n) — no     │        │ Always available      │
  │ works offline  │      │   pre-declared failures│        │                      │
  └────────────────┘      └──────────────────────┘        └──────────────────────┘

  WHY PDDL:                WHY LLM:                        WHY HUMAN:
  Provably valid plans.     Cannot declare all failure       System has physical,
  No hallucinated steps.    states at design time.           observability, and
  Safety enforced as hard   O(n·k) to O(n²·k²) failure      capability limits.
  preconditions. Planner    enumeration is infeasible        Human exceeds all
  guarantees every pre-     at scale. LLM interprets         three.
  condition is met.         failures at runtime from
                            first principles.
```

### In DES terms

| Layer | DES interpretation | Fails when |
|---|---|---|
| **Layer 1** | Find path in G = (Q, Σ, δ, q₀, Qₘ) | Blocking states — system in q? ∉ Q |
| **Layer 2** | Extend to G' = (Q∪Q', Σ∪Σ', δ∪δ', q, Qₘ) — add feasible events to restore liveness | No feasible event exists, state unobservable, safety unverifiable |
| **Layer 3** | External supervisor restores system to q ∈ Q where G or G' is live | Final fallback — always available |

Each layer handles what the layer before it cannot. The system degrades gracefully from instant deterministic recovery, to LLM-assisted reasoning, to human judgment — matching recovery effort to failure complexity.
