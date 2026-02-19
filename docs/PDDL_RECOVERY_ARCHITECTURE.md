# PDDL-Based Recovery Architecture

## Three-Layer Failure Recovery: Compiled PDDL, LLM Reasoning, Human Intervention

This document explains the recovery planning architecture for the CAIS-SPADE-LLM manufacturing system using the 6W+1H framework. It covers what PDDL is, how it maps to the tools catalog, and why three layers — compiled PDDL, LLM extension, and human intervention — are each necessary.

---

## Scenario (used throughout this document)

**Assembly requirement:**
- Assemble part SG (small gear) from picking location prusa-mk4-1 to assembly_board-v1
- Assemble part MCP (medium circular pin) from picking location prusa-mk4-2 to assembly_board-v1

**Safety constraint:** SG must be assembled before MCP.

**Resources:**

| Resource | Capabilities | Reachable locations |
|---|---|---|
| xarm6@localhost | gripper, camera, arm mobility | prusa-mk4-1, assembly-board-v1 |
| ur5e@localhost | gripper, camera, arm mobility | prusa-mk4-1, prusa-mk4-2, ur5e-workspace, assembly-board-v1 |

**The action catalog** (`tools.json`) defines only the happy path — no failure states declared. Each action has both a **robot state transition** (`in_state`/`out_state`) and a **part state transition** (`part_in_state` → completed state):

| # | Function | Robot: in → out | Part: in → out | Key params |
|---|----------|----------------|----------------|------------|
| 1 | `move_to_pick_location` | `idle` → `at_pick` | — | `origin_resource_location`, `part_name` |
| 2 | `pick_part` | `at_pick` → `picked` | `printed` → `in_gripper` | `origin_resource_location`, `part_name` |
| 3 | `move_loaded_to_destination` | `picked` → `positioned` | `in_gripper` → `in_transit` | `destination_location`, `part_name` |
| 4 | `place_part` | `positioned` → `idle` | `in_transit` → `printed` | `destination_location`, `part_name` |
| 5 | `assemble_part` | `positioned` → `idle` | `in_transit` → `verified` | `destination_location`, `part_name` |

Key distinction: `place_part` is **temporary staging** — the part returns to `printed` (re-pickable). `assemble_part` is **final assembly** — the part reaches `verified` (done). Safety constraints gate `assemble_part`, not `place_part`.

**Unexpected disruption:**
- xarm6 fails to assemble SG at assembly-board-v1 (task T4: `assemble_part`)
- SG is now physically at ur5e-workspace (misplaced)
- ur5e is in state `positioned`, holding MCP (part state: `in_transit`)
- ur5e cannot assemble MCP (safety: SG must go first)
- **System deadlock — no agent can make progress toward the goal**

Note: ur5e CAN `place_part` MCP (temporary staging, no safety check) to unstick itself. But SG is in an unnamed state — no catalog action can produce `(part-state-verified sg)` from it.

**Assumption:** failure is observable (sensors report data, but no pre-labeled failure states).

**Ideal recovery:**
1. ur5e places MCP back at prusa-mk4-2 (temporary staging via `place_part`)
2. ur5e locates the dropped SG at ur5e-workspace
3. ur5e picks and assembles SG at assembly-board-v1 (safety satisfied)
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
- **Goal**: desired facts (SG verified at assembly-board-v1, MCP verified at assembly-board-v1)

A **classical planner** (pyperplan BFS in this system) searches for an ordered sequence of actions that transforms init into goal. The plan is **provably valid** — every action's preconditions are guaranteed satisfied before it executes. No hallucinated steps, no skipped prerequisites, no safety violations.

### PDDL as a Discrete Event System (DES)

The `tools.json` catalog defines two coupled automata:

**Robot automaton** G_r = (Q_r, Σ, δ_r):

| DES concept | PDDL mapping | tools.json source |
|---|---|---|
| State set Q_r | Robot state predicates | `in_state`/`out_state`: {idle, at_pick, picked, positioned} |
| Event set Σ | Actions | 5 functions: {move_to_pick_location, pick_part, move_loaded_to_destination, place_part, assemble_part} |
| Transition δ_r | Precondition → effect | `in_state` → precondition, `out_state` → effect |

**Part automaton** G_p = (Q_p, Σ, δ_p):

| DES concept | PDDL mapping | tools.json source |
|---|---|---|
| State set Q_p | Part state predicates | `part_in_state` / completed state: {printed, in_gripper, in_transit, verified} |
| Event set Σ | Same actions | Same 5 functions (shared alphabet) |
| Transition δ_p | `part_in_state` → completed state | `part_transition.completed.state` |

The forward-path automata:

```
Robot:   idle ──► at_pick ──► picked ──► positioned ──► idle (cycle)
                  move_to     pick       move_loaded    place/assemble
                  _pick_loc   _part      _to_dest       _part

Part:  printed ──► in_gripper ──► in_transit ──┬──► printed  (place_part: staging)
                    pick_part      move_loaded  │
                                   _to_dest     └──► verified (assemble_part: final)
```

With only happy-path declarations, the automata have no representation for failure states. When a failure occurs, a part can enter a state q? ∉ Q_p — the automaton has no vocabulary for it, no predicate to describe it, and no transition out of it.

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
- PDDL **cannot skip prerequisites** — `pick_part` requires `at_pick`; the planner cannot jump from `idle` to `picked`
- PDDL **cannot violate safety** — if MCP assembly is blocked by a safety predicate, no valid plan includes it before SG
- PDDL **cannot hallucinate actions** — only actions defined in the domain exist
- The planner either finds a **provably valid** sequence, or reports no solution

Compared to pure LLM replanning:
- An LLM might propose "pick SG" while ur5e is holding MCP — physically impossible
- An LLM might assemble MCP before SG — violating safety
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

**2. Extend the state vocabulary.** The compiled domain has no predicate for what SG is after a failed assembly. The LLM creates one (e.g., `part-state-displaced`) — defining a new element of Q that did not exist in the automaton.

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
├─ xarm6: T1 ✓ → T2 ✓ → T3 ✓ → T4 ✗ FAILED (assemble SG)
├─ ur5e:  T5 ✓ → T6 ✓ → T7 ✓ → T8 ○ BLOCKED (safety: SG first)
│
DEADLOCK DETECTED (CCA: no agent can progress toward goal)
│
├── LAYER 1: Compiled PDDL ──► pyperplan ──► partial solution only
│   (ur5e can unstick via place_part, but SG has no representable state)
│
├── LAYER 2: LLM extends domain ──► pyperplan ──► full recovery plan ✓
│   (LLM interprets SG displacement, adds predicate and recovery action)
│   CCA safety check ──► PASS
│
│   If Layer 2 also fails:
├── LAYER 3: Human intervention
│   (retrieve part, confirm state, redefine goal)
│
RECOVERY EXECUTION
└─ ur5e executes recovery tasks → assembly complete
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
   │    5 catalog actions (2 state machines: robot + part)      │
   │    safety constraint encoding (gates assemble_part only)   │
   │                                                            │
   │  system_state ──► Problem                                  │
   │    robot states, part states, known locations, goal        │
   │    NOTE: failed part may have no representable state       │
   │                                                            │
   │  pyperplan ──► solution? ──YES──► translate ──► merge      │
   │                    │                              │        │
   │                   NO                           execute     │
   │                    │                                       │
   │  Gap analysis:                                             │
   │    "SG not representable — no predicate for its state.     │
   │     Goal (part-state-verified sg) unreachable."            │
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

- Predicate: `(assembly-allowed ?p - part)` — is this part cleared for final assembly?
- Init: `(assembly-allowed sg)` — SG can be assembled. MCP is NOT listed.
- Unlock action: `unlock-mcp-assembly` fires when `(part-state-verified sg)` becomes true, setting `(assembly-allowed mcp)`
- Only `assemble-part` checks `(assembly-allowed ?p)`. `place-part` (temporary staging) does NOT — any robot can stage any part at any time.

### Which failures each layer handles

**Layer 1 — Compiled PDDL (alternative paths through existing events):**

Handles failures where all entities remain in cataloged states and only resource availability or reachability changes.

Example: xarm6 goes offline before picking SG. SG is still `printed` at prusa-mk4-1. ur5e is `idle`. The planner reassigns all SG tasks to ur5e by omitting `(resource-available xarm6-localhost)` from the problem. All states are in Q, all events are in Σ, the planner finds a different path through the same automaton.

**Layer 2 — LLM extension (new events for states outside the vocabulary):**

Handles failures where entities enter states not modeled in the forward path.

Example: the deadlock scenario. SG is in an unnamed state (misplaced, not in any cataloged state). The LLM interprets sensor data, creates a new predicate (`part-state-displaced`), and designs a recovery action (`locate-and-recover`) that bridges the unnamed state back to the forward path (`printed`).

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
  T1: move_to_pick_location(xarm6, SG, prusa-mk4-1)              robot: idle → at_pick
  T2: pick_part(xarm6, SG, prusa-mk4-1)                          robot: at_pick → picked
                                                                   part:  printed → in_gripper
  T3: move_loaded_to_destination(xarm6, SG, assembly-board-v1)    robot: picked → positioned
                                                                   part:  in_gripper → in_transit
  T4: assemble_part(xarm6, SG, assembly-board-v1)                 robot: positioned → idle
                                                                   part:  in_transit → verified

ur5e handles MCP (from prusa-mk4-2), parallel start:
  T5: move_to_pick_location(ur5e, MCP, prusa-mk4-2)              robot: idle → at_pick
  T6: pick_part(ur5e, MCP, prusa-mk4-2)                          robot: at_pick → picked
                                                                   part:  printed → in_gripper
  T7: move_loaded_to_destination(ur5e, MCP, assembly-board-v1)    robot: picked → positioned
                                                                   part:  in_gripper → in_transit
  T8: assemble_part(ur5e, MCP, assembly-board-v1)                 robot: positioned → idle
                                                                   part:  in_transit → verified

DAG: T1→T2→T3→T4, T5→T6→T7→T8, T4→T8 (safety: SG before MCP)
```

T1-T3 and T5-T7 execute in parallel. T8 waits for T4.

### Phase B: Failure and deadlock

```
T1-T3: ✓ completed     T5-T7: ✓ completed
T4:    ✗ FAILED         T8:    ○ BLOCKED (safety)
```

System state after failure:

| Entity | Robot state | Part state | Detail |
|---|---|---|---|
| xarm6 | idle | — | Gripper released after failed assembly |
| ur5e | positioned | — | Waiting to assemble MCP |
| SG | — | ??? | Raw sensor data only — no labeled state |
| MCP | — | in_transit | Last successful: move_loaded_to_destination |

Raw failure report from xarm6:
```json
{
  "task_id": "T4",
  "function": "assemble_part",
  "status": "failed",
  "error": "assembly verification failed",
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

Why it is a deadlock (in DES terms): the system is in a state where no event in Σ can reach the goal. ur5e CAN `place_part` MCP back to storage (temporary staging, no safety check), but SG has no representable state — it is not `printed`, not `in_gripper`, not `in_transit`, not `verified`. No action's `part_in_state` precondition matches SG. The goal `(part-state-verified sg)` is unreachable.

### Phase C: Layer 1 — Compiled PDDL (no LLM)

The compiler produces a domain from `tools.json` (5 actions, dual state machines for robot + part, safety encoding) and a problem from system state.

**Domain** (compiled mechanically from `tools.json`):

```pddl
(define (domain manufacturing)
  (:requirements :strips :typing :negative-preconditions)
  (:types resource part location - object)

  (:predicates
    ;; Robot state (one active per resource at a time)
    (robot-state-idle ?r - resource)
    (robot-state-at-pick ?r - resource)
    (robot-state-picked ?r - resource)
    (robot-state-positioned ?r - resource)
    ;; Part state (one active per part at a time)
    (part-state-printed ?p - part)
    (part-state-in-gripper ?p - part)
    (part-state-in-transit ?p - part)
    (part-state-verified ?p - part)
    ;; Relations
    (holding ?r - resource ?p - part)
    (part-at ?p - part ?l - location)
    (can-reach ?r - resource ?l - location)
    (resource-available ?r - resource)
    (assembly-allowed ?p - part)
  )

  (:action move-to-pick-location
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-idle ?r) (resource-available ?r)
      (can-reach ?r ?l) (part-at ?p ?l)
      (part-state-printed ?p))
    :effect (and
      (robot-state-at-pick ?r) (not (robot-state-idle ?r))))

  (:action pick-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-at-pick ?r) (can-reach ?r ?l)
      (part-at ?p ?l) (part-state-printed ?p))
    :effect (and
      (robot-state-picked ?r) (holding ?r ?p)
      (part-state-in-gripper ?p)
      (not (robot-state-at-pick ?r)) (not (part-state-printed ?p))
      (not (part-at ?p ?l))))

  (:action move-loaded-to-destination
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-picked ?r) (holding ?r ?p)
      (part-state-in-gripper ?p) (can-reach ?r ?l))
    :effect (and
      (robot-state-positioned ?r) (part-state-in-transit ?p)
      (not (robot-state-picked ?r)) (not (part-state-in-gripper ?p))))

  (:action place-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-positioned ?r) (holding ?r ?p)
      (part-state-in-transit ?p) (can-reach ?r ?l))
    :effect (and
      (robot-state-idle ?r) (resource-available ?r)
      (part-state-printed ?p) (part-at ?p ?l)
      (not (robot-state-positioned ?r)) (not (holding ?r ?p))
      (not (part-state-in-transit ?p))))

  (:action assemble-part
    :parameters (?r - resource ?p - part ?l - location)
    :precondition (and
      (robot-state-positioned ?r) (holding ?r ?p)
      (part-state-in-transit ?p) (can-reach ?r ?l)
      (assembly-allowed ?p))
    :effect (and
      (robot-state-idle ?r) (resource-available ?r)
      (part-state-verified ?p) (part-at ?p ?l)
      (not (robot-state-positioned ?r)) (not (holding ?r ?p))
      (not (part-state-in-transit ?p))))

  (:action unlock-mcp-assembly
    :parameters ()
    :precondition (part-state-verified sg)
    :effect (assembly-allowed mcp))
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
    ;; xarm6: idle, available (released gripper after failure)
    (robot-state-idle xarm6-localhost)
    (resource-available xarm6-localhost)
    (can-reach xarm6-localhost prusa-mk4-1)
    (can-reach xarm6-localhost assembly-board-v1)

    ;; ur5e: positioned, holding MCP
    (robot-state-positioned ur5e-localhost)
    (resource-available ur5e-localhost)
    (holding ur5e-localhost mcp)
    (can-reach ur5e-localhost prusa-mk4-1)
    (can-reach ur5e-localhost prusa-mk4-2)
    (can-reach ur5e-localhost assembly-board-v1)
    (can-reach ur5e-localhost ur5e-workspace)

    ;; SG: no representable state — not printed, not in_gripper, not in_transit, not verified
    ;; MCP: in transit (held by ur5e, after move_loaded_to_destination)
    (part-state-in-transit mcp)

    ;; Safety: SG is allowed for assembly (but unreachable), MCP is not yet allowed
    (assembly-allowed sg))
  (:goal (and
    (part-state-verified sg) (part-at sg assembly-board-v1)
    (part-state-verified mcp) (part-at mcp assembly-board-v1)))
)
```

**Result: NO SOLUTION.** SG has no state predicate — no action's precondition matches it, so no chain of actions can produce `(part-state-verified sg)`. The goal is unreachable.

Note: the compiled PDDL CAN unstick ur5e — `place-part(ur5e, mcp, prusa-mk4-2)` would transition ur5e back to idle and MCP back to printed. But this alone does not solve the goal because SG remains unrepresentable.

**Gap analysis:** "SG has no representable state. Goal `(part-state-verified sg)` unreachable. Raw failure data available for Layer 2."

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

**LLM Step 3 — Design recovery action:**

```pddl
;; RECOVERY ACTION: locate-and-recover
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

Note: only ONE recovery action is needed. `release-to-storage` (from the earlier version of this doc) is no longer necessary — `place-part` in the catalog already does temporary staging (returns part to `printed` without safety checks). The LLM only needs to bridge the gap the catalog truly cannot express: the displaced-to-printed transition.

In DES terms: Q_p is extended with one new state (`displaced`). Σ is extended with one new event (`locate-and-recover`). A new transition connects the `displaced` dead-end back to the live portion of the automaton (`printed`).

**pyperplan solves the extended domain:**

```
Step  Action                                                     Robot state    Part state
────  ──────────────────────────────────────────────────────────  ─────────────  ────────────
 1    place-part(ur5e, mcp, prusa-mk4-2)                         positioned→idle  in_transit→printed
 2    locate-and-recover(ur5e, sg, ur5e-workspace)                idle (no change) displaced→printed
 3    move-to-pick-location(ur5e, sg, ur5e-workspace)             idle→at_pick     —
 4    pick-part(ur5e, sg, ur5e-workspace)                         at_pick→picked   printed→in_gripper
 5    move-loaded-to-destination(ur5e, sg, assembly-board-v1)     picked→positioned in_gripper→in_transit
 6    assemble-part(ur5e, sg, assembly-board-v1)                  positioned→idle  in_transit→verified
      unlock-mcp-assembly()                                       —               —
 7    move-to-pick-location(ur5e, mcp, prusa-mk4-2)              idle→at_pick     —
 8    pick-part(ur5e, mcp, prusa-mk4-2)                          at_pick→picked   printed→in_gripper
 9    move-loaded-to-destination(ur5e, mcp, assembly-board-v1)   picked→positioned in_gripper→in_transit
10    assemble-part(ur5e, mcp, assembly-board-v1)                positioned→idle  in_transit→verified
```

This matches the ideal recovery:
1. ur5e puts MCP back at printer (step 1 — catalog `place_part`, no safety check)
2. ur5e locates dropped SG (step 2 — LLM recovery action)
3. ur5e picks and assembles SG (steps 3-6 — catalog actions, safety satisfied at step 6)
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
     │    Catches: safety violations (e.g., MCP assembled before SG)
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
ur5e cannot open gripper. `place-part` is physically infeasible. xarm6 cannot reach ur5e-workspace. Zero feasible recovery events. Human manually releases gripper and retrieves parts. System reset.

**SG is damaged (goal unachievable):**
Camera detects SG but reports visible damage (teeth broken). Assembling it produces a defective product. LLM reports options: reprint (2hr), use spare, partial assembly, abort. Human makes business decision. System receives new goal.

**xarm6 position uncertain (safety unverifiable):**
xarm6 encoder faulted after failure, position unknown. LLM proposes recovery plan for ur5e. CCA cannot verify xarm6 is clear of ur5e-workspace — collision risk unverifiable. Plan rejected. Human confirms xarm6 position. CCA can now verify safety. System re-enters Layer 2.

---

## 8. Side-by-Side Comparison: Static PDDL vs. LLM-Extended PDDL

This section compares two failure scenarios end-to-end: one where compiled PDDL (Layer 1) is sufficient, and one where LLM extension (Layer 2) is required.

### Scenario A: Static PDDL succeeds (resource rerouting)

**Failure:** xarm6 goes offline (e-stop) **before** picking SG. All entities remain in cataloged states.

```
System state at failure:
  xarm6: offline (e-stop during T1)
  ur5e:  idle
  SG:    printed at prusa-mk4-1 (untouched)
  MCP:   printed at prusa-mk4-2 (untouched)
```

**Step 1 — Compile domain:** Same 5-action domain from tools.json (identical to Phase C above).

**Step 2 — Compile problem:**

```pddl
(define (problem reroute-after-xarm6-offline)
  (:domain manufacturing)
  (:objects
    ur5e-localhost xarm6-localhost - resource
    sg mcp - part
    prusa-mk4-1 prusa-mk4-2 assembly-board-v1 - location)
  (:init
    ;; xarm6: OFFLINE — resource-available omitted
    (robot-state-idle xarm6-localhost)
    (can-reach xarm6-localhost prusa-mk4-1)
    (can-reach xarm6-localhost assembly-board-v1)
    ;; ^^^ xarm6 exists but (resource-available xarm6-localhost) is ABSENT

    ;; ur5e: idle, available
    (robot-state-idle ur5e-localhost)
    (resource-available ur5e-localhost)
    (can-reach ur5e-localhost prusa-mk4-1)
    (can-reach ur5e-localhost prusa-mk4-2)
    (can-reach ur5e-localhost assembly-board-v1)

    ;; Both parts: printed, at their printers
    (part-state-printed sg)
    (part-at sg prusa-mk4-1)
    (part-state-printed mcp)
    (part-at mcp prusa-mk4-2)

    ;; Safety: SG allowed, MCP not yet
    (assembly-allowed sg))
  (:goal (and
    (part-state-verified sg) (part-at sg assembly-board-v1)
    (part-state-verified mcp) (part-at mcp assembly-board-v1)))
)
```

Key difference from the deadlock scenario: every entity has a cataloged state. SG is `printed` — a state the compiled domain knows about. xarm6 is simply not `resource-available`, so no action will select it.

**Step 3 — pyperplan BFS finds solution** (see Section 9 for detailed BFS trace):

```
Step  Action                                                     Resource  Part
────  ──────────────────────────────────────────────────────────  ────────  ────
 1    move-to-pick-location(ur5e, sg, prusa-mk4-1)               ur5e      SG
 2    pick-part(ur5e, sg, prusa-mk4-1)                            ur5e      SG
 3    move-loaded-to-destination(ur5e, sg, assembly-board-v1)     ur5e      SG
 4    assemble-part(ur5e, sg, assembly-board-v1)                  ur5e      SG
      unlock-mcp-assembly()
 5    move-to-pick-location(ur5e, mcp, prusa-mk4-2)              ur5e      MCP
 6    pick-part(ur5e, mcp, prusa-mk4-2)                           ur5e      MCP
 7    move-loaded-to-destination(ur5e, mcp, assembly-board-v1)    ur5e      MCP
 8    assemble-part(ur5e, mcp, assembly-board-v1)                 ur5e      MCP
```

**Result: SOLVED.** No new predicates, no new actions, no LLM. The failure only changed which resources are available — the automaton vocabulary was sufficient.

**Step 4 — Translate to DAG patch:** Each PDDL action maps 1:1 to a tools.json function. The translator produces 8 task nodes with ur5e as resource_jid, replacing the original xarm6 tasks.

---

### Scenario B: LLM required (displaced part, deadlock)

**Failure:** xarm6 fails `assemble_part` for SG. SG displaced to ur5e-workspace. ur5e holding MCP. Deadlock.

```
System state at failure:
  xarm6: idle (gripper released)
  ur5e:  positioned, holding MCP (part: in_transit)
  SG:    physically at ur5e-workspace — NO CATALOGED STATE
  MCP:   in_transit (held by ur5e)
```

**Step 1 — Compile domain:** Same 5-action domain (identical).

**Step 2 — Compile problem:** SG has no state predicate. MCP is `in_transit`.

**Step 3 — pyperplan BFS:** NO SOLUTION. `(part-state-verified sg)` is unreachable — no action can fire on SG because no precondition matches it.

**Step 4 — Gap analysis → Layer 2:** "SG has no representable state."

**Step 5 — LLM extends domain:** Adds `(part-state-displaced ?p)`, `locate-and-recover` action. Sets `(part-state-displaced sg)` and `(part-at sg ur5e-workspace)` in init.

**Step 6 — pyperplan BFS on extended domain:** SOLVED in 10 steps (see Phase D above).

---

### Comparison summary

| Aspect | Scenario A (static) | Scenario B (LLM) |
|---|---|---|
| **Failure type** | Resource offline | Part displaced to unknown state |
| **All states in Q?** | Yes — all entities in cataloged states | No — SG in uncataloged state |
| **Domain changes** | None | +1 predicate, +1 action |
| **Problem changes** | Remove `resource-available` | +2 init facts (displaced, location) |
| **LLM needed?** | No | Yes — to interpret and name the new state |
| **Solution length** | 8 steps | 10 steps (+2 for recovery) |
| **Latency** | ~0ms (compiled + BFS) | ~2-3s (LLM call + BFS) |
| **Guarantee** | Same as forward path | Same (after LLM output validated by BFS + CCA) |

The dividing line is clear: **if every entity is in a state the catalog declares, compiled PDDL suffices. If any entity is in a state the catalog has no name for, the LLM must extend the vocabulary.**

---

## 9. How BFS Works — Detailed pyperplan Walkthrough

### What BFS does

Breadth-First Search (BFS) is the simplest complete planning algorithm. Given a PDDL domain and problem, it:

1. **Grounds** all actions — substitutes every possible combination of objects into action parameters, creating a finite set of concrete actions
2. **Starts** from the initial state (the set of true predicates in `:init`)
3. **Explores** by trying every applicable grounded action, generating successor states
4. **Queues** successors in FIFO order (breadth-first — shorter plans found first)
5. **Stops** when a state satisfying all `:goal` predicates is found
6. **Returns** the action sequence that led to the goal state

BFS guarantees the **shortest plan** (fewest actions) because it explores all n-step plans before any (n+1)-step plan.

### Grounding phase

Before search begins, pyperplan creates every concrete action instance. For Scenario A (rerouting), with 2 resources, 2 parts, and 3 locations:

```
move-to-pick-location has 3 parameters (resource × part × location):
  → 2 × 2 × 3 = 12 grounded instances

pick-part has 3 parameters:
  → 2 × 2 × 3 = 12 grounded instances

move-loaded-to-destination has 3 parameters:
  → 2 × 2 × 3 = 12 grounded instances

place-part has 3 parameters:
  → 2 × 2 × 3 = 12 grounded instances

assemble-part has 3 parameters:
  → 2 × 2 × 3 = 12 grounded instances

unlock-mcp-assembly has 0 parameters:
  → 1 grounded instance

Total: 61 grounded actions
```

Most of these will never fire — their preconditions will never be satisfied (e.g., `pick-part(xarm6, sg, assembly-board-v1)` requires SG to be at assembly-board-v1, which is the destination, not the origin). BFS only expands actions whose preconditions match the current state.

### BFS trace for Scenario A (xarm6 offline, reroute to ur5e)

State representation: each state is a **set of true predicates**. Below, we abbreviate predicates for readability:

```
Abbreviations:
  ri(R) = robot-state-idle(R)        rap(R) = robot-state-at-pick(R)
  rpk(R) = robot-state-picked(R)     rps(R) = robot-state-positioned(R)
  pp(P) = part-state-printed(P)      pig(P) = part-state-in-gripper(P)
  pit(P) = part-state-in-transit(P)  pv(P) = part-state-verified(P)
  pa(P,L) = part-at(P,L)            h(R,P) = holding(R,P)
  cr(R,L) = can-reach(R,L)          ra(R) = resource-available(R)
  aa(P) = assembly-allowed(P)
```

**Initial state S₀:**
```
{ ri(ur5e), ra(ur5e), cr(ur5e,mk4-1), cr(ur5e,mk4-2), cr(ur5e,asm),
  ri(xarm6), cr(xarm6,mk4-1), cr(xarm6,asm),     ← NOTE: no ra(xarm6)
  pp(sg), pa(sg,mk4-1), pp(mcp), pa(mcp,mk4-2),
  aa(sg) }
```

**Goal:**
```
{ pv(sg), pa(sg,asm), pv(mcp), pa(mcp,asm) }
```

---

**Iteration 0 — expand S₀:**

Queue: `[S₀]`
Visited: `{}`

Pop S₀. Check goal: NO (no `pv` predicates). Find applicable actions:

| # | Action | Why applicable |
|---|--------|---------------|
| 1 | `move-to-pick-location(ur5e, sg, mk4-1)` | ri(ur5e) ✓, ra(ur5e) ✓, cr(ur5e,mk4-1) ✓, pa(sg,mk4-1) ✓, pp(sg) ✓ |
| 2 | `move-to-pick-location(ur5e, mcp, mk4-2)` | ri(ur5e) ✓, ra(ur5e) ✓, cr(ur5e,mk4-2) ✓, pa(mcp,mk4-2) ✓, pp(mcp) ✓ |
| — | `move-to-pick-location(xarm6, ...)` | BLOCKED: no ra(xarm6) |

All other actions (pick, move-loaded, place, assemble) also blocked — wrong robot states or missing preconditions.

Generate successors:
- S₁ = apply action 1 to S₀: `ri(ur5e)` removed, `rap(ur5e)` added
- S₂ = apply action 2 to S₀: `ri(ur5e)` removed, `rap(ur5e)` added (for mcp)

Queue: `[S₁, S₂]`
Visited: `{S₀}`

---

**Iteration 1 — expand S₁:**

Pop S₁ (ur5e at_pick, near SG at mk4-1). Check goal: NO. Find applicable actions:

| # | Action | Why applicable |
|---|--------|---------------|
| 1 | `pick-part(ur5e, sg, mk4-1)` | rap(ur5e) ✓, cr(ur5e,mk4-1) ✓, pa(sg,mk4-1) ✓, pp(sg) ✓ |

Generate successor:
- S₃ = apply pick-part: `rap(ur5e)` → `rpk(ur5e)`, `pp(sg)` → `pig(sg)`, add `h(ur5e,sg)`, remove `pa(sg,mk4-1)`

Queue: `[S₂, S₃]`
Visited: `{S₀, S₁}`

---

**Iteration 2 — expand S₂:**

Pop S₂ (ur5e at_pick, near MCP at mk4-2). Find applicable:

| # | Action | Why applicable |
|---|--------|---------------|
| 1 | `pick-part(ur5e, mcp, mk4-2)` | rap(ur5e) ✓, pa(mcp,mk4-2) ✓, pp(mcp) ✓ |

Generate S₄ (ur5e picked mcp).

Queue: `[S₃, S₄]`

---

**Iterations 3-4 — continue expanding:**

S₃ (ur5e picked sg) → `move-loaded-to-destination(ur5e, sg, asm)` → S₅ (ur5e positioned, sg in_transit)
S₄ (ur5e picked mcp) → `move-loaded-to-destination(ur5e, mcp, asm)` → S₆ (ur5e positioned, mcp in_transit)

---

**Iteration 5 — expand S₅ (ur5e positioned with SG):**

Applicable actions:
| # | Action | Why applicable |
|---|--------|---------------|
| 1 | `assemble-part(ur5e, sg, asm)` | rps(ur5e) ✓, h(ur5e,sg) ✓, pit(sg) ✓, cr(ur5e,asm) ✓, **aa(sg) ✓** |
| 2 | `place-part(ur5e, sg, asm)` | rps(ur5e) ✓, h(ur5e,sg) ✓, pit(sg) ✓, cr(ur5e,asm) ✓ (no aa check) |
| 3 | `place-part(ur5e, sg, mk4-1)` | Same but different location |
| ... | other place-part locations | Also applicable |

BFS explores ALL of these. But `assemble-part` leads toward the goal (produces `pv(sg)`), while `place-part` leads backward (returns sg to `printed`). BFS will find the `assemble-part` path first because it is shorter.

S₇ = assemble-part(ur5e, sg, asm): `pv(sg)`, `pa(sg,asm)`, ur5e → idle.

---

**Iteration 6 — expand S₆ (ur5e positioned with MCP):**

Applicable:
| # | Action | Why |
|---|--------|-----|
| 1 | `assemble-part(ur5e, mcp, asm)` | BLOCKED: **aa(mcp) is FALSE** — safety constraint |
| 2 | `place-part(ur5e, mcp, ...)` | Applicable (no safety check) → returns MCP to printed |

So this branch can only stage MCP and start over. It's a longer path — BFS will find the SG-first path from S₇ before this becomes relevant.

---

**Iteration 7 — expand S₇ (SG verified, ur5e idle):**

`unlock-mcp-assembly()` is now applicable: `pv(sg)` ✓. Produces S₈ with `aa(mcp)`.

Then: move-to-pick(ur5e, mcp, mk4-2) → pick(ur5e, mcp, mk4-2) → move-loaded(ur5e, mcp, asm) → assemble(ur5e, mcp, asm).

Final state satisfies goal: `pv(sg)`, `pa(sg,asm)`, `pv(mcp)`, `pa(mcp,asm)`. **DONE.**

---

### BFS properties relevant to recovery planning

| Property | Implication |
|---|---|
| **Complete** | If a solution exists, BFS will find it. If it returns no solution, none exists. |
| **Optimal** | BFS finds the shortest plan (fewest actions). No unnecessary detours. |
| **No heuristic needed** | Unlike A* or FF, BFS requires no domain-specific heuristic. Important because recovery domains are LLM-generated — no time to tune heuristics. |
| **Exponential worst case** | O(b^d) where b = branching factor, d = solution depth. For our domains (~60 grounded actions, ~10 step solutions), this is tractable (<1ms). |
| **Deterministic** | Same domain + problem always produces the same plan. Reproducible for testing and debugging. |

### Why BFS works for manufacturing recovery

The state spaces are small:
- ~2-5 resources × ~2-10 parts × ~5-20 locations = hundreds of objects
- ~50-200 grounded actions after instantiation
- Solution depths of 5-15 steps
- Total reachable states: thousands, not millions

BFS explores this in milliseconds. More sophisticated planners (FF, LAMA) would be faster on larger problems but add complexity with no benefit here. The bottleneck in Layer 2 is the LLM call (~2-3s), not the BFS search (~0ms).

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
