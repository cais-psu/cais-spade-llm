# TSS-Based LLM Bridge v3: Implementation Blueprint

> **Design document for the next-generation recovery bridge integrating TSS-inspired
> supervisory control with advanced LLM techniques.**
>
> Author: Auto-generated from TSS paper analysis + codebase exploration
> Date: 2026-03-23
> Status: Design — not yet implemented

---

## Context

**Problem:** The v2 Universal Repair Session (`universal_repair_session.py`) uses a
ReAct-style multi-turn loop where the LLM proposes repair programs and a
deterministic two-layer validator checks them. Three weaknesses:

1. **Parse fragility** — LLM returns raw JSON text; malformed output wastes turns
2. **Flat rejection feedback** — LLM has no structured view of which states are
   reachable, safe, or blocked
3. **Redundant computation** — Every turn rebuilds the full prompt (~17k chars) and
   reruns full validation (Layer A + Layer B) from scratch

**Solution:** A TSS-inspired v3 bridge integrating three LLM techniques:

| Technique | What It Does |
|---|---|
| **(A) Constrained decoding** | OpenAI structured outputs enforce `RepairProgram` schema at generation time — eliminates parse errors |
| **(B) CoT with state projection** | Mandatory `reasoning` field forces step-by-step state-transition analysis before proposing actions |
| **(C) Projection tool-use** | LLM calls `project_primitive_sequence()` to simulate primitives and see projected snapshots before committing |

Plus three TSS-inspired structural improvements:

| Improvement | TSS Origin |
|---|---|
| **State classification feedback** | Y / G\Y / X\G partitioning (Thm 3.1) |
| **Model-delta prompts** | Model delta Δ (Def 4.1) — send only what changed |
| **Incremental validation cache** | ITSS/GTSS (Alg 11-12) — reuse synthesis artifacts |

**TSS paper:** Thuijsman & Reniers (2022), "Transformational Supervisor Synthesis
for Evolving Systems", *Discrete Event Dynamic Systems* 32:317-358.

---

## 1. Why This Is TSS-Inspired

### 1.1 Formal Mapping

| TSS Concept | Paper Reference | v3 Bridge Mapping |
|---|---|---|
| State space X | Def 2.1 | Product graph states in `compute_winning_set()` |
| Supervisor states Y | Def 3.4 | Winning set W — reachable + safe + can reach accepting |
| Good states G | Def 3.2 | States not violating any safety DFA |
| Bad states X\G | Def 3.2 | States where at least one DFA reaches its violation state |
| G\Y (safe unreachable) | Implicit in Thm 3.1 | Safe states the program doesn't reach |
| Model delta Δ | Def 4.1 | Diff between Turn N and Turn N+1 repair programs |
| Incremental synthesis GTSS | Alg 12 | `ValidationCache` — reuse FSA + winning set across turns |
| Controllable events Σ_c | Def 2.3 | Primitive actions the LLM chooses to invoke |
| Uncontrollable events Σ_u | Def 2.3 | Environment events (failures, sensor readings) |
| Nonblocking | Def 2.5 | Continuation viability check in Layer B |

### 1.2 Key TSS Insight Applied

TSS stores **G** (good states) alongside **Y** (supervisor states) so that when the
plant model changes by Δ, only the affected partition needs recomputation. The v3
bridge applies this principle:

- **Turn N:** Full validation → cache FSA, product graph, winning set W, state classification
- **Turn N+1:** Diff repair program → if delta is small, reuse cached Layer B artifacts;
  if delta is large, fall back to full validation

This is analogous to TSS's GTSS algorithm (Algorithm 12), which partitions Δ into
"free" adaptations (Δ° — no effect on Y,G) and "expensive" adaptations (Δ× — require
reachability search), processing Δ° first.

---

## 2. Architecture: Target Session Flow

```
Turn 1 (full prompt):
  build_recovery_context()
  → build_full_tss_prompt(context, primitives, schema, obligations)
  → ask_llm_structured(response_format=SCHEMA, tools=[project_primitive_sequence])
  → [tool loop: LLM calls project_primitive_sequence → gets snapshot → continues]
  → parse structured response (reasoning + repair_program)
  → validate_repair_program()          [full Layer A + Layer B]
  → classify_product_states()          → TSSStateClassification
  → cache_validation_artifacts()       → ValidationCache
  → if rejected: extract constraints, loop

Turn 2+ (delta prompt):
  build_recovery_context()
  → build_delta_prompt(context, tss_classification, constraints, last_rejected)
    [primitives: "unchanged from Turn 1", schema: "unchanged from Turn 1"]
    [adds: Model Delta section with Y/G\Y/X\G + blocked transitions + suggestions]
  → ask_llm_structured(response_format=SCHEMA, tools=[project_primitive_sequence])
  → [tool loop]
  → parse structured response
  → diff_programs(new_program, cached) → ProgramDelta
  → if can_reuse_layer_b(delta):
      validate Layer A only, reuse cached Layer B
    else:
      full validation
  → update classification + cache
  → if rejected: loop
```

### Current vs Target Comparison

| Aspect | v2 (Current) | v3 (Target) |
|---|---|---|
| LLM output format | Raw JSON text, `json.loads()` | Structured output (schema-enforced) |
| Reasoning | Implicit in `rationale` field | Explicit `reasoning` object with CoT |
| State simulation | None — LLM guesses | `project_primitive_sequence` tool |
| Rejection feedback | Flat text constraints | Y/G\Y/X\G state classification |
| Turn 2+ prompt | Full rebuild (~17k chars) | Delta prompt (~6k chars) |
| Validation | Full Layer A+B every turn | Incremental (cache Layer B artifacts) |

---

## 3. LLM Technique A: Constrained Decoding

### 3.1 Current State

`universal_repair_session.py:520` calls `ask_llm(prompt, with_functions=False)` which
returns raw text. The response is parsed via `json.loads()`. If the JSON is malformed,
the turn is wasted.

### 3.2 Target

Use OpenAI's `response_format` parameter with `type: "json_schema"` to enforce the
exact response structure at generation time.

### 3.3 Schema (Discriminated Union)

```json
{
  "name": "repair_turn_response",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string",
        "enum": ["observe", "repair_program"]
      },
      "reasoning": {
        "type": "object",
        "description": "Mandatory chain-of-thought reasoning before proposing actions",
        "properties": {
          "current_state_analysis": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One entry per resource: current state, held parts, location"
          },
          "goal_gap_analysis": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One entry per obligation: what must change"
          },
          "transition_plan": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "step": {"type": "string"},
                "resource": {"type": "string"},
                "from_state": {"type": "string"},
                "to_state": {"type": "string"},
                "primitive": {"type": "string"},
                "safety_note": {"type": "string"}
              },
              "required": ["step", "resource", "primitive"],
              "additionalProperties": false
            },
            "description": "Step-by-step: each primitive, projected state, safety note"
          },
          "safety_check": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One entry per safety rule: why plan does not violate it"
          }
        },
        "required": ["current_state_analysis", "goal_gap_analysis",
                      "transition_plan", "safety_check"],
        "additionalProperties": false
      },
      "function_defs": {
        "type": ["array", "null"],
        "description": "SynthesizedTaskFn definitions (null for observe type)"
      },
      "steps": {
        "type": ["array", "null"],
        "description": "RepairStep sequence (null for observe type)"
      },
      "success_conditions": {
        "type": ["array", "null"],
        "description": "Reachability conditions (null for observe type)"
      },
      "rationale": {
        "type": ["string", "null"],
        "description": "One-line summary"
      },
      "observe_request": {
        "type": ["object", "null"],
        "description": "For observe type: {resource_jid, primitive, params, store_as}"
      }
    },
    "required": ["type", "reasoning"],
    "additionalProperties": false
  }
}
```

### 3.4 Integration Point

New method in `llm_agent.py` (insert after `ask_llm()` at ~line 264):

```python
async def ask_llm_structured(
    self,
    prompt: str,
    *,
    response_format: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    tool_executor: Callable[[str, dict], Any] | None = None,
    max_tool_rounds: int = 3,
) -> dict[str, Any]:
    """Call LLM with structured output + optional tool use."""
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_tool_rounds + 1):
        kwargs = {"model": self._model, "messages": messages,
                  "response_format": response_format}
        if tools:
            kwargs["tools"] = tools
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if choice.message.tool_calls and tool_executor:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                result = tool_executor(tc.function.name,
                                       json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})
            continue
        return json.loads(choice.message.content)
    raise RuntimeError("Exceeded max tool rounds")
```

---

## 4. LLM Technique B: Chain-of-Thought with State Projection

### 4.1 Prompt Instruction

Added to system prompt in `_build_repair_program_prompt()`:

```
## Reasoning Requirements

Before proposing any actions, you MUST fill the `reasoning` object completely:

1. `current_state_analysis` — List each resource's current state, held parts,
   and location. Example: "ur5e: idle, holding mcp_01, at feeder_mcp"

2. `goal_gap_analysis` — For each obligation, state what must change to satisfy it.
   Example: "mcp_01 must go from 'picked' to 'assembled' — requires place + assemble"

3. `transition_plan` — For each primitive you plan to use, show:
   - Which resource executes it
   - What state it transitions FROM and TO
   - The primitive name and key parameters
   - A safety note explaining why this doesn't violate any safety rule
   You may call `project_primitive_sequence` to verify intermediate states.

4. `safety_check` — For each safety rule in the obligations, explain why your
   proposed plan does not violate it. Be specific about resource locations
   and timing.
```

### 4.2 Validator Use of CoT

The `reasoning.transition_plan` can be cross-checked against `function_defs[].primitive_program`:
- If the transition plan mentions a primitive not in the program → warning
- If the transition plan's projected states differ from validation's projection → diagnostic signal
- This is informational, not blocking — the deterministic validator remains the authority

---

## 5. LLM Technique C: Projection Tool-Use

### 5.1 Tool Definition

```json
{
  "type": "function",
  "function": {
    "name": "project_primitive_sequence",
    "description": "Simulate a sequence of primitives on a resource and return the projected state snapshot after all steps. Use this to verify your plan before committing. Returns {is_valid, projected_snapshot, error}.",
    "parameters": {
      "type": "object",
      "properties": {
        "resource_jid": {
          "type": "string",
          "description": "The resource JID to simulate on (e.g., 'ur5e@factory')"
        },
        "steps": {
          "type": "array",
          "description": "Ordered list of primitives to simulate",
          "items": {
            "type": "object",
            "properties": {
              "primitive": {"type": "string", "description": "Primitive name"},
              "params": {"type": "object", "description": "Primitive parameters"}
            },
            "required": ["primitive", "params"]
          }
        }
      },
      "required": ["resource_jid", "steps"]
    }
  }
}
```

### 5.2 Execution Handler

New method in `UniversalRepairSessionMixin` (`universal_repair_session.py`):

```python
def _execute_projection_tool(
    self,
    tool_name: str,
    arguments: dict[str, Any],
    recovery_context: RecoveryContext,
) -> dict[str, Any]:
    """Execute a projection tool call from the LLM."""
    if tool_name != "project_primitive_sequence":
        return {"error": f"Unknown tool: {tool_name}"}
    resource_jid = arguments["resource_jid"]
    steps = arguments["steps"]
    catalog = recovery_context.available_primitives.get(resource_jid, [])
    snapshot = recovery_context.resource_snapshots.get(resource_jid, {})
    if not catalog:
        return {"error": f"No primitive catalog for {resource_jid}"}
    is_valid, projected, error = validate_and_project_steps(
        steps, catalog, snapshot
    )
    return {
        "is_valid": is_valid,
        "projected_snapshot": projected,
        "error": error,
    }
```

Uses existing `validate_and_project_steps()` from `primitive_semantics.py:1091` —
**no changes needed** to that function.

### 5.3 Tool Loop

The tool loop runs inside `ask_llm_structured()`. The LLM can call
`project_primitive_sequence` up to 3 times per turn. Each call:

1. LLM emits a `tool_calls` response
2. Session executes the tool via `_execute_projection_tool()`
3. Result is appended to messages as a `tool` role message
4. LLM continues with the enriched context
5. Final response is the structured `repair_turn_response`

---

## 6. TSS Improvement 1: State Classification Feedback

### 6.1 Dataclass

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class TSSStateClassification:
    """TSS-style Y / G\\Y / X\\G state classification.

    Ref: Thuijsman & Reniers (2022), Theorem 3.1
    """
    winning_states: list[dict[str, Any]] = field(default_factory=list)
    """Y: Reachable + safe + can reach an accepting state."""

    safe_unreachable: list[dict[str, Any]] = field(default_factory=list)
    """G\\Y: Safe (no DFA violation) but not reachable from the current
    program's projected path."""

    unsafe_states: list[dict[str, Any]] = field(default_factory=list)
    """X\\G: At least one safety DFA reaches its violation state."""

    blocked_transitions: list[dict[str, Any]] = field(default_factory=list)
    """Transitions pruned during forward BFS due to safety violations.
    Each entry: {from_state, to_state, event, violated_rule_ids}."""

    suggested_adaptations: list[str] = field(default_factory=list)
    """Human-readable hints derived from the classification."""
```

### 6.2 How to Compute

Extend `compute_winning_set()` in `plan_safety_validator.py:471`:

```python
# Current (line ~581): when a state violates a safety DFA during forward BFS:
if violates:
    continue  # skip this state

# Target: also track the rejected transition:
if violates:
    rejected_transitions.append({
        "from_state": current_state,
        "to_state": next_state,
        "event": event_label,
        "violated_rule_ids": [rule_ids[i] for i, q in enumerate(q_vec)
                              if q == dfas[rule_ids[i]].get("violation_state")],
    })
    continue
```

Return `rejected_transitions` + `all_explored_states` in the result dict.

Classification function (`tss_state_classifier.py`):

```python
def classify_product_states(
    winning_set_result: dict[str, Any],
) -> TSSStateClassification:
    W = winning_set_result["W"]
    state_meta = winning_set_result["state_meta"]
    rejected = winning_set_result.get("rejected_transitions", [])

    all_states = set(state_meta.keys())
    violated_states = {r["to_state"] for r in rejected}

    winning = [state_meta[s] for s in W]
    safe_unreachable = [state_meta[s] for s in (all_states - W - violated_states)]
    unsafe = [state_meta[s] for s in violated_states if s in state_meta]

    return TSSStateClassification(
        winning_states=winning,
        safe_unreachable=safe_unreachable,
        unsafe_states=unsafe,
        blocked_transitions=rejected,
        suggested_adaptations=derive_suggested_adaptations(...),
    )
```

### 6.3 Prompt Rendering

```python
def classification_to_prompt_section(cls: TSSStateClassification) -> str:
    lines = ["## State Classification (TSS-inspired)"]
    lines.append(f"- Winning (Y): {len(cls.winning_states)} states reachable and safe")
    if cls.safe_unreachable:
        lines.append(f"- Safe unreachable (G\\Y): {len(cls.safe_unreachable)} states")
        for s in cls.safe_unreachable[:3]:  # show top 3
            lines.append(f"  - {s}")
    if cls.unsafe_states:
        lines.append(f"- Unsafe (X\\G): {len(cls.unsafe_states)} states")
    if cls.blocked_transitions:
        lines.append(f"- Blocked transitions: {len(cls.blocked_transitions)}")
        for bt in cls.blocked_transitions[:3]:
            lines.append(f"  - {bt['event']} blocked by {bt['violated_rule_ids']}")
    if cls.suggested_adaptations:
        lines.append("\n### Suggested Fix")
        for s in cls.suggested_adaptations:
            lines.append(f"- {s}")
    return "\n".join(lines)
```

---

## 7. TSS Improvement 2: Model-Delta Prompts

### 7.1 Turn 1 Prompt Structure (~17k chars)

```
[System instruction: full recovery planner role description]
[Turn budget: "Turn 1/8"]
[Recovery context: resources, parts, obligations]     ← always fresh
[Available primitives per resource]                    ← ~8k chars, STATIC
[RepairProgram JSON schema description]                ← ~4k chars, STATIC
[Library candidates]                                   ← if available
[Reasoning instruction: fill reasoning before proposing]
```

### 7.2 Turn 2+ Delta Prompt (~6k chars)

```
[System instruction: "Continuing repair session. Schema and primitives
 unchanged from Turn 1."]
[Turn budget: "Turn 2/8"]
[Recovery context: resources, parts, obligations]     ← always fresh
[Primitive catalog: "Unchanged from Turn 1."]         ← 1 line
[Schema: "Unchanged from Turn 1."]                    ← 1 line
[Model Delta: TSS state classification + suggestions]  ← NEW, ~2k chars
[Discovered constraints from rejections]
[Last rejected proposal (abbreviated)]
[Instruction: fix the issues identified in Model Delta]
```

### 7.3 Implementation

Refactor `_build_repair_program_prompt()` (line 842):

```python
def _build_repair_program_prompt(
    self,
    recovery_context: RecoveryContext,
    session_state: dict,
    *,
    tss_classification: TSSStateClassification | None = None,
) -> str:
    turn_idx = session_state["turn_index"]
    is_delta = turn_idx > 1 and tss_classification is not None

    parts = []
    if is_delta:
        parts.append("You are continuing a repair session. "
                      "Schema and primitives are unchanged from Turn 1.")
    else:
        parts.append(FULL_SYSTEM_INSTRUCTION)

    parts.append(f"Turn {turn_idx}/{session_state['max_turns']}.")
    parts.append(recovery_context_to_prompt_dict(recovery_context))

    if is_delta:
        parts.append("Primitive catalog: unchanged from Turn 1.")
        parts.append("Output schema: unchanged from Turn 1.")
        parts.append(classification_to_prompt_section(tss_classification))
    else:
        parts.append(self._render_primitive_catalogs(recovery_context))
        parts.append(REPAIR_PROGRAM_SCHEMA_DESCRIPTION)

    # Always include constraints and last rejected
    if session_state["discovered_constraints"]:
        parts.append(self._render_constraints(session_state))
    if session_state.get("last_rejected_proposal"):
        parts.append(self._render_last_rejected(session_state))

    parts.append(REASONING_INSTRUCTION)
    return "\n\n".join(parts)
```

**Token savings:** ~60% reduction on Turn 2+ (eliminate ~11k chars of repeated
catalog + schema).

---

## 8. TSS Improvement 3: Incremental Validation Cache

### 8.1 Cache Structure

```python
@dataclass
class ValidationCache:
    """Cached validation artifacts from a previous turn."""
    turn_index: int
    program_fingerprint: str
    fn_fingerprints: dict[str, str]   # fn_name → primitive_fingerprint
    projected_nodes: list[dict[str, Any]]
    compiled_fsa: dict[str, Any] | None
    winning_set_result: dict[str, Any] | None
    fn_projected_snapshots: dict[str, dict[str, Any]]

@dataclass
class ProgramDelta:
    """Diff between current and cached repair programs."""
    is_identical: bool = False
    modified_fns: list[str] = field(default_factory=list)
    new_fns: list[str] = field(default_factory=list)
    removed_fns: list[str] = field(default_factory=list)
    steps_changed: bool = False
```

### 8.2 Delta Logic

```python
def compute_program_fingerprint(program: RepairProgram) -> str:
    """SHA256 of the serialized program (excluding reasoning)."""
    d = repair_program_to_dict(program)
    d.pop("reasoning", None)
    d.pop("rationale", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def diff_programs(program: RepairProgram, cache: ValidationCache) -> ProgramDelta:
    new_fp = compute_program_fingerprint(program)
    if new_fp == cache.program_fingerprint:
        return ProgramDelta(is_identical=True)

    new_fn_names = {fn.name for fn in program.function_defs}
    cached_fn_names = set(cache.fn_fingerprints.keys())

    modified, new_fns = [], []
    for fn in program.function_defs:
        fn_fp = compute_primitive_fingerprint(fn)
        if fn.name in cache.fn_fingerprints:
            if fn_fp != cache.fn_fingerprints[fn.name]:
                modified.append(fn.name)
        else:
            new_fns.append(fn.name)

    removed = list(cached_fn_names - new_fn_names)
    steps_changed = ... # compare step sequences

    return ProgramDelta(
        modified_fns=modified,
        new_fns=new_fns,
        removed_fns=removed,
        steps_changed=steps_changed,
    )

def can_reuse_layer_b(delta: ProgramDelta) -> bool:
    """Layer B can be reused if no structural changes to functions or steps."""
    return delta.is_identical or (
        not delta.new_fns
        and not delta.removed_fns
        and not delta.steps_changed
    )
```

Uses existing `compute_primitive_fingerprint()` from `mutation_types.py:441`.

### 8.3 Session Integration

In `run_universal_repair_session()`, around the validation call:

```python
# Before validation:
if validation_cache and turn_idx > 1:
    delta = diff_programs(program, validation_cache)
    if can_reuse_layer_b(delta):
        # Run Layer A only
        validated = validate_repair_program(
            program, ..., skip_layer_b=True,
            cached_layer_b=validation_cache.winning_set_result,
        )
    else:
        validated = validate_repair_program(program, ...)
else:
    validated = validate_repair_program(program, ...)

# After validation (always):
validation_cache = ValidationCache(
    turn_index=turn_idx,
    program_fingerprint=compute_program_fingerprint(program),
    fn_fingerprints={fn.name: compute_primitive_fingerprint(fn)
                     for fn in program.function_defs},
    projected_nodes=validated.validation_artifacts.get("projected_nodes", []),
    compiled_fsa=validated.validation_artifacts.get("compiled_fsa"),
    winning_set_result=validated.validation_artifacts.get("winning_set_result"),
    fn_projected_snapshots=validated.validation_artifacts.get("fn_snapshots", {}),
)
```

---

## 9. Example Flow: Turn 1 → Reject → Turn 2 → Accept

### Scenario

- UR5e holds mcp_01 (state: picked, gripper: closed)
- xArm6 is idle (empty gripper)
- **Goal obligation:** mcp_01.state must reach "assembled"
- **Safety rule:** mutex_workstation — at most one robot at workstation at a time

### Turn 1: Full Prompt + Projection Tool + Rejection

**LLM calls projection tool** (within the structured output turn):
```json
{
  "name": "project_primitive_sequence",
  "arguments": {
    "resource_jid": "ur5e@factory",
    "steps": [
      {"primitive": "move_to", "params": {"target": "workstation_a"}},
      {"primitive": "place_part", "params": {"part_name": "mcp_01"}}
    ]
  }
}
```

**Tool result:**
```json
{
  "is_valid": true,
  "projected_snapshot": {
    "current_state": "idle",
    "held_part": null,
    "gripper_state": "open",
    "current_location": "workstation_a"
  },
  "error": null
}
```

**LLM structured response:**
```json
{
  "type": "repair_program",
  "reasoning": {
    "current_state_analysis": [
      "ur5e: idle, holding mcp_01, gripper closed",
      "xarm6: idle, empty gripper, at home"
    ],
    "goal_gap_analysis": [
      "mcp_01 is 'picked' but needs 'assembled' — requires place + assemble at workstation"
    ],
    "transition_plan": [
      {
        "step": "1",
        "resource": "ur5e@factory",
        "from_state": "holding mcp_01",
        "to_state": "at workstation",
        "primitive": "move_to(workstation_a)",
        "safety_note": "xarm6 is at home, not at workstation — mutex safe"
      },
      {
        "step": "2",
        "resource": "ur5e@factory",
        "from_state": "at workstation",
        "to_state": "mcp_01 placed",
        "primitive": "place_part(mcp_01)",
        "safety_note": "xarm6 still at home"
      }
    ],
    "safety_check": [
      "mutex_workstation: Only ur5e will be at workstation. xarm6 stays at home. No violation."
    ]
  },
  "function_defs": [{
    "name": "deliver_mcp",
    "intent": "Move to workstation and place mcp_01",
    "resource_constraints": {"resource_type": "ur5e"},
    "inputs": {},
    "preconditions": {"held_part": {"equals": "mcp_01"}},
    "effects": {"held_part": {"set": null}, "current_location": {"set": "workstation_a"}},
    "primitive_program": [
      {"primitive": "move_to", "params": {"target": "workstation_a"}},
      {"primitive": "place_part", "params": {"part_name": "mcp_01"}}
    ],
    "expected_post_state": {"held_part": null, "current_location": "workstation_a"}
  }],
  "steps": [
    {"kind": "call_function", "payload": {"function_name": "deliver_mcp", "resource_jid": "ur5e@factory", "args": {}}}
  ],
  "success_conditions": [
    {"entity_kind": "part", "entity": "mcp_01", "field": "state", "expected": "placed"}
  ],
  "rationale": "Deliver mcp_01 to workstation for assembly."
}
```

**Validation:** REJECTED
- Layer B, obligation_discharge: "obligation not discharged: mcp_01.state must
  reach assembled, but program only reaches placed"

**TSS State Classification:**
```
Y (winning):         2 states — (initial) and (after move_to)
G\Y (safe unreachable): 1 state — (assembled, q_accept, sig_final)
  → The 'assembled' state exists in the FSA and is safe, but your program
    stops at 'placed'. Need assemble_part to reach it.
X\G (unsafe):        0 states
Blocked transitions: none
Suggested: "Add assemble_part(mcp_01) after place_part to reach assembled state."
```

### Turn 2: Delta Prompt + Acceptance

**Delta prompt** (abbreviated — no primitive catalog, no schema):
```
You are continuing a repair session. Schema and primitives unchanged from Turn 1.
Turn 2/8 (6 turns remaining).

## Current System State
{"resources": {"ur5e@factory": {"current_state": "idle", "held_part": "mcp_01", ...}, ...},
 "obligations": [{"type": "goal", "entity": "mcp_01", "expected": "assembled"}, ...]}

Primitive catalog: unchanged from Turn 1.
Output schema: unchanged from Turn 1.

## State Classification (TSS-inspired)
- Winning (Y): 2 states reachable and safe
- Safe unreachable (G\Y): 1 state
  - The 'assembled' goal state is reachable in the FSA via assemble_part
    but your program does not include it
- Unsafe (X\G): 0 states
- Blocked transitions: none

### Suggested Fix
- Add assemble_part(mcp_01) after place_part to reach the assembled state.

## Discovered Constraints
1. obligation not discharged: mcp_01.state must reach assembled

## Last Rejected Proposal
deliver_mcp: move_to(workstation_a) → place_part(mcp_01)
Reason: missing assembly step — reached 'placed' but needed 'assembled'
```

**LLM response** (adds assemble_part):
```json
{
  "type": "repair_program",
  "reasoning": {
    "current_state_analysis": ["ur5e: idle, holding mcp_01", "xarm6: idle, at home"],
    "goal_gap_analysis": ["Previous plan stopped at 'placed'. Must add assemble_part."],
    "transition_plan": [
      {"step": "1", "resource": "ur5e@factory", "primitive": "move_to(workstation_a)",
       "from_state": "holding mcp_01", "to_state": "at workstation",
       "safety_note": "xarm6 at home"},
      {"step": "2", "resource": "ur5e@factory", "primitive": "place_part(mcp_01)",
       "from_state": "at workstation", "to_state": "mcp_01 placed",
       "safety_note": "still alone at workstation"},
      {"step": "3", "resource": "ur5e@factory", "primitive": "assemble_part(mcp_01)",
       "from_state": "mcp_01 placed", "to_state": "mcp_01 assembled",
       "safety_note": "still alone at workstation"}
    ],
    "safety_check": ["mutex_workstation: Only ur5e at workstation throughout. Safe."]
  },
  "function_defs": [{
    "name": "deliver_and_assemble_mcp",
    "intent": "Move ur5e to workstation, place mcp_01, and assemble it",
    "resource_constraints": {"resource_type": "ur5e"},
    "inputs": {},
    "preconditions": {"held_part": {"equals": "mcp_01"}},
    "effects": {"held_part": {"set": null}, "current_location": {"set": "workstation_a"}},
    "primitive_program": [
      {"primitive": "move_to", "params": {"target": "workstation_a"}},
      {"primitive": "place_part", "params": {"part_name": "mcp_01"}},
      {"primitive": "assemble_part", "params": {"part_name": "mcp_01"}}
    ],
    "expected_post_state": {"held_part": null, "current_location": "workstation_a"}
  }],
  "steps": [
    {"kind": "call_function", "payload": {"function_name": "deliver_and_assemble_mcp",
     "resource_jid": "ur5e@factory", "args": {}}}
  ],
  "success_conditions": [
    {"entity_kind": "part", "entity": "mcp_01", "field": "state", "expected": "assembled"}
  ],
  "rationale": "Complete delivery and assembly of mcp_01 at workstation."
}
```

**Validation:** ACCEPTED
- Layer A: function valid, preconditions met, effects projected correctly
- Layer B: FSA compiled, safety DFAs satisfied, continuation viable, obligation discharged
- Risk: HIGH (new function, not in library)
- Requires operator approval: yes

**Incremental cache note:** Turn 2 has a different function name + extra primitive,
so `can_reuse_layer_b()` returns False → full Layer B runs. If the LLM had only
changed `success_conditions` without touching `function_defs`, Layer B would have
been skipped.

---

## 10. Files to Create

| File | Path | Responsibility |
|---|---|---|
| `tss_schemas.py` | `llm_bridge/tss_schemas.py` | JSON schema for constrained decoding + projection tool definition + response parsing |
| `tss_state_classifier.py` | `llm_bridge/tss_state_classifier.py` | `TSSStateClassification` dataclass + classify + render + suggest |
| `tss_validation_cache.py` | `llm_bridge/tss_validation_cache.py` | `ValidationCache` + `ProgramDelta` + fingerprinting + delta detection |

All paths relative to `cais_spade_llm/agents/intelligent_product/replanner/`.

---

## 11. Files to Modify

| File | Key Changes |
|---|---|
| `llm_agent.py` | Add `ask_llm_structured()` with `response_format` + `tools` + tool-call loop |
| `universal_repair_session.py` | Wire structured call, projection tool, TSS classification, delta prompts, validation cache into session loop |
| `repair_program_validator.py` | Add `validation_artifacts` to return; add `validate_repair_program_incremental()` |
| `mutation_types.py` | Add `reasoning` field to `RepairProgram`; add `validation_artifacts` to `ValidatedRepairProgram` |
| `plan_safety_validator.py` | Track `rejected_transitions` during forward BFS in `compute_winning_set()` |

See Section 7 of the plan file (`/home/jongh/.claude/plans/memoized-skipping-harp.md`)
for exact line numbers and function-level change descriptions.

---

## 12. Implementation Order

| Phase | What | Depends On | Risk |
|---|---|---|---|
| **1. Foundation** | Create 3 new files; add fields to `mutation_types.py` | Nothing | Low |
| **2. Constrained decoding (A)** | `ask_llm_structured()` in `llm_agent.py`; wire into session | Phase 1 | Low-Med |
| **3. Projection tool (C)** | `_execute_projection_tool()`; tool executor callback | Phase 2 | Low |
| **4. State classification** | Extend `compute_winning_set()`; classify in session | Phase 1 | Med |
| **5. Delta prompts (B)** | Refactor prompt builder into full/delta variants | Phase 4 | Low |
| **6. Validation cache** | `ValidationCache` in session; incremental validator | Phase 1 | Med |
| **7. Integration** | Wire all components together | Phases 2-6 | Low |

**Recommended start:** Phases 1-3 (constrained decoding + projection tool) are
highest impact and lowest risk. Phases 4-6 (TSS-specific) can follow incrementally.

---

## 13. Instructions for a New Claude Session

### Step 1: Read the TSS paper context

The TSS paper (Thuijsman & Reniers 2022) is about **incremental supervisor
re-synthesis**. You don't need to read the full paper. Key concepts:

- **State classification Y / G\Y / X\G**: Y = supervisor states (reachable + safe +
  can reach accepting). G\Y = good but unreachable. X\G = bad/unsafe.
- **Model delta Δ**: Instead of re-synthesizing from scratch when the model changes,
  TSS computes only the affected states.
- **GTSS (Algorithm 12)**: Groups atomic adaptations by type for efficiency.
  17-36% faster on industrial models.

### Step 2: Read the existing codebase (in this order)

1. `llm_bridge/mutation_types.py` — all dataclasses (`RepairProgram`,
   `SynthesizedTaskFn`, `RepairStep`, `ValidatedRepairProgram`)
2. `llm_bridge/universal_repair_session.py` — session loop
   (`run_universal_repair_session` at line 387,
   `_build_repair_program_prompt` at line 842)
3. `llm_bridge/primitive_semantics.py` — projection engine
   (`validate_and_project_steps` at line 1091)
4. `llm_bridge/repair_program_validator.py` — two-layer validation
   (`validate_repair_program` at line 48)
5. `llm_bridge/recovery_context_builder.py` — context building
   (`build_recovery_context` at line 38)
6. `central_controller/plan_safety_validator.py` — winning set computation
   (`compute_winning_set` at line 471)
7. `shared_information/llm_agent.py` — LLM interface (`ask_llm` at line 201)

### Step 3: Create three new files (Phase 1)

1. **`tss_schemas.py`** — JSON schema for OpenAI structured output +
   projection tool definition. See Section 3.3 and 5.1 for schemas.
2. **`tss_state_classifier.py`** — `TSSStateClassification` dataclass +
   `classify_product_states()` + `classification_to_prompt_section()`.
   See Section 6 for code.
3. **`tss_validation_cache.py`** — `ValidationCache` + `ProgramDelta` +
   `diff_programs()` + `can_reuse_layer_b()`. See Section 8 for code.

### Step 4: Add constrained decoding (Phase 2)

1. Add `ask_llm_structured()` to `llm_agent.py` — see Section 3.4.
2. Replace `ask_llm()` call in session loop (line ~520) with `ask_llm_structured()`.
3. Update response parsing to use `parse_structured_response()` from `tss_schemas`.

### Step 5: Add projection tool (Phase 3)

1. Add `_execute_projection_tool()` to `UniversalRepairSessionMixin` — see Section 5.2.
2. Pass it as `tool_executor` callback to `ask_llm_structured()`.
3. Limit to 3 tool calls per turn via `max_tool_rounds=3`.

### Step 6: Add state classification (Phase 4)

1. Extend `compute_winning_set()` to track `rejected_transitions` — see Section 6.2.
2. Add `classify_product_states()` call after Layer B validation in session loop.
3. Store classification in session state for delta prompts.

### Step 7: Add delta prompts (Phase 5)

1. Refactor `_build_repair_program_prompt()` — see Section 7.3.
2. Turn 1: current full prompt (no change).
3. Turn 2+: delta prompt with TSS classification section + model delta.

### Step 8: Add validation cache (Phase 6)

1. After each validation, build `ValidationCache` from artifacts.
2. Before validation on Turn 2+, call `diff_programs()` and `can_reuse_layer_b()`.
3. If reusable, skip Layer B; return cached result with only Layer A re-run.

### Step 9: Test

- Existing `test_case3_recovery_main.py` must still pass (backward compat)
- Add `test_tss_schemas.py` — schema accepts valid, rejects invalid responses
- Add `test_tss_state_classifier.py` — classification from mock winning set
- Add `test_tss_validation_cache.py` — fingerprinting + delta detection
- Add `test_projection_tool.py` — tool execution matches `validate_and_project_steps()`

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `response_format` + `tools` not supported simultaneously on gpt-5 | Techniques A+C can't coexist | Test early; fallback: two-phase call (tool phase → commit phase) |
| Schema too complex for `strict: true` mode | Parse errors return | Simplify with nullable fields; use `anyOf` for union types |
| Delta prompts cause LLM to lose context | Quality drops on Turn 2+ | Always include obligations + state snapshots; test with real scenarios |
| Validation cache returns stale results | False accept of bad program | Cache is optimization only; any doubt → fall back to full validation |
| Projection tool adds latency | Slower turns | Cap at 3 calls; tool is optional |
| State classification overhead | Slower Turn 1 | Classification is O(|states|) — negligible vs validation |

---

## References

1. Thuijsman, S.B.A. & Reniers, M.A. (2022). Transformational supervisor synthesis
   for evolving systems. *Discrete Event Dynamic Systems*, 32, 317-358.
   DOI: 10.1007/s10626-021-00354-0

2. Thuijsman, S.B.A. & Reniers, M.A. (2023). Correction to: Transformational
   supervisor synthesis for evolving systems. *Discrete Event Dynamic Systems*.
   DOI: 10.1007/s10626-023-00384-w

3. MATLAB implementation: https://github.com/sbthuijsman/JDEDS_TSS
