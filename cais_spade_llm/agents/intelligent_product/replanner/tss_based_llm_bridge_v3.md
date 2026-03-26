# TSS-Based LLM Bridge v3: Architecture & Implementation

> **Design and implementation document for the recovery bridge integrating TSS-inspired
> supervisory control with advanced LLM techniques.**
>
> Author: Auto-generated from TSS paper analysis + codebase exploration
> Date: 2026-03-23
> Status: Implemented — v3 is the primary bridge; v2 code removed; quality fixes applied

---

## Context

**Problem:** The v2 Universal Repair Session (`universal_repair_session.py`) uses a
ReAct-style multi-turn loop where the LLM proposes repair programs and a
deterministic two-layer validator checks them. The main bottleneck is now the
**bridge layer** rather than the safety core. Four weaknesses:

1. **Parse fragility** — LLM returns raw JSON text; malformed output wastes turns
2. **Thin feedback surface** — the validator produces structured rejection
   reasons, but the bridge mostly feeds back flat text, so the LLM cannot easily
   distinguish "unsafe", "goal still unmet", "bad primitive sequence", or
   "observation needed"
3. **Prompt churn** — every turn rebuilds large static sections (catalogs,
   schema, repeated instructions) even when only one small constraint changed
4. **Duplicate work** — repeated or near-identical proposals can trigger the same
   validation/debug loop again with little new signal

**Scope boundary:** this v3 design is intentionally **LLM-bridge first**.

- The bridge owns prompting, structured decoding, tool use, turn memory, delta
  prompting, and feedback shaping.
- The validator remains the authority on safety, continuation viability, and
  obligation discharge.
- Major redesign of `compute_winning_set()` or supervisor synthesis is **out of
  scope** for the first v3 iteration.

**Solution:** A TSS-inspired v3 bridge integrating three LLM techniques:

| Technique | What It Does |
|---|---|
| **(A) Constrained decoding** | OpenAI structured outputs enforce `RepairProgram` schema at generation time — eliminates parse errors |
| **(B) Structured planning analysis** | A compact `reasoning` object forces the model to spell out state gaps, intended transitions, and safety checks before proposing actions |
| **(C) Projection tool-use** | LLM calls `project_primitive_sequence()` to simulate primitives and see projected snapshots before committing |

Plus three TSS-inspired bridge improvements:

| Improvement | TSS Origin |
|---|---|
| **Bridge feedback summary** | Partition feedback into "blocked", "still missing", and "admissible" slices instead of one flat rejection blob |
| **Model-delta prompts** | Model delta Δ (Def 4.1) — carry only the changed bridge state between turns |
| **Turn cache + duplicate suppression** | Reuse prior bridge artifacts across small deltas without making the validator secondary |

**TSS paper:** Thuijsman & Reniers (2022), "Transformational Supervisor Synthesis
for Evolving Systems", *Discrete Event Dynamic Systems* 32:317-358.

---

## 1. Why This Is TSS-Inspired

### 1.1 Bridge-Level Mapping

| TSS Concept | Paper Reference | v3 Bridge Mapping |
|---|---|---|
| Model delta Δ | Def 4.1 | Diff between Turn N and Turn N+1 bridge state: context fingerprint, last rejected proposal, and normalized feedback |
| Controllable events Σ_c | Def 2.3 | Primitive actions and top-level observe requests chosen by the LLM |
| Uncontrollable events Σ_u | Def 2.3 | Failure events, fresh observations, runtime snapshot changes |
| Nonblocking | Def 2.5 | Existing `continuation_viable` signal returned by Layer B |
| Supervisor memory | Sec. 4 | `TurnCache` storing prior prompt artifacts, feedback summary, and proposal fingerprints |
| Good/bad partitioning | Thm 3.1 | Bridge-facing buckets such as blocked steps, unmet obligations, violated rules, and still-viable continuations |

### 1.2 Key TSS Insight Applied

TSS's most useful idea for this design is not "rewrite the safety engine"; it is
"do not restart from scratch after every small delta." The v3 bridge applies that
idea at the session layer:

- **Turn N:** build a full prompt, get a structured proposal, validate it, then
  cache the rendered static sections, normalized validator feedback, and
  proposal/context fingerprints
- **Turn N+1:** rebuild fresh runtime context, but reuse compact bridge memory so
  the next prompt focuses on what changed rather than re-explaining the entire
  world
- **If the proposal and context are identical:** reuse the previous validated
  result instead of re-running the same bridge path again

This keeps the bridge incremental while leaving the validator authoritative.

### 1.3 Explicit Non-Goals

- v3 is **not** a major redesign of `compute_winning_set()`
- v3 is **not** a new supervisor synthesis algorithm
- v3 does **not** move validator-owned safety/risk decisions into the LLM

---

## 2. Architecture: Target Session Flow

```
Turn 1 (full prompt):
  build_recovery_context()
  → build_full_bridge_prompt(context, primitives, schema, obligations)
  → ask_llm_structured(response_format=SCHEMA, tools=[project_primitive_sequence])
  → [tool loop: LLM calls project_primitive_sequence → gets snapshot → continues]
  → parse structured response (reasoning + repair_program)
  → validate_repair_program()          [existing validator remains authority]
  → summarize_validation_feedback()    → BridgeFeedbackSummary
  → cache_turn_artifacts()             → TurnCache
  → if rejected: extract constraints, loop

Turn 2+ (bridge delta prompt):
  build_recovery_context()
  → build_delta_prompt(context, feedback, constraints, last_rejected, turn_cache)
    [includes compact schema reminder + relevant catalog excerpts]
    [reattaches full static sections if the context changed materially]
  → ask_llm_structured(response_format=SCHEMA, tools=[project_primitive_sequence])
  → [tool loop]
  → parse structured response
  → diff_programs(new_program, cached) → ProgramDelta
  → if identical proposal and identical context:
      reuse cached validated result
    else:
      full validation
  → update feedback + cache
  → if rejected: loop
```

### v2 → v3 Comparison

| Aspect | v2 (Removed) | v3 (Current) |
|---|---|---|
| LLM output format | Raw JSON text, `json.loads()` | Structured output (`response_format` with `strict: false`) |
| Reasoning | Implicit in `rationale` field | Explicit `reasoning` object with 4-field structured analysis |
| State simulation | None — LLM guesses | `project_primitive_sequence` tool (optional mid-turn) |
| Rejection feedback | Flat text constraints | Bridge feedback summary: 6 buckets + suggested adaptations |
| Turn 2+ prompt | Full rebuild (~17k chars) | Delta prompt from cached static sections + fresh runtime context |
| Validation | Full Layer A+B every turn | Full validation authoritative; identical-turn results memoized |
| Observation guidance | None | `observation_status` flag + `reachability_analysis` |
| Decomposition guidance | None | System instruction rules for multi-function plans |
| Workspace safety net | None | `_validate_workspace_feasibility()` check (A2.5) |

---

## 3. LLM Technique A: Constrained Decoding

### 3.1 Current State

`universal_repair_session.py:520` calls `ask_llm(prompt, with_functions=False)` which
returns raw text. The response is parsed via `json.loads()`. If the JSON is malformed,
the turn is wasted.

### 3.2 Implementation

OpenAI's `response_format` parameter with `type: "json_schema"` guides the output
toward the schema shape. **`strict` is set to `false`** because `function_defs` and
`steps` contain free-form LLM-synthesized objects whose schemas cannot be fully
enumerated at definition time. The `parse_structured_response()` function in
`tss_schemas.py` performs runtime validation instead.

### 3.3 Schema (Discriminated Union)

Defined in `tss_schemas.py` as `REPAIR_TURN_RESPONSE_SCHEMA`:

```json
{
  "name": "repair_turn_response",
  "strict": false,
  "schema": {
    "type": "object",
    "properties": {
      "type": {
        "type": "string",
        "enum": ["observe", "repair_program"]
      },
      "reasoning": {
        "type": "object",
        "description": "Mandatory structured planning analysis before proposing actions",
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
              "required": ["step", "resource", "primitive"]
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
                      "transition_plan", "safety_check"]
      },
      "function_defs": {
        "anyOf": [{"type": "array", "items": {}}, {"type": "null"}],
        "description": "SynthesizedTaskFn definitions (null for observe type)"
      },
      "steps": {
        "anyOf": [{"type": "array", "items": {}}, {"type": "null"}],
        "description": "RepairStep sequence (null for observe type)"
      },
      "success_conditions": {
        "anyOf": [{"type": "array", "items": {}}, {"type": "null"}],
        "description": "Reachability conditions (null for observe type)"
      },
      "rationale": {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "One-line summary"
      },
      "observe_request": {
        "anyOf": [
          {
            "type": "object",
            "properties": {
              "resource_jid": {"type": "string"},
              "primitive": {"type": "string"},
              "params": {"type": "object"},
              "store_as": {"type": "string"}
            }
          },
          {"type": "null"}
        ],
        "description": "For observe type: {resource_jid, primitive, params, store_as}"
      }
    },
    "required": ["type", "reasoning"]
  }
}
```

**Key design notes:**
- `strict: false` — free-form `function_defs`/`steps` can't be fully schematized
- `anyOf` instead of `type: ["array", "null"]` — OpenAI API requires `anyOf` for nullable types
- `observe_request` has explicit `properties` — enables structured observation requests
- `additionalProperties` omitted — not needed with `strict: false`

### 3.4 Integration Point

Method in `llm_agent.py` (`ask_llm_structured()`, added after `ask_llm()`):

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

## 4. LLM Technique B: Structured Planning Analysis with State Projection

### 4.1 Prompt Instruction

In the system prompt (`_V3_FULL_SYSTEM_INSTRUCTION` in `universal_repair_session.py`):

```
## Planning Analysis Requirements

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

Keep the entries compact and factual. This field is a bridge-facing analysis
surface, not a place for long free-form narration.
```

### 4.2 Validator Use of Analysis

The `reasoning.transition_plan` can be cross-checked against `function_defs[].primitive_program`:
- If the transition plan mentions a primitive not in the program → warning
- If the transition plan's projected states differ from validation's projection → diagnostic signal
- This is informational, not blocking — the deterministic validator remains the authority

---

## 4b. Observation & Reachability Signals

When GPT-5 first ran on Case 3, it skipped observation and assigned an unreachable
robot. Root cause: the prompt didn't make information gaps or workspace violations
visible enough. These signals fix that — staying within the ReAct pattern (no hard
gates).

### 4b.1 Observation Status Flag

**File:** `recovery_context_builder.py` — `recovery_context_to_prompt_dict()`

For each part with state `misplaced`, `displaced`, or `unknown`, the prompt builder
checks whether `observed_pose` is present:

```python
part_entry["observation_status"] = (
    "observed" if has_pose
    else "UNOBSERVED — live detect_parts required before planning"
)
```

This makes the information gap visible in the LLM's input state, so its `reasoning`
naturally identifies the need to observe before planning.

### 4b.2 Reachability Analysis

**File:** `recovery_context_builder.py` — `recovery_context_to_prompt_dict()`

A `reachability_analysis` block is computed per resource-part pair by checking whether
the part's pose (observed or last-known) falls within each resource's
`workspace_bounds`:

```json
"reachability_analysis": {
  "LG": {
    "ur5e@localhost": {"reachable": true, "reason": "y=0.198 within ur5e y_max=1.1"},
    "xarm6@localhost": {"reachable": false, "reason": "y=0.198 > xarm6 y_max=0.1"}
  }
}
```

This pre-computes what would otherwise require the LLM to cross-reference
`workspace_bounds` tables manually — reducing reasoning errors.

### 4b.3 Observation Policy in System Instruction

**File:** `universal_repair_session.py` — `_V3_FULL_SYSTEM_INSTRUCTION`

```
## Observation Policy
Before proposing a repair_program, check if any displaced/misplaced part has
observation_status "UNOBSERVED" in the system state. If so, your reasoning
should identify this gap and emit type="observe" to get the live pose first.
Planning with unknown coordinates leads to workspace violations.

Check the reachability_analysis to see which resources can reach each part.
Never assign a pick/place to a resource marked "reachable": false.
```

### 4b.4 Soft Feedback Gate

**File:** `universal_repair_session.py` — session loop

If the LLM emits `repair_program` referencing parts that haven't been observed, the
session does **not** hard-block. Instead, it adds a `discovered_constraint`:

```python
if _unobserved:
    discovered_constraints.append({
        "source": "bridge_observation_policy",
        "constraint": f"Parts {', '.join(_unobserved)} have not been observed — "
                      "coordinates may be inaccurate. Consider observing first.",
    })
```

This feeds back naturally via the ReAct loop — the LLM sees the constraint on the
next turn and can choose to observe or proceed with caution.

---

## 4c. Function Decomposition Rules

**File:** `universal_repair_session.py` — `_V3_FULL_SYSTEM_INSTRUCTION`

Added to the system instruction to address the problem of monolithic single-function
recovery plans:

```
## Function Decomposition Rules

1. Each function_def must perform ONE logical operation (clear, pick, place, stage).
   Do NOT combine multiple operations into one function.

2. Assign each function to the robot whose workspace_bounds CONTAIN the target
   coordinates. Check reachability_analysis — a robot marked "reachable": false
   for a part CANNOT pick/place that part.

3. When multiple robots are involved, sequence steps so:
   - Clear/stow operations first (free the workspace)
   - Part recovery next (pick displaced part, place at goal)
   - Resume operations last (re-pick parts for nominal suffix)

4. A typical multi-robot recovery involves 3–5 separate functions, not 1.
```

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

Method in `universal_repair_session.py` (`_execute_projection_tool`):

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

## 6. TSS Improvement 1: Bridge-Facing Feedback Summary

### 6.1 Dataclass

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class BridgeFeedbackSummary:
    """Compact bridge-owned summary of validator outcome.

    TSS-inspired at the *feedback* layer: separate what is blocked,
    what is still missing, and what remains viable.
    """
    blocked_steps: list[str] = field(default_factory=list)
    """Local issues: bad primitive usage, bad step sequencing, invalid mutation."""

    unmet_obligations: list[str] = field(default_factory=list)
    """Goal/reentry obligations still not discharged by the proposal."""

    violated_rules: list[str] = field(default_factory=list)
    """Safety violations or rule-specific rejection messages."""

    continuation_notes: list[str] = field(default_factory=list)
    """Continuation/dead-end notes from Layer B when present."""

    validator_notes: list[str] = field(default_factory=list)
    """Miscellaneous validator diagnostics that do not fit another bucket."""

    suggested_adaptations: list[str] = field(default_factory=list)
    """Human-readable repair suggestions derived from the buckets above."""
```

### 6.2 How to Compute

No `compute_winning_set()` changes are required for the initial bridge-focused v3.
The summary is derived primarily from `ValidatedRepairProgram.rejection_reasons`
plus current obligations.

```python
def summarize_validation_feedback(
    validated: ValidatedRepairProgram,
) -> BridgeFeedbackSummary:
    blocked_steps: list[str] = []
    unmet_obligations: list[str] = []
    violated_rules: list[str] = []
    continuation_notes: list[str] = []
    validator_notes: list[str] = []

    for reason in validated.rejection_reasons:
        check = str(reason.get("check", "")).strip()
        msg = str(reason.get("message", "")).strip()
        if not msg:
            continue

        if check in {"schema", "function_synthesis", "step_sequencing", "mutation"}:
            blocked_steps.append(msg)
        elif check == "obligation_discharge":
            unmet_obligations.append(msg)
        elif check == "ltlf_safety":
            violated_rules.append(msg)
        elif check == "continuation_viability":
            continuation_notes.append(msg)
        else:
            validator_notes.append(msg)

    return BridgeFeedbackSummary(
        blocked_steps=blocked_steps,
        unmet_obligations=unmet_obligations,
        violated_rules=violated_rules,
        continuation_notes=continuation_notes,
        validator_notes=validator_notes,
        suggested_adaptations=derive_suggested_adaptations(
            blocked_steps=blocked_steps,
            unmet_obligations=unmet_obligations,
            violated_rules=violated_rules,
        ),
    )
```

Optional future hook: if the validator later exposes lightweight bridge-friendly
artifacts such as "blocked transitions" or "reachable goal hints", the bridge can
append them here. That is an optional enhancement, not a prerequisite.

### 6.3 Prompt Rendering

```python
def feedback_to_prompt_section(feedback: BridgeFeedbackSummary) -> str:
    lines = ["## Bridge Feedback Summary"]
    if feedback.blocked_steps:
        lines.append(f"- Blocked steps: {len(feedback.blocked_steps)}")
        for msg in feedback.blocked_steps[:3]:
            lines.append(f"  - {msg}")
    if feedback.unmet_obligations:
        lines.append(f"- Unmet obligations: {len(feedback.unmet_obligations)}")
        for msg in feedback.unmet_obligations[:3]:
            lines.append(f"  - {msg}")
    if feedback.violated_rules:
        lines.append(f"- Violated rules: {len(feedback.violated_rules)}")
        for msg in feedback.violated_rules[:3]:
            lines.append(f"  - {msg}")
    if feedback.continuation_notes:
        lines.append(f"- Continuation notes: {len(feedback.continuation_notes)}")
        for msg in feedback.continuation_notes[:2]:
            lines.append(f"  - {msg}")
    if feedback.validator_notes:
        lines.append(f"- Other validator notes: {len(feedback.validator_notes)}")
    if feedback.suggested_adaptations:
        lines.append("\n### Suggested Revisions")
        for msg in feedback.suggested_adaptations:
            lines.append(f"- {msg}")
    return "\n".join(lines)
```

---

## 6b. Workspace Bounds Feasibility Check

**File:** `repair_program_validator.py`

A new validation sub-check (`_validate_workspace_feasibility()`) catches literal
coordinate violations before they reach the safety DFA layer.

### Position in Validation Pipeline

Inserted as step **A2.5** between function synthesis validation (A2) and step
sequencing (A3):

```
A1: Schema validation
A2: Function synthesis validation (preconditions, effects, primitives)
A2.5: Workspace bounds feasibility ← NEW
A3: Step sequencing validation
B1: LTL/DFA safety check
B2: Obligation discharge
B3: Continuation viability
```

### Implementation

```python
def _validate_workspace_feasibility(
    program: RepairProgram,
    fn_execution_order: list[tuple[str, str]],
    workspace_bounds: dict[str, dict[str, float]],
) -> list[str]:
```

For each function in the execution order:
1. Look up the assigned resource's workspace bounds
2. Scan literal `x`, `y`, `z` coordinates in primitive params
3. Check each coordinate against `{axis}_min_m` / `{axis}_max_m` bounds
4. Return error strings for violations

**Note:** Primitives using `context_ref` (runtime-resolved) are skipped — this check
catches hardcoded coordinates only. Rejection feedback flows back via the ReAct loop.

### Wiring

In `universal_repair_session.py`, workspace bounds are extracted from
`recovery_context.resource_snapshots` and passed to `validate_repair_program()`:

```python
ws_bounds: dict[str, dict[str, float]] = {}
for jid, snap in resource_snapshots.items():
    wb = snap.get("workspace_bounds")
    if isinstance(wb, dict):
        ws_bounds[jid] = wb
validate_repair_program(program, ..., workspace_bounds=ws_bounds)
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

### 7.2 Turn 2+ Delta Prompt (~8-10k chars, bridge-managed)

```
[System instruction: "Continuing repair session. Use the fresh state below and
 revise only the failing parts of the previous proposal."]
[Turn budget: "Turn 2/8"]
[Recovery context: resources, parts, obligations]     ← always fresh
[Relevant primitive excerpts for touched resources]    ← from bridge cache
[Compact schema reminder]                              ← from bridge cache
[Bridge feedback summary + suggestions]                ← NEW
[Discovered constraints from rejections]
[Last rejected proposal (abbreviated)]
[Instruction: fix only the issues identified in feedback]
```

### 7.3 Implementation

`_build_v3_prompt()` in `universal_repair_session.py`:

```python
def _build_repair_program_prompt(
    self,
    recovery_context: RecoveryContext,
    session_state: dict,
    *,
    turn_cache: TurnCache | None = None,
    feedback: BridgeFeedbackSummary | None = None,
) -> str:
    turn_idx = session_state["turn_index"]
    is_delta = turn_idx > 1 and turn_cache is not None

    parts = []
    if is_delta:
        parts.append(
            "You are continuing a repair session. Use the fresh system state "
            "below and revise only the failing parts of the previous proposal."
        )
    else:
        parts.append(FULL_SYSTEM_INSTRUCTION)

    parts.append(f"Turn {turn_idx}/{session_state['max_turns']}.")
    parts.append(recovery_context_to_prompt_dict(recovery_context))

    if is_delta:
        parts.append(turn_cache.render_relevant_catalog_excerpt(recovery_context))
        parts.append(turn_cache.render_schema_reminder())
        if feedback is not None:
            parts.append(feedback_to_prompt_section(feedback))
        if turn_cache.requires_full_regrounding(recovery_context):
            parts.append(self._render_primitive_catalogs(recovery_context))
            parts.append(REPAIR_PROGRAM_SCHEMA_DESCRIPTION)
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

**Token savings:** typically 30-50% on Turn 2+ while preserving enough grounding
to avoid relying on hidden model memory. If the context changes materially, the
bridge can fall back to the full prompt.

---

## 8. TSS Improvement 3: Turn Cache and Duplicate Suppression

### 8.1 Cache Structure

```python
@dataclass
class TurnCache:
    """Bridge-owned cache for prompt deltas and duplicate-turn reuse."""
    turn_index: int
    context_fingerprint: str
    program_fingerprint: str | None
    rendered_catalog_by_resource: dict[str, str]
    rendered_schema_section: str
    last_feedback: BridgeFeedbackSummary | None
    last_rejected_proposal: dict[str, Any] | None
    last_validated_result: dict[str, Any] | None

@dataclass
class ProgramDelta:
    """Diff between current and cached repair programs."""
    is_identical: bool = False
    changed_functions: list[str] = field(default_factory=list)
    changed_success_conditions: bool = False
    touched_resources: list[str] = field(default_factory=list)
```

### 8.2 Delta Logic

```python
def compute_context_fingerprint(recovery_context: RecoveryContext) -> str:
    payload = recovery_context_to_prompt_dict(recovery_context)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()

def compute_program_fingerprint(program: RepairProgram) -> str:
    """SHA256 of the serialized program (excluding bridge-only narration)."""
    d = repair_program_to_dict(program)
    d.pop("rationale", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def diff_programs(program: RepairProgram, cache: TurnCache) -> ProgramDelta:
    new_fp = compute_program_fingerprint(program)
    if new_fp == cache.program_fingerprint:
        return ProgramDelta(is_identical=True)

    # Bridge-level delta only: what changed for prompting and duplicate detection.
    changed_functions = [fn.name for fn in program.function_defs]
    changed_success_conditions = True
    touched_resources = sorted({
        str(step.payload.get("resource_jid", "")).strip()
        for step in program.steps
        if step.kind == RepairStepKind.CALL_FUNCTION
    })

    return ProgramDelta(
        changed_functions=changed_functions,
        changed_success_conditions=changed_success_conditions,
        touched_resources=touched_resources,
    )

def can_reuse_validation_result(
    *,
    delta: ProgramDelta,
    context_fingerprint: str,
    cache: TurnCache,
) -> bool:
    """Safe memoization rule for bridge v3.

    Reuse only when the proposal is identical AND the recovery context is
    identical. This suppresses duplicate-turn work without weakening the
    validator's authority.
    """
    return (
        delta.is_identical
        and context_fingerprint == cache.context_fingerprint
        and cache.last_validated_result is not None
    )
```

This is intentionally conservative. Full incremental Layer B reuse is deferred
until the validator exposes a stable internal cache key of its own.

### 8.3 Session Integration

In `run_v3_repair_session()`, around the validation call:

```python
context_fingerprint = compute_context_fingerprint(recovery_context)
program_delta = diff_programs(program, turn_cache) if turn_cache else ProgramDelta()

if turn_cache and can_reuse_validation_result(
    delta=program_delta,
    context_fingerprint=context_fingerprint,
    cache=turn_cache,
):
    validated = validated_program_from_dict(turn_cache.last_validated_result)
else:
    validated = validate_repair_program(program, ...)

# After validation (always):
turn_cache = TurnCache(
    turn_index=turn_idx,
    context_fingerprint=context_fingerprint,
    program_fingerprint=compute_program_fingerprint(program),
    rendered_catalog_by_resource=self._cache_catalog_sections(recovery_context),
    rendered_schema_section=REPAIR_PROGRAM_SCHEMA_DESCRIPTION,
    last_feedback=summarize_validation_feedback(validated),
    last_rejected_proposal=repair_program_to_dict(program),
    last_validated_result=validated_program_to_dict(validated),
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

**Bridge Feedback Summary:**
```
- Blocked steps: none
- Unmet obligations:
  - obligation not discharged: mcp_01.state must reach assembled
- Violated rules: none
- Continuation notes: none
- Suggested revision:
  - Add assemble_part(mcp_01) after place_part so the plan reaches the
    declared goal instead of stopping at "placed"
```

### Turn 2: Delta Prompt + Acceptance

**Delta prompt** (abbreviated bridge delta):
```
You are continuing a repair session. Use the fresh state below and revise only
the failing parts of the previous proposal.
Turn 2/8 (6 turns remaining).

## Current System State
{"resources": {"ur5e@factory": {"current_state": "idle", "held_part": "mcp_01", ...}, ...},
 "obligations": [{"type": "goal", "entity": "mcp_01", "expected": "assembled"}, ...]}

## Relevant Primitive Excerpts
ur5e@factory: move_to, place_part, assemble_part

## Output Schema Reminder
- Respond with one JSON object of type observe or repair_program.
- If repair_program, include function_defs, steps, success_conditions, rationale.

## Bridge Feedback Summary
- Blocked steps: none
- Unmet obligations:
  - obligation not discharged: mcp_01.state must reach assembled

### Suggested Revisions
- Add assemble_part(mcp_01) after place_part so the plan reaches assembled.

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

**Turn cache note:** Turn 2 changes the proposal, so the bridge runs full
validation again. If the LLM had repeated the exact same rejected program against
the exact same context, v3 would reuse the cached validated result and skip the
duplicate turn work.

---

## 9b. Two-Level Architecture — Nominal Catalog vs LLM Synthesis

The system operates at two distinct levels:

### Level 1: Nominal Task Catalog (`tools.json`)

Five pre-defined functions per robot define the **happy-path FSM**:

| Function | `in_state` | `out_state` | Part Transition |
|---|---|---|---|
| `move_home` | any | idle | — |
| `pick_approach` | idle | at_pick | — |
| `pick_grasp` | at_pick | picked | ready → in_gripper |
| `place_approach` | picked | positioned | in_gripper → in_transit |
| `place_insert` | positioned | placed | in_transit → assembled |

These define a linear FSM: `idle → at_pick → picked → positioned → placed → idle`.
A classical planner (PDDL, DES forward search) can sequence these trivially for
normal operation.

### Level 2: LLM-Synthesized Recovery Functions

Recovery requires operations **outside the nominal catalog**:

- **Reverse transitions** — releasing a held part back to its origin (no `stow` or
  `return_part` exists in `tools.json`)
- **Cross-state jumps** — `recovery_required → idle` is not an `in_state` for any
  catalog function except `move_home`
- **Novel compositions** — recovery pick from a non-standard drop location with
  runtime-observed coordinates
- **Dataflow binding** — `context_ref` wiring observation outputs into primitive
  parameters (e.g., detected pose → `move_cartesian` target)

The LLM synthesizes these missing `function_defs` from raw primitives (Level 2),
while the deterministic validator ensures they satisfy the same
precondition/effect/safety constraints as catalog functions.

### The Prompt Shows Both Levels

The LLM prompt includes:
- `available_task_actions` — the 10 catalog functions from `tools.json` (Level 1)
- `available_primitives` — per-resource primitive catalogs filtered through
  `filter_synthesis_primitive_catalog()` (Level 2)

This allows the LLM to reuse catalog functions when they fit and synthesize new
ones when recovery demands operations the catalog doesn't cover.

---

## 9c. Why Not Hybrid PDDL + LLM?

The system is **already a hybrid** in practice:

| Layer | Approach | What It Does |
|---|---|---|
| **Classical DES (deterministic)** | Layer A + B validators | Safety validation (LTL→DFA), precondition/effect checking, state projection, obligation discharge, continuation viability |
| **LLM (heuristic)** | Bridge synthesis | Proposes candidate programs — never trusted, always verified |

**Why a PDDL front-end doesn't help:**

1. The nominal catalog defines a trivial FSM that a classical planner can sequence.
   But recovery requires **operations not in the catalog** — reverse transitions,
   cross-state jumps, and novel compositions with runtime dataflow. A PDDL planner
   can only sequence existing action schemas; it cannot synthesize new ones.

2. The LLM synthesizes **action schemas** (complete `function_defs` with primitive
   sequences and `context_ref` dataflow bindings), not just action sequences. This
   is fundamentally different from classical planning.

3. The validator already provides the formal correctness guarantees that a classical
   planner would add: precondition checking, effect projection, safety DFA
   satisfaction, obligation discharge, and continuation viability.

4. Adding a PDDL front-end for the abstract layer (which the LLM already solves in
   1–2 turns) doesn't reduce failure modes — it adds integration complexity while
   the hard part (primitive program synthesis with dataflow) still requires an LLM.

---

## 9d. Corrected Case 3 Workflow — Turn-by-Turn Trace

This section traces the full recovery for the Case 3 scenario (xArm6 stuck at
board, UR5e holding MCP, LG unassembled) using the v3 bridge improvements.

**Key corrections from earlier drafts:**
1. No library candidates in Turn 1 (Case 3 has no prior recovery history)
2. Composite primitives (`grasp_part`, `release_part`) — not raw hidden ones
3. Five separate `function_defs` (matching preprogrammed bridge structure)
4. `compute_pick_targets` / `compute_place_targets` in primitive sequences
5. `workspace_bounds` in recovery context for reachability reasoning
6. `available_task_actions` (tools.json) shown in prompt alongside primitives

### Turn 1: Observe

**INPUT:**
- System state: xarm6 `recovery_required` at board, ur5e `picked` holding MCP
- `workspace_bounds`: ur5e y∈[-0.15, 1.10], xarm6 y∈[-1.00, 0.10]
- Available primitives (per resource): `move_cartesian`, `move_pose`,
  `move_to_named_pose`, `grasp_part`, `release_part`, `detect_parts`,
  `compute_pick_targets`, `compute_place_targets`, `get_current_pose`
  (hidden: `open_gripper`, `close_gripper`, `attach_part`, `detach_part`)
- Available task actions (tools.json): `move_home`, `pick_approach`, `pick_grasp`,
  `place_approach`, `place_insert` — per robot (10 total)
- Obligations: board mutex (no simultaneous occupancy) + LG must reach `assembled`
- No library candidates (fresh system)

**OUTPUT → `observe`:**
```json
{
  "action": "observe",
  "observe": {
    "function_name": "detect_parts",
    "resource_jid": "ur5e@localhost",
    "parameters": {"part_name": "lg_01"}
  },
  "reasoning": "LG location unknown. From workspace_bounds: LG last at y≈0.198 on board — xarm6 y_max=0.10 cannot reach, ur5e y_max=1.10 can. Need exact pose before planning pick."
}
```

**LLM reasoning:** The LLM sees `workspace_bounds` in the recovery context and
reasons that LG (at the board, y≈0.198) is outside xarm6's reachable envelope
(y_max=0.10) but within ur5e's (y_max=1.10). Therefore ur5e must handle LG recovery.

### Turn 2: First Repair Proposal (REJECTED)

**INPUT (delta prompt):**
- Fresh recovery context + observation results (LG exact pose from `detect_parts`)
- Cached primitive excerpts + compact schema reminder
- No prior rejections yet

**OUTPUT → `repair_program` with 4 `function_defs`:**
```yaml
function_defs:
  1. clear_xarm6:  [xarm6] move_to_named_pose(home)
  2. return_mcp:   [ur5e]  release_part() at prusa-mk4-2 origin
  3. pick_lg:      [ur5e]  detect_parts → compute_pick_targets → move_cartesian(context_ref) → grasp_part
  4. insert_lg:    [ur5e]  compute_place_targets → move_cartesian(context_ref) → release_part

steps: [call clear_xarm6, call return_mcp, call pick_lg, call insert_lg]
success_conditions: [LG.state = assembled]
```

**REJECTED — Layer B `obligation_discharge`:** MCP was in ur5e's gripper before
recovery. After `return_mcp` releases it and `insert_lg` completes, MCP remains at
prusa-mk4-2 origin — not restored to ur5e's gripper. The continuation plan expects
ur5e to still hold MCP for subsequent `place_approach` + `place_insert`.

**Bridge feedback summary:**
```
unmet_obligations: ["MCP must be restored to ur5e gripper for continuation"]
suggested_adaptations: ["Add a repick_mcp step after insert_lg"]
```

### Turn 3: Corrected Repair Proposal (ACCEPTED)

**INPUT (delta prompt):**
- Fresh recovery context + observation store (LG pose cached)
- Bridge feedback summary: unmet obligation "MCP must be restored"
- Discovered constraint: `{"constraint": "MCP must be in ur5e gripper at program end"}`
- Last rejected proposal (for reference)

**OUTPUT → `repair_program` with 5 `function_defs`:**
```yaml
function_defs:
  1. clear_xarm6:      [xarm6] move_to_named_pose(home)
  2. return_mcp:       [ur5e]  move_cartesian(prusa-mk4-2 origin) → release_part(mcp_01)
  3. pick_lg:          [ur5e]  detect_parts(lg_01) → compute_pick_targets(lg_01) → move_cartesian(context_ref: pick_targets) → grasp_part(lg_01)
  4. insert_lg:        [ur5e]  compute_place_targets(lg_01, assembly_board-v1) → move_cartesian(context_ref: place_targets) → release_part(lg_01)
  5. repick_mcp:       [ur5e]  move_cartesian(prusa-mk4-2) → grasp_part(mcp_01)

steps: [call 1, call 2, call 3, call 4, call 5]
success_conditions: [LG.state = assembled, MCP.state = in_gripper]
```

**ACCEPTED — Validation detail:**

| Check | Result |
|---|---|
| **Layer A: Primitive existence** | All primitives in catalog ✓ |
| **Layer A: Precondition chain** | `clear_xarm6` projects xarm6→idle; `return_mcp` projects ur5e→idle+empty; `pick_lg` requires ur5e idle+empty ✓; `insert_lg` requires ur5e holding LG ✓; `repick_mcp` requires ur5e idle+empty ✓ |
| **Layer B: FSA safety** | No simultaneous board occupancy (xarm6 cleared before ur5e approaches) ✓ |
| **Layer B: Obligation discharge** | LG.state=assembled ✓, MCP.state=in_gripper ✓ |
| **Layer B: Continuation viable** | Plan suffix resumable from ur5e holding MCP ✓ |

---

## 9e. Formal DES Framework

The v3 bridge can be understood through the lens of **Discrete Event Systems (DES)**
and supervisory control theory.

### State Space

The global system state is the product automaton:

```
S = S_r1 × S_r2 × ... × S_rn × S_p1 × S_p2 × ... × S_pm
```

where:
- `S_ri` = state of resource *i* (e.g., `{idle, at_pick, picked, positioned, placed, recovery_required, ...}`)
- `S_pj` = state of part *j* (e.g., `{ready, in_gripper, in_transit, assembled}`)

Each state `s ∈ S` is a snapshot of all resource states and part states.

### Events (Controllable)

Each primitive `σ` is a **controllable event** with:
- **Guard** `g(σ, s)`: precondition predicate over current state
- **Update** `δ(σ, s) → s'`: deterministic state transition (effect projection)

A `function_def` is a **sequence of events**: `f = σ₁ · σ₂ · ... · σₖ`

A `repair_program` is an **ordered composition** of function_defs:
`P = f₁ · f₂ · ... · fₙ`

### Safety Specification

Safety rules are expressed as **LTL formulas** over atomic propositions (APs) on
the state space. Each LTL formula is compiled to a **DFA (Deterministic Finite
Automaton)** via `ltlf2dfa`.

A program `P` is **safe** iff the projected state trace `s₀, s₁, ..., sₙ` is
accepted by **all** safety DFAs — i.e., the trace never reaches a rejecting state
in any DFA.

### Each Turn as Supervisor Synthesis Attempt

Each bridge turn maps to a **supervisor synthesis attempt**:

1. **LLM proposes** a candidate supervisor (repair program `P`)
2. **Validator checks:**
   - Forward reachability: project `s₀ →^P sₙ` through all events
   - Safety: check projected trace against all DFAs
   - Goal satisfaction: verify `sₙ ⊨ obligations`
   - Continuation: verify plan suffix is executable from `sₙ`
3. **If rejected:** the validator identifies which check failed and produces a
   structured feedback summary — the LLM gets a narrower search space for the
   next attempt

### Incremental Refinement as Constraint Narrowing

Rejection narrows the search space. Formally:

```
L₀ = Σ*                          (all possible programs)
L₁ = L₀ ∩ L_safety               (safety-satisfying programs)
L₂ = L₁ ∩ L_obligations          (obligation-discharging programs)
L₃ = L₂ ∩ L_continuation         (continuation-viable programs)
L₄ = L₃ ∩ L_discovered           (programs satisfying discovered constraints)
```

Each rejected turn adds a **discovered constraint** that further narrows `L`:
- Turn 2 rejection ("MCP not restored") adds `L_mcp_restore` to the intersection
- Turn 3 proposal satisfies the narrowed language `L₄ ∩ L_mcp_restore`

### Projection as Forward Reachability

The projection tool (`validate_and_project_steps()`) computes:

```
project(s, f) = δ(σₖ, δ(σₖ₋₁, ... δ(σ₁, s) ...))
```

This is **forward reachability** over the product automaton — checking that each
guard holds and computing the resulting state after each event.

### Memoization as Language Equivalence

The turn cache checks whether two proposals are **language-equivalent** — if a
new proposal produces the same state trace as a previously rejected one, the
cached rejection is reused without re-running validation. This corresponds to
checking trace equivalence in the DFA product.

### Concept Mapping: DES ↔ Bridge

| DES Concept | Bridge Implementation |
|---|---|
| Product automaton state `s` | `RecoveryContext` (resource snapshots + part states) |
| Controllable event `σ` | Primitive in catalog (with guard = precondition, update = effect) |
| Supervisor candidate | `repair_program` (ordered function_defs) |
| Safety specification DFA | Compiled LTL→DFA automata in Layer B validator |
| Forward reachability | `validate_and_project_steps()` in `primitive_semantics.py` |
| Supervisor synthesis | LLM turn (propose) + validator (verify) |
| Model delta | Delta prompt (cached static + fresh dynamic) |
| Incremental memory | `TurnCache` + `observation_store` + `discovered_constraints` |
| Good/bad partitioning | `BridgeFeedbackSummary` buckets |

---

## 9f. Incremental Knowledge Accumulation

The **bridge is the memory**. The LLM has no cross-call memory. Each turn's delta
prompt carries forward all accumulated knowledge via four mechanisms:

```
Turn 1: LLM receives full world state + workspace_bounds
        → observes (detect_parts LG)
        → accumulates: exact LG pose in observation_store

Turn 2: LLM receives delta prompt + observation results
        → proposes 4-function recovery
        → REJECTED: "MCP not restored"
        → accumulates: discovered_constraint + structured feedback

Turn 3: LLM receives delta prompt + feedback + constraint + last_rejected_proposal
        → adds 5th function (repick_mcp)
        → ACCEPTED: all obligations discharged
```

Each accumulator in the session loop:

| Accumulator | Source | Carried Forward Via |
|---|---|---|
| `observation_store` | `observe` turn results | Delta prompt `observation_results` section |
| `discovered_constraints` | Validator rejections | Delta prompt `discovered_constraints` section |
| `last_rejected_proposal` | Previous turn's rejected program | Delta prompt `last_rejected_proposal` section |
| `BridgeFeedbackSummary` | Validator + bridge analysis | Delta prompt `feedback_summary` section |

This design means the system is **not hardcoded** for any specific failure mode.
The same loop handles breakdowns, product demand shifts, resource unavailability,
or any unexpected failure — the LLM reasons from the current state and accumulated
constraints to synthesize whatever recovery is needed.

---

## 9g. Relationship to ReAct Pattern

The v2/v3 bridge follows the **ReAct pattern** (Reason → Act → Observe):

```
┌─────────────────────────────────────────────────┐
│  Turn 1: Reason (world state) → Act (observe)   │
│  Turn 2: Reason (+ observation) → Act (propose) │
│  Turn 3: Reason (+ feedback) → Act (refine)     │
└─────────────────────────────────────────────────┘
```

**v3 enriches the base ReAct loop with TSS-inspired improvements:**

| ReAct Component | v2 (Base) | v3 (TSS-Enriched) |
|---|---|---|
| **Reasoning** | Unstructured text in JSON | Structured `reasoning` field with explicit state analysis |
| **Acting** | `observe` or `repair_program` | Same + `project_primitive_sequence` tool for mid-turn simulation |
| **Observing** | Observation results as flat text | Observation store with typed results + delta prompts |
| **Feedback** | Raw rejection reasons | `BridgeFeedbackSummary` with partitioned buckets |
| **Memory** | Full prompt rebuild each turn | `TurnCache` + delta prompts (TSS model-delta) |
| **Convergence** | Hope-based (same prompt, hope for different result) | Constraint narrowing (each rejection provably shrinks search space) |

The TSS-inspired additions (delta prompts, turn cache, feedback partitioning) are
not a replacement for ReAct but an **enrichment** that improves convergence speed
and token efficiency while maintaining the same Reason→Act→Observe structure.

**Note on `preprogrammed_bridge_scenarios.py`:** This file contains reference
implementations showing the *ideal* recovery outcome for each test case. It is
**not used in production** — the LLM bridge must discover the same solutions
autonomously. It exists solely for testing and validating that the bridge produces
equivalent results.

---

## 10. V3 File Inventory

All paths relative to `cais_spade_llm/agents/intelligent_product/replanner/`.

### V3-Specific Files (created for v3)

| File | Purpose |
|---|---|
| `llm_bridge/tss_schemas.py` | `REPAIR_TURN_RESPONSE_SCHEMA` (constrained decoding), `PROJECT_PRIMITIVE_SEQUENCE_TOOL` (tool definition), `parse_structured_response()` (runtime parser) |
| `llm_bridge/tss_feedback.py` | `BridgeFeedbackSummary` frozen dataclass, `summarize_validation_feedback()` (partitions rejections), `feedback_to_prompt_section()` (renders for prompt), `_derive_suggested_adaptations()` (rule-based suggestions) |
| `llm_bridge/tss_turn_cache.py` | `TurnCache` dataclass (prompt caching + duplicate suppression), `ProgramDelta` dataclass, `compute_context_fingerprint()` / `compute_program_fingerprint()` (SHA256), `diff_programs()`, `can_reuse_validation_result()` |

### Modified Files

| File | v3 Changes |
|---|---|
| `llm_bridge/universal_repair_session.py` | v3-only session module: `run_v3_repair_session()`, `_build_v3_prompt()` (full/delta), `_execute_projection_tool()`, observation policy, decomposition rules in system instruction, soft feedback gate for unobserved parts |
| `llm_bridge/mutation_types.py` | Added `reasoning: dict[str, Any]` to `RepairProgram`, added `validated_program_from_dict()` for cache deserialization |
| `llm_bridge/recovery_context_builder.py` | Added `observation_status` flag, `reachability_analysis` block, `workspace_bounds` extraction |
| `llm_bridge/repair_program_validator.py` | Added `workspace_bounds` parameter, `_validate_workspace_feasibility()` check (A2.5) |
| `agents/shared_information/llm_agent.py` | Added `ask_llm_structured()` — `response_format` + `tools` + tool-call loop |
| `test/test_case3_recovery_main.py` | Added `_run_live_v3_session()`, default mode `v3-live`, `FakeProductAgent.ask_llm_structured()` |

### Shared Foundation (unchanged by v3)

| File | Role |
|---|---|
| `llm_bridge/primitive_semantics.py` | `validate_and_project_steps()` — used by projection tool |
| `llm_bridge/bridge_safety.py` | `_bridge_critical_parts()` — used by observation policy |
| `llm_bridge/bridge_session.py` | Entry point calling `run_universal_repair_session()` |

---

## 11. Reading Order for Understanding v3

1. `llm_bridge/tss_schemas.py` — schema + tool definition + parser
2. `llm_bridge/tss_feedback.py` — feedback partitioning
3. `llm_bridge/tss_turn_cache.py` — caching + delta logic
4. `llm_bridge/mutation_types.py` — core dataclasses
5. `llm_bridge/recovery_context_builder.py` — context + observation/reachability signals
6. `llm_bridge/repair_program_validator.py` — two-layer validation + workspace check
7. `llm_bridge/universal_repair_session.py` — session loop (ties everything together)
8. `agents/shared_information/llm_agent.py` — `ask_llm_structured()` method

---

## 12. Implementation Status

| Component | Status | Notes |
|---|---|---|
| Constrained decoding (Technique A) | Done | `strict: false`, `anyOf` nullable types |
| Structured planning analysis (Technique B) | Done | 4-field `reasoning` object mandatory |
| Projection tool-use (Technique C) | Done | Available but LLM may not always invoke it |
| Bridge feedback summary (TSS 1) | Done | 6 buckets + suggested adaptations |
| Model-delta prompts (TSS 2) | Done | Turn 2+ uses cached sections + fresh context |
| Turn cache + duplicate suppression (TSS 3) | Done | SHA256 fingerprinting, conservative reuse |
| Observation status flag | Done | `UNOBSERVED` on parts lacking `observed_pose` |
| Reachability analysis | Done | Pre-computed per resource-part pair |
| Decomposition rules | Done | System instruction guidance |
| Workspace bounds validator | Done | A2.5 check on literal coordinates |

### Potential Next Steps

- Test with more scenarios beyond Case 3 to validate generalization
- Add unit tests for v3-specific modules (`tss_schemas`, `tss_feedback`, `tss_turn_cache`)
- Tune decomposition rules based on observed LLM behavior across models
- Consider `strict: true` if OpenAI adds support for partially-strict schemas

---

## 13. Risks & Mitigations

| Risk | Status | Mitigation |
|---|---|---|
| `response_format` + `tools` not supported simultaneously on gpt-5 | **Resolved** — works with `strict: false` | Both `response_format` and `tools` passed in same call |
| Schema too complex for `strict: true` mode | **Resolved** — using `strict: false` | `anyOf` for nullable types; runtime validation via `parse_structured_response()` |
| Delta prompts over-compress context | **Mitigated** | Fresh obligations + state always included; `requires_full_regrounding()` falls back to full prompt |
| Duplicate-turn cache returns stale results | **Mitigated** | Requires both program AND context fingerprints to match |
| Projection tool adds latency | **Mitigated** | Capped at 3 calls; tool is optional (LLM may not invoke) |
| Feedback summary mirrors validator phrasing too literally | **Mitigated** | Normalized into bridge-owned buckets + rule-based suggestions |
| LLM skips observation for unobserved parts | **Mitigated** | `observation_status` flag + `reachability_analysis` + soft feedback gate |
| LLM produces monolithic single-function plans | **Mitigated** | Decomposition rules in system instruction |
| LLM assigns robot outside workspace bounds | **Mitigated** | `_validate_workspace_feasibility()` check + `reachability_analysis` |

---

## References

1. Thuijsman, S.B.A. & Reniers, M.A. (2022). Transformational supervisor synthesis
   for evolving systems. *Discrete Event Dynamic Systems*, 32, 317-358.
   DOI: 10.1007/s10626-021-00354-0

2. Thuijsman, S.B.A. & Reniers, M.A. (2023). Correction to: Transformational
   supervisor synthesis for evolving systems. *Discrete Event Dynamic Systems*.
   DOI: 10.1007/s10626-023-00384-w

3. MATLAB implementation: https://github.com/sbthuijsman/JDEDS_TSS

---

## 14. Implementation Record

> Initial: 2026-03-23 — v3 core (constrained decoding, projection tool, feedback,
> delta prompts, turn cache).
> Updated: 2026-03-23 — quality fixes (observation signals, reachability analysis,
> decomposition rules, workspace bounds validator).

### 14.1 Files Created

See Section 10 (V3 File Inventory) for the complete listing.

### 14.2 Files Modified

See Section 10 (V3 File Inventory) for the complete listing, including quality fix changes.

### 14.3 Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│  run_universal_repair_session()                     │
│  (backward compat entry point in bridge_session.py) │
│       │ delegates to                                │
│       ▼                                             │
│  run_v3_repair_session()                            │
│       │                                             │
│       ├─ _build_v3_prompt()      (full or delta)    │
│       │   ├─ TurnCache           (cached sections)  │
│       │   ├─ BridgeFeedbackSummary (partitioned)    │
│       │   └─ _render_primitive_catalogs_v3()        │
│       │                                             │
│       ├─ ask_llm_structured()    (constrained JSON) │
│       │   ├─ response_format     (REPAIR_TURN_...)  │
│       │   ├─ tools               (PROJECT_PRIM_...) │
│       │   └─ tool_executor       (projection tool)  │
│       │       └─ validate_and_project_steps()       │
│       │                                             │
│       ├─ parse_structured_response()                │
│       │                                             │
│       ├─ _run_repair_validation()                   │
│       │   └─ validate_repair_program() (2-layer)    │
│       │                                             │
│       ├─ summarize_validation_feedback()            │
│       │                                             │
│       ├─ TurnCache update + fingerprinting          │
│       │   ├─ compute_context_fingerprint()          │
│       │   ├─ compute_program_fingerprint()          │
│       │   ├─ diff_programs()                        │
│       │   └─ can_reuse_validation_result()          │
│       │                                             │
│       └─ constraint extraction on rejection         │
└─────────────────────────────────────────────────────┘
```

### 14.4 v3 Session Loop Flow

1. Build `RecoveryContext` from `prepared_bridge_request` (always fresh)
2. Build prompt via `_build_v3_prompt()`:
   - **Turn 1**: full system instruction + reasoning requirements + primitives + schema + library candidates
   - **Turn 2+**: delta instruction + cached catalog excerpt + schema reminder + feedback summary + fresh context + constraints + last rejected proposal
   - Falls back to full prompt if `TurnCache.requires_full_regrounding()` detects material context change
3. Call `ask_llm_structured()` with:
   - `response_format=REPAIR_TURN_RESPONSE_SCHEMA` (constrained decoding)
   - `tools=[PROJECT_PRIMITIVE_SEQUENCE_TOOL]` (mid-turn simulation)
   - `tool_executor` → `_execute_projection_tool()` → `validate_and_project_steps()`
4. Parse response via `parse_structured_response()` (discriminated union: `observe` | `repair_program`)
5. If `observe`: execute observation, store result, continue
6. If `repair_program`:
   - Check duplicate via `diff_programs()` + `can_reuse_validation_result()`
   - If not duplicate: run `_run_repair_validation()` (2-layer validator)
   - Build `BridgeFeedbackSummary` from validation result
   - Update `TurnCache` with rendered sections + fingerprints
   - If valid: accept and return
   - If rejected: extract constraints, store rejection, continue

### 14.5 Key Design Decisions

1. **Fresh LLM calls per turn** — each turn is a fresh `ask_llm_structured()` call; the bridge is the memory (not a stateful conversation). This prevents context poisoning.
2. **v2 backward compatibility** — `run_universal_repair_session()` is kept as a method that delegates to `run_v3_repair_session()`, since `bridge_session.py:1242` calls it.
3. **Monkey-patching pattern** — v3 standalone functions (`run_v3_repair_session`, `_build_v3_prompt`, `_execute_projection_tool`) are defined at module level and attached to `UniversalRepairSessionMixin` via assignment. This keeps the functions testable while maintaining the mixin pattern.
4. **`_REPAIR_PROGRAM_SCHEMA_SECTION`** — the v2 text schema constant is retained and reused by v3 for prompt grounding (it describes the JSON structure the LLM should follow).

### 14.6 Running the v3 Bridge

```bash
# Default: v3 live session on Case 3 (F5 in VSCode)
python3 test/test_case3_recovery_main.py

# Explicit v3 mode with custom model
python3 test/test_case3_recovery_main.py --mode v3-live --model gpt-5

# Run pytest suite
python3 test/test_case3_recovery_main.py --mode test
```
