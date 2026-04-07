# Disclosure: Prompt Engineering Strategies for Offline Replanning

This document details the prompt engineering strategies and violation trace formatting employed in the **caIs-spade-llm** replanning feedback mechanism. These strategies are designed to ensure safety-aligned, graph-consistent plan repairs while minimizing LLM reasoning bias and maximizing reproducibility.

## 1. Core Prompt Engineering Strategies

The replanning agent follows a structured reasoning process guided by four key principles:

1. **Context-Specific Role Assumption**: The LLM is explicitly cast as a "Plan Repair Expert (PRE-EXECUTION MODE)" to distinguish its goal from runtime recovery or initial generation.
2. **Strict Graph-Based Repair Rules**: To maintain plan integrity, a set of "Global Constraints" is enforced, primarily centered on maintaining a Directed Acyclic Graph (DAG) and preventing symmetric dependencies.
3. **Formal Verification Feedback (FSA-DFA Product)**: Instead of vague error messages, the LLM is provided with formal "witness traces" extracted from a product-space search between the plan's Finite State Automaton (FSA) and the safety rules' Deterministic Finite Automata (DFA).
4. **Data Consistency Enforcement ("Saying ≠ Doing")**: A critical instruction forces the LLM to update the actual JSON task fields (e.g., `predecessors`) rather than just describing the fix in a natural language `change_reason` field.

---

## 2. Specific Prompt Templates

### 2.1. System Instructions (Offline Mode)

The following block represents the foundational instructions provided to the agent for every offline repair attempt:

```text
You are a Plan Repair Expert (PRE-EXECUTION MODE).
You are given a PROPOSED execution plan that violates safety rules BEFORE any execution has started.

CONTEXT:
- NO tasks have executed yet - this is pure static analysis.
- Violations come from FSA (Finite State Automaton) verification.
- You will see "witness traces" showing problematic execution paths in the FSA.

YOUR GOAL:
Fix the plan structure to satisfy ALL safety rules before execution begins.

GLOBAL CONSTRAINTS (MUST FOLLOW):
- The repaired plan MUST remain a DAG (no cycles in predecessor relations).
- NEVER add mutual/symmetric dependencies (do NOT add A as predecessor of B AND B as predecessor of A).
- For edge repairs, ONLY edit `predecessors`. Do NOT edit `successors` (the system will rebuild successors automatically).
- Do not remove required work: if a task contributes to satisfying a requirement, prefer re-ordering or coordination instead of deletion.

STRICT RULES FOR MODIFICATION:
1. ANALYZE THE VIOLATION LOGIC:
   - TEMPORAL / ORDERING CONSTRAINT (e.g., "A must happen before B"):
       -> MODIFY EDGES by adding a predecessor relation to enforce ordering.
   - MISSING PREREQUISITE (e.g., "Action A requires Setup B"):
       -> INSERT a new task to satisfy the requirement and connect it appropriately.
   - ATTRIBUTE / PARAMETER CONSTRAINT (e.g., "Invalid resource assignment"):
       -> MODIFY ATTRIBUTES such as `resource_jid` or `params`.

2. CRITICAL - DATA CONSISTENCY:
- **SAYING ≠ DOING**: Do not just describe the fix in `change_reason`; you MUST update the actual JSON fields!
- If your `change_reason` says "removed dependency on X", then X MUST NOT be in `predecessors`.
- If your `change_reason` says "added dependency on Y", then Y MUST be in `predecessors`.
- When modifying an existing task, provide the COMPLETE FINAL `predecessors` list, not a diff.
```

---

## 3. Violation Trace Formatting

To ensure the LLM understands the formal failure, each violation is formatted as follows:

- **Rule ID**: The unique identifier for the violated safety rule.
- **Requirement**: The natural language description of the constraint.
- **Logic**: The corresponding LTLf (Linear Temporal Logic on finite traces) formula.
- **Witness Trace (Task IDs)**: A sequential list of task IDs that, if executed, lead the plan into a violating state in the safety DFA.
- **Relevant Tasks (Projected Subgraph)**: To reduce token noise, only the tasks involved in the witness trace and their immediate neighborhood are serialized in the prompt.
- **Relevant Predecessor Map**: A projected mapping of local dependencies within the violation subgraph.

### Example Formatting:
```text
- Rule ID: rule_mcp_before_sg
  Requirement: The MCP part must be placed before the SG part is acquired.
  Logic: (!acquire_SG U place_MCP)
  Witness trace (task_ids): ["acquire_SG_001", "place_SG_002"]
  Relevant tasks (projected subgraph for this rule):
  [
    {
      "id": "acquire_SG_001",
      "function_name": "acquire",
      "params": {"part": "SG"},
      "_FOCUS_HERE": " <<< THIS TASK IS INVOLVED IN A VIOLATION"
    },
    ...
  ]
  Relevant predecessor map (within this rule): {"place_SG_002": ["acquire_SG_001"]}
```

---

## 4. Input Representation Strategies

### 4.1. Focused Highlighting
In the `plan_nodes` JSON payload, nodes that are directly involved in witness traces are tagged with a `_FOCUS_HERE` metadata field. This encourages the LLM's cross-attention mechanisms to prioritize these nodes when generating repair sequences.

### 4.2. Actionable Scope Filtering
To minimize cognitive load, the prompt primarily includes "actionable" nodes (i.e. nodes with `status != "completed"`). In the offline mode, this typically includes the entire plan, whereas in online mode, this drastically reduces the search space for the LLM.

---

## 5. Structured Output Format

The LLM is required to respond with a JSON object containing a list of `tasks`. Each task must specify its `id` and the fields to be updated, added, or deleted:

```json
{
  "tasks": [
    {
      "id": "TASK_ID",
      "predecessors": ["NEW_PRED_1", "NEW_PRED_2"],
      "change_reason": "Explanation of the repair logic"
    },
    {
      "id": "NEW_TASK_ID",
      "function_name": "...",
      "params": { ... },
      "predecessors": [...],
      "change_reason": "INSERTION: Added missing prerequisite"
    }
  ]
}
```
