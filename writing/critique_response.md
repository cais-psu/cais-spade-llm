# Strategy: Addressing Architectural Critiques

This document provides a structured "Response to Reviewers" and a corresponding addition to the "Discussion" or "Limitations" section of your research paper.

---

## 1. Point-by-Point Response to Reviewers

| Critique Point | Proposed Response |
| :--- | :--- |
| **Stochastic vs. Formal Synthesis** | We acknowledge the stochastic nature of the LLM. However, our architecture follows a **Shielded LLM** paradigm: safety is not a "learned" property of the LLM, but a **formal guarantee provided by the Symbolic Validator (CCA)**. The LLM serves as a heuristic generator, while the CCA serves as a formal supervisor. |
| **Witness Trace as Context** | Treating the witness trace as context (rather than a hard constraint) is a deliberate choice. It allows the LLM to perform **semantic repairs**—such as inserting missing prerequisite tasks or relocating parts—that exceed the scope of simple edge-swapping. The formal correctness of these repairs is then verified by the CCA before execution. |
| **Risk of Oscillation** | To prevent infinite loops or oscillation between unsafe states, the system implements an **identity-hash check** (`_task_nodes_hash`). If the repaired plan is logically identical to the current failing plan, the system identifies a search failure and halts. |
| **No Convergence Guarantee** | Theoretical convergence is an open challenge for stochastic synthesizers. Our system addresses this through **Empirical Bounding**: we enforce a maximum of 3 repair attempts (for both startup and runtime) before pausing for human intervention, which proved sufficient for the assembly scenarios tested. |

---

## 2. Manuscript Addition: Architectural Rationale & Guarantees

*The following text can be added to your **Discussion**, **Architecture**, or **Limitations** section.*

### 2.X Architectural Rationale: The Shielded Stochastic Search

A central challenge in using Large Language Models (LLMs) for assembly plan repair is the lack of formal guarantees regarding convergence and correctness. Unlike traditional Counterexample-Guided Inductive Synthesis (CEGIS) which uses a formal synthesizer (e.g., an SMT solver), our framework uses an LLM as a "Stochastic Heuristic Synthesizer."

To bridge the gap between stochastic generation and formal safety requirements, we employ a **Shielded LLM architecture**. In this paradigm, the LLM is responsible for the *efficiency* of the search (proposing repairs that align with the assembly's semantic requirements), while the Central Controller Agent (CCA) is responsible for the *formal safety* of the result.

Any plan modification proposed by the LLM is subjected to a post-generation verification step where the plan's Finite State Automaton (FSA) is checked against the Deterministic Finite Automata (DFA) of the safety rules. If the repair fails this check, the feedback is fed back into the LLM context for a maximum of $N=3$ attempts.

#### 2.X.1 Mitigation of Oscillation and Search Failures
To prevent the LLM from oscillating between two unsafe plans or returning a non-functional modification, the `ProductAgent` implements an **Identity-Hash Guard**. By hashing the topological structure and parameters of the task nodes, the system detects when a repair attempt has failed to produce a distinct plan. In such cases, the system aborts the automated repair and triggers a "paused_after_failure" state, requesting human intervention. This ensures that the system fails safely rather than looping infinitely in a stochastic search space.

---

## 3. Reference to Code Implementation

When writing your paper, you can cite the following practical mechanisms (verified in the codebase):
- **Oscillation Detection**: Hashing of plan state in `ProductAgent._task_nodes_hash()` and the comparison logic in `run_kickoff_validation`.
- **Empirical Bounding**: `_runtime_repair_max_attempts = 3` and startup `max_retries = 3` constants in `ProductAgent.py`.
- **Trace Formatting**: Selection of `REPLAN_OFFLINE_INSTRUCTIONS` vs `REPLAN_ONLINE_INSTRUCTIONS` to specialize the LLM's repair persona.
