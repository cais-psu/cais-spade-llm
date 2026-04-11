# WODES Reviewer Revision Notes

This document stages the text and analysis requested by the reviewers for the current WODES manuscript. The PDF source is not present in this repository, so these revisions are prepared here for transfer into the paper source.

## 1. Reviewer-Facing Summary

The revision should make five points explicit:

1. The current verifier is an implemented explicit-state DFS validator, not a symbolic model checker.
2. The case study is implemented in practice on the dual-robot Gazebo/ROS2/SPADE stack, but it is still a small feasibility study.
3. The paper needs a direct baseline against a pure-LLM planner that receives the same safety text but no formal verifier-driven repair loop.
4. The LLM repair loop has no formal convergence guarantee.
5. The implementation enforces bounded repair attempts and aborts no-change oscillations instead of looping indefinitely.

## 2. Manuscript Insertion: Complexity of the Centralized DFS Validator

Place this subsection after Algorithm 1 or immediately after the figure that introduces DFS-based validation.

### 2.X Complexity of Explicit-State Validation

Let \(I\) be the set of resource agents, with \(n = |I|\). Let \(X_j\) and \(T_j\) denote the local state and transition sets of resource agent \(j\), and let \(Q_r\) denote the state set of the DFA compiled for safety rule \(r\). Define uniform upper bounds \(s = \max_{j \in I} |X_j|\) and \(t = \max_{j \in I} |T_j|\).

The centralized product state space explored by the Central Controller Agent is

\[
|X_{plan}| = \prod_{j \in I} |X_j| = O(s^n),
\]

which is exponential in the number of agents. The joint transition relation induced by asynchronous resource evolution has size

\[
|E_{plan}| = O\left(\sum_{j \in I} |T_j| \prod_{\ell \neq j} |X_\ell|\right),
\]

which simplifies under the uniform bounds above to

\[
|E_{plan}| = O(n t s^{n-1}).
\]

For a single safety rule \(r\), the validator performs DFS over the product of the plan FSA and the rule DFA, giving a worst-case cost of

\[
O(|X_{plan}|\,|Q_r| + |E_{plan}|\,|Q_r|).
\]

Because the implementation validates each safety DFA separately, the total cost across all rules is

\[
O\left((|X_{plan}| + |E_{plan}|)\sum_r |Q_r|\right).
\]

This complexity is consistent with the current implementation in `cais_spade_llm/agents/central_controller/plan_safety_validator.py`, which performs explicit-state DFS and caps the explored product space with `MAX_PRODUCT_STATES = 200000`. Accordingly, the present implementation is suitable for the small offline assembly study reported here, but it is not claimed to be a scalable symbolic verifier for large manufacturing cells.

## 3. Manuscript Insertion: Baseline Comparison

The main experimental table should compare exactly two methods:

- `Pure LLM (safety text only, no formal verifier)`
- `LLM + Formal Verification`

These correspond directly to the current experiment methods:

- `llm_nl_safety`
- `verified`

The recommended main table is:

| Scenario | Robots | Parts | Pure LLM Safety-Valid % | LLM + Formal Verification Safety-Valid % | Delta |
|---|---:|---:|---:|---:|---:|
| S1 | ... | ... | ... | ... | ... |
| S2 | ... | ... | ... | ... | ... |
| S3 | ... | ... | ... | ... | ... |

For the paper, `Safety-Valid %` should be presented as the primary accuracy-style metric. The complementary invalid rate is redundant and should not be shown in the main paper table. Likewise, `verification time`, `product states`, and `auto-replans` are implementation diagnostics rather than the clearest reviewer-facing outcome metrics for the baseline comparison.

## 4. Manuscript Insertion: Scalability Limitations and Future Work

The discussion should explicitly acknowledge that blind explicit-state DFS will not scale gracefully as the number of robots, buffers, and intermediate transport states grows. A concise paragraph can be inserted as follows.

### 2.X Scalability Limits of Explicit Enumeration

The present case study intentionally targets a small dual-robot cell to demonstrate end-to-end feasibility. For larger manufacturing cells, explicit-state reachability over the full centralized product space will encounter the expected state-explosion bottleneck. Future scaling should therefore replace or augment explicit DFS with more compact verification methods, including BDD-based symbolic reachability, compositional or assume-guarantee reasoning across resource subsystems, and partial-order or local-monitor decompositions that avoid constructing the full global product when independence can be exploited.

This paragraph is important because it clarifies that the current implementation is real, but not presented as the final answer for industrial-scale verification.

## 5. Manuscript Insertion: Shielded Stochastic Search and Bounded Repair

The current codebase supports a stronger and more concrete discussion of the replanning loop.

### 2.X Shielded Stochastic Search

Our repair loop resembles Counterexample-Guided Inductive Synthesis in that the verifier returns a concrete counterexample trace, but it differs in a crucial respect: the synthesizer is replaced by a stochastic LLM. Consequently, the framework does not provide a formal convergence guarantee. The LLM may fail to repair a plan, may generate another unsafe plan, or may revisit a previously rejected structure.

We therefore frame the architecture as a **shielded stochastic search** rather than a complete synthesis procedure. The LLM is responsible for proposing semantically meaningful plan repairs, while the Central Controller Agent remains the formal safety gate: every candidate plan must pass the DFA-based validation stage before execution.

The implementation adds two practical safeguards.

1. **Bounded repair attempts.** Offline and runtime repair are both capped at five automatic attempts before the system pauses for human intervention. In the current implementation, this bound appears as `auto_replan_max_attempts = 5` in the offline bundle workflow and `_runtime_repair_max_attempts = 5` in the `ProductAgent`.
2. **Oscillation / no-change detection.** The repair loop hashes the task-node structure through `_task_nodes_hash()` and aborts when a repair attempt fails to change the candidate plan, preventing infinite repetition of the same unsafe solution.

With these safeguards, the worst-case number of candidate validations per episode is at most \(N_{repair} + 1\), where \(N_{repair}\) is the configured repair-attempt bound. Each validation still incurs the explicit DFS cost given in Section 2.X above.

This language is more precise than implying any formal convergence theorem. It makes clear that the system is bounded and safe-by-rejection, but not guaranteed to synthesize a valid plan in finite time for every task instance.

## 6. Manuscript Insertion: Clarifying That the Case Study Is Implemented

The case-study section should explicitly state that the dual-robot study is not only conceptual. Suggested wording:

### 2.X Practical Case-Study Scope

The dual-robot assembly study is implemented in practice using the Gazebo/ROS2/SPADE software stack described in the architecture section. The experiments operate on generated task-plan bundles, compiled safety automata, verifier-produced witness traces, and persisted trial artifacts. We therefore present the case study as an implemented feasibility demonstration rather than a purely illustrative example. At the same time, we emphasize that the chosen two-robot setup is intentionally small and should not be interpreted as evidence that explicit centralized DFS alone is sufficient for large industrial cells.

## 7. Code-Level Evidence for the Revision

The following code paths support the revised discussion:

- Explicit-state DFS bound: `cais_spade_llm/agents/central_controller/plan_safety_validator.py`
- Product-state cap: `PlanSafetyValidator.MAX_PRODUCT_STATES = 200000`
- Offline bounded repair loop: `cais_spade_llm/bundles/bundle_compiler.py`
- No-change oscillation guard: `BundleCompiler._task_nodes_hash()`
- Runtime repair bound: `cais_spade_llm/agents/intelligent_product/product_agent.py`
- Approved-rule offline baseline and verifier comparison: `cais_spade_llm/experiments/offline_study.py`

## 8. Important Experimental Note Before Reporting S1

Do not report the current S1 invalid rate until the precedence semantics for `before` rules have been corrected and the offline study has been rerun. The witness trace currently suggests that S1 was misclassified by the old ordering interpretation, so the previously generated S1 invalid result should be treated as stale.
