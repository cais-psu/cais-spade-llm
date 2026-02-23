# TODO: Experimental Setup and Results for Journal Paper

## Goal

Design experiments that demonstrate **why the DES + LLM layered architecture matters** — not just that it works, but that it outperforms the alternatives and that each component contributes.

---

## Experimental Platform

### Hardware / Simulation
- **ProtoTwin simulation**: UR5e + xArm6 collaborative assembly (see `prototwin_multi_robot_demo.md`)
- **Assembly task**: SG + MCP parts onto assembly board, with safety constraint SAFE_1 (SG before MCP)
- **Real hardware** (if available): Same task on physical UR5e + xArm6 for validation subset

### Failure Scenarios to Implement

Define a **taxonomy of failure types** that map to the three recovery layers:

| ID | Failure Type | Description | Expected Recovery Layer |
|----|---|---|---|
| F1 | Part slippage (in-vocabulary) | SG slips during placement, lands at known location within workspace | Layer 1 (DES BFS) |
| F2 | Gripper fault (in-vocabulary) | Gripper fails to close, pick action fails, part remains at source | Layer 1 (DES BFS) |
| F3 | Part displaced (out-of-vocabulary) | SG knocked to unexpected location, detected by camera at XYZ coordinates | Layer 2 (LLM bridge) |
| F4 | Partial assembly failure | Part placed but not secured, state ambiguous between "placed" and "assembled" | Layer 2 (LLM bridge) |
| F5 | Cascading failure | Robot A fails mid-task, blocking Robot B's predecessor dependency | Layer 1 or 2 depending on state |
| F6 | Unrecoverable | Part fallen outside all robot workspaces, no camera visibility | Layer 3 (human escalation) |

Each failure should be **reproducibly injectable** via configuration (like existing `sg_slippage_mode`).

---

## Experiment 1: Layer Effectiveness (CRITICAL)

### Question
Does the layered architecture (DES + LLM) outperform either component alone?

### Baselines
1. **Pure LLM replanning**: On failure, send full system state to LLM, ask it to generate a complete recovery plan (task sequence) directly. No DES, no BFS, no M_e. This is what most LLM-for-robotics papers do.
2. **Pure DES replanning (Layer 1 only)**: On failure, run DES bidding + BFS. If no path found, fail immediately (no LLM bridge).
3. **Full system (Layer 1 + Layer 2)**: DES BFS first, LLM bridge if stuck.

### Metrics
| Metric | Definition |
|---|---|
| **Recovery success rate** | % of injected failures that reach goal state |
| **Recovery latency** | Wall-clock time from failure detection to recovery plan ready |
| **Plan safety** | % of recovery plans that satisfy all DFA safety constraints |
| **Plan feasibility** | % of recovery plans where every action is physically executable (no infeasible transitions) |
| **Plan optimality** | Number of recovery actions (fewer = better) |

### Protocol
- Run each failure scenario (F1-F6) × each baseline × N trials (N ≥ 10 for LLM variance)
- For pure LLM baseline: use same prompt context (system state, tools catalog, safety rules as text)
- For pure DES baseline: same bidding + BFS, just no LLM fallback
- Record all metrics per trial

### Expected Results
- Pure DES succeeds on F1, F2 but fails on F3, F4 (out-of-vocabulary) → limited success rate
- Pure LLM succeeds broadly but generates unsafe plans (violates SAFE_1) or infeasible transitions at some rate → lower safety/feasibility scores
- Full system matches pure LLM success rate while matching pure DES safety/feasibility → best of both

---

## Experiment 2: Validation Pipeline Ablation

### Question
Does each validation tier contribute to catching bad LLM output?

### Setup
- Use only Layer 2 failures (F3, F4) where LLM bridge is invoked
- Run with progressively more validation tiers enabled:
  - V0: No validation (raw LLM output injected)
  - V1: Schema + DES consistency only (Tier 2)
  - V2: V1 + DFA safety pre-check (Tier 1)
  - V3: V2 + re-query loop (Tier 3)
  - V4: V3 + runtime checkpointing (Tier 4)

### Metrics
| Metric | Definition |
|---|---|
| **Rejection rate per tier** | % of LLM proposals rejected at each validation level |
| **Final success rate** | % of failures recovered after all validation + re-queries |
| **Safety violation rate** | % of executed recovery plans that violate a DFA |
| **Re-query count** | Average number of LLM re-queries before valid output |

### Expected Results
- V0 has highest throughput but worst safety violation rate
- V1 catches structural hallucinations (missing fields, unknown functions)
- V2 catches safety violations (SG/MCP ordering)
- V3 recovers from rejections by giving LLM feedback
- V4 catches world-model errors at runtime

Present as **stacked bar chart**: for each trial, show where in the pipeline the LLM output was caught or passed.

---

## Experiment 3: Recovery Latency Breakdown

### Question
Where does time go during replanning, and is it acceptable for online recovery?

### Setup
- Instrument each phase of the recovery pipeline with timestamps
- Run across all failure types

### Phases to Measure
1. Failure detection (event received → replan triggered)
2. State collection (CCA queries ResourceAgents, ProductAgent builds state)
3. Bid computation (per-resource DES BFS)
4. M_e compilation + global BFS
5. LLM bridge call (if Layer 2) + validation
6. Re-query loop iterations (if validation rejects)
7. FSA recompilation after recovery plan applied

### Presentation
- **Waterfall chart** per failure type showing time per phase
- **Table** with mean ± std for each phase across trials
- Highlight that Layer 1 recovery is fast (ms) while Layer 2 adds LLM latency (seconds) — this justifies trying Layer 1 first

---

## Experiment 4: Safety Constraint Enforcement

### Question
Do DFA safety constraints actually prevent unsafe recovery plans?

### Setup
- Inject failures that create temptation to violate safety:
  - F5-variant: UR5e has MCP ready to assemble, but SG (xArm6's job) has failed. Pure LLM might suggest assembling MCP first.
  - Concurrent pick scenario: Both robots idle, both could pick simultaneously (violates mutex)

### Baselines
- Pure LLM (no DFA checking): Does it violate SAFE_1 or mutex?
- Full system with DFA validation: Are violations caught?

### Metrics
| Metric | Definition |
|---|---|
| **Safety violation count** | Number of plans that violate any DFA, per baseline |
| **Violation type breakdown** | Which specific DFA rules are violated |
| **Recovery after rejection** | When DFA rejects a plan, does re-query produce a safe alternative? |

---

## Experiment 5: Scalability (if time permits)

### Question
How does recovery planning scale with system complexity?

### Variables
- Number of parts: 2, 4, 6, 8
- Number of robots: 2, 3 (if ProtoTwin supports)
- Number of safety DFAs: 1, 2, 4

### Metrics
| Metric | Definition |
|---|---|
| **BFS states explored** | Size of visited set in `compute_bid()` and `plan_on_environment_model()` |
| **Bid computation time** | Per-resource DES search time |
| **M_e size** | Number of states and transitions in compiled environment model |
| **Total replanning time** | End-to-end recovery latency |

### Expected Results
- Bid computation scales with per-resource state space (independent of other robots) → linear in N robots
- M_e compilation is linear in total bid size
- BFS on M_e scales with M_e size (not exponential, since M_e is constructed from bids, not full product)
- Contrast with hypothetical full parallel composition cost

---

## Experiment 6: DES vs LLM for Offline Planning (if `des_bidding_for_offline_planning.md` is implemented)

### Question
Can DES bidding replace LLM for initial plan generation?

### Setup
- Same assembly task, no failures
- Compare: LLM-generated DAG vs DES-generated DAG (from `des_bidding_for_offline_planning.md` TODO)

### Metrics
| Metric | Definition |
|---|---|
| **Plan correctness** | Does the generated DAG pass offline FSA + DFA validation without repair? |
| **Planning time** | Time to generate plan (LLM call vs DES BFS) |
| **Plan optimality** | Number of tasks in DAG |
| **Repair iterations** | Number of LLM repair cycles needed after FSA validation (LLM baseline only) |

### Expected Results
- DES plans are correct by construction (0 repair iterations)
- LLM plans may need 1-2 repair iterations
- DES planning is faster (no LLM call for step 2)
- Both produce similar-quality plans

---

## Results Presentation Summary

| Figure | Type | Shows |
|---|---|---|
| Fig 1 | Table | Recovery success rate × failure type × baseline (Exp 1) |
| Fig 2 | Grouped bar chart | Safety + feasibility scores per baseline (Exp 1) |
| Fig 3 | Stacked bar chart | Validation pipeline catch rates per tier (Exp 2) |
| Fig 4 | Waterfall chart | Recovery latency breakdown per phase (Exp 3) |
| Fig 5 | Table | Safety violation counts per baseline (Exp 4) |
| Fig 6 | Line plots | Scalability curves: states explored, time vs N parts/robots (Exp 5) |
| Fig 7 | Table | Offline DES vs LLM plan quality (Exp 6, if applicable) |

## Statistical Considerations

- LLM outputs are non-deterministic → run **minimum 10 trials** per condition for LLM-involving experiments
- Report mean ± standard deviation for all quantitative metrics
- Use **temperature=0** for LLM calls during experiments for reproducibility (note this in paper)
- For pure DES experiments (deterministic), single run suffices but run 3x to confirm
- Consider **McNemar's test** or **Fisher's exact test** for comparing success rates between baselines (binary outcome, paired samples)

## Minimum Viable Experiment Set (for paper submission)

If time is limited, prioritize in this order:

1. **Experiment 1** (Layer effectiveness) — this is the core result, non-negotiable
2. **Experiment 4** (Safety enforcement) — validates the formal safety claim
3. **Experiment 3** (Latency breakdown) — shows online recovery is practical
4. **Experiment 2** (Validation ablation) — shows validation pipeline contributes
5. **Experiment 5** (Scalability) — strengthens the paper but can be brief
6. **Experiment 6** (Offline DES planning) — only if implemented, otherwise mention as future work
