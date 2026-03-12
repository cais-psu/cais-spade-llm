# Paper Outline: CAIS-SPADE-LLM

## Working Title

*Layered Recovery in Multi-Agent Manufacturing: Integrating DES Replanning with LLM-Based Reasoning Under Formal Safety Constraints*

---

## Target Venue

> **TODO**: Choose target venue. Candidates:
> - **Journal**: IEEE Transactions on Automation Science and Engineering (T-ASE), Journal of Manufacturing Systems (JMS), Robotics and Computer-Integrated Manufacturing (RCIM)
> - **Conference**: IEEE CASE, IEEE ITSC, IFAC MIM, IEEE ICRA (if robotics-focused)

---

## Section Map

Each section below maps to the code/artifacts that **support the claims made** in that section.

### 1. Introduction

**Claim**: Manufacturing systems need recovery mechanisms that combine formal correctness with the flexibility to handle novel failures.

**Pitch**: Existing approaches are either formal but brittle (pure DES/FSA — can't handle out-of-vocabulary failures) or flexible but unsafe (pure LLM — hallucinates infeasible or unsafe plans). This paper presents a layered architecture that gets both.

**Code references (for your own tracing, not for the paper)**:
- System overview: [`cais_spade_llm/ui_main.py`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/ui_main.py)
- Agent creation: [`cais_spade_llm/agent_creator.py`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agent_creator.py)

---

### 2. Related Work

**Key areas to cover**:
- Multi-agent manufacturing (holonic, PROSA, ADACOR, SPADE-based)
- DES in manufacturing (supervisory control, BFS-based planning)
- LLMs for robotics/planning (SayCan, Code-as-Policies, ProgPrompt, Inner Monologue)
- Formal safety in autonomous systems (LTLf, DFA runtime monitoring)
- **Gap**: No existing work combines DES replanning + LLM bridge + formal DFA safety in a single recovery architecture

---

### 3. System Architecture

**Claim**: A multi-agent system where product agents, resource agents, and a central controller collaborate via XMPP messaging.

**Subsections**:

#### 3.1 Agent Model
- Intelligent Product Agent (planning, task dispatch, replanning)
- Resource Agents (UR5e, xArm6, sensor, conveyor)
- Central Controller Agent (coordination, safety enforcement)

**Code**:
- [`agents/intelligent_product/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/intelligent_product/)
- [`agents/resource_agent/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/resource_agent/)
- [`agents/central_controller/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/)

#### 3.2 Planning Pipeline
- NL requirements → LLM parsing → structured requirements → task DAG → global FSA
- Geometry enrichment from product specifications

**Code**:
- [`agents/intelligent_product/process_planner.py`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/intelligent_product/process_planner.py)
- [`specification/products/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/specification/products/)
- [`prompts.py`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/prompts.py)

#### 3.3 Execution Platform
- Dual-robot simulation (UR5e + xArm6 in Gazebo)
- ROS 2 / MoveIt integration

**Code**:
- [`ros2/`](file:///home/jongh/projects/cais-spade-llm/ros2/)
- [`resources/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/resources/)

---

### 4. Safety Framework (LTLf / DFA)

**Claim**: Safety constraints are specified in natural language, compiled to LTLf formulas and DFA monitors, and enforced at both plan time and runtime.

**Subsections**:

#### 4.1 Safety Specification Pipeline
- NL safety rules → LLM parsing → structured rules → atomic propositions → LTLf → DFA

#### 4.2 Offline Validation
- DAG/FSA checked against DFA before execution begins

#### 4.3 Runtime Monitoring
- DFA state advanced on each execution event; violations trigger replanning

**Code**:
- [`agents/central_controller/safety_logic.py`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/safety_logic.py)
- [`agents/central_controller/offline_safety_validator.py`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/offline_safety_validator.py)
- [`agents/central_controller/online_safety_monitor.py`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/central_controller/online_safety_monitor.py)
- [`specification/safety/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/specification/safety/)

---

### 5. Layered Recovery Architecture (Core Contribution)

**Claim**: A three-layer recovery architecture where each layer adds flexibility at the cost of formal guarantee strength.

#### 5.1 Layer 1 — DES Replanning (Compiled BFS)
- Resource bidding via per-resource DES models
- Environment model M_e compilation from bids
- BFS on M_e to find recovery path
- Fast (milliseconds), formally correct, but limited to in-vocabulary failures

#### 5.2 Layer 2 — LLM Bridge (Guided Exploration)
- Invoked only when Layer 1 finds no path
- LLM generates synthetic bridge events to extend M_e
- **Validation pipeline** (Tiers 1–4): schema check → DFA safety pre-check → re-query loop → runtime checkpointing
- Slower (seconds), flexible, safety-validated

#### 5.3 Layer 3 — Human Escalation
- When both automated layers fail, system escalates to human operator
- Operator console UI for inspection and manual override

**Code**:
- [`agents/intelligent_product/replanner/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/agents/intelligent_product/replanner/)
- [`prompts.py` (LLM bridge prompts)](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/prompts.py)
- [`monitor/`](file:///home/jongh/projects/cais-spade-llm/cais_spade_llm/monitor/)

**Design note** (from [`TODO/safety_constraints_in_replanning.md`](file:///home/jongh/projects/cais-spade-llm/TODO/safety_constraints_in_replanning.md)):
- DFA states augment BFS to prune unsafe transitions
- Validation pipeline tiers for LLM bridge output

---

### 6. Experiments and Evaluation

> Full experiment design is in [`TODO/experimental_setup_and_results.md`](file:///home/jongh/projects/cais-spade-llm/TODO/experimental_setup_and_results.md)

#### 6.1 Experimental Platform
- UR5e + xArm6 assembly task (SG + MCP on assembly board)
- Failure taxonomy: F1–F6 (in-vocabulary → out-of-vocabulary → unrecoverable)

#### 6.2 Experiment 1: Layer Effectiveness *(core result)*
- Baselines: pure LLM, pure DES, full layered system
- Metrics: recovery success rate, latency, plan safety, plan feasibility, plan optimality

#### 6.3 Experiment 2: Validation Pipeline Ablation
- Progressive validation tiers V0–V4
- Metrics: rejection rate per tier, final success rate, safety violation rate, re-query count

#### 6.4 Experiment 3: Recovery Latency Breakdown
- Per-phase timestamping (failure detection → bid computation → BFS → LLM call → validation)
- Waterfall charts per failure type

#### 6.5 Experiment 4: Safety Constraint Enforcement
- Injection of safety-tempting failures
- Compare: pure LLM (no DFA) vs full system
- Metrics: safety violation count, violation type, recovery after rejection

#### 6.6 Experiment 5: Scalability *(if time permits)*
- Scale parts (2→8), robots (2→3), DFAs (1→4)
- Metrics: BFS states, bid time, M_e size, total replanning time

---

### 7. Discussion

**Points to address**:
- Why layered > monolithic (formal + flexible, try cheaper layer first)
- Limitations of LLM bridge (hallucination rate, latency)
- Generalizability beyond assembly tasks
- Path toward shared typed IR unifying product and safety semantics (see [`TODO/shared_typed_ir_for_product_and_safety.md`](file:///home/jongh/projects/cais-spade-llm/TODO/shared_typed_ir_for_product_and_safety.md))

---

### 8. Conclusion and Future Work

**Future work candidates**:
- Shared typed IR for product + safety (already designed in TODO)
- DES bidding for offline initial planning (already designed in TODO)
- Physical perception pipeline completion
- Multi-product concurrent scheduling

---

## Figures to Produce

| ID | Description | Source |
|----|-------------|--------|
| Fig 1 | System architecture diagram (agents, messaging, layers) | Hand-drawn or auto-generated |
| Fig 2 | Planning pipeline flow (NL → requirements → DAG → FSA) | From `process_planner.py` flow |
| Fig 3 | Safety compilation pipeline (NL → rules → APs → LTLf → DFA) | From `safety_logic.py` flow |
| Fig 4 | Layered recovery architecture (Layer 1 → 2 → 3) | Core contribution figure |
| Fig 5 | Recovery success rate table (Exp 1) | From experiments |
| Fig 6 | Validation pipeline stacked bar (Exp 2) | From experiments |
| Fig 7 | Latency waterfall (Exp 3) | From experiments |
| Fig 8 | Safety violation comparison (Exp 4) | From experiments |
| Fig 9 | Example DFA for SAFE_1 constraint | From `safety/SAFE_1_dfa.dot` |
