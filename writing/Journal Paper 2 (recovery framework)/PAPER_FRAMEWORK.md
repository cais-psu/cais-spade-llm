# Paper framework and implementation scope

Source: *A Neuro-Symbolic Architecture for Safe and Feasible Self-Recovery in Multi-Agent
Manufacturing Systems* (Lim, Shih, Zhao, Watts, Meira-Góes, Kovalenko), RCIM preprint draft,
shared on October 19, 2026. Sections 2.3, 2.4 and 6.2–7 were still empty in that draft.

Use this file to keep implementation work inside the paper's framework. Before adding a mechanism,
name the section or equation it implements. If it maps to none, ask the user first.

## Research challenges and contributions

- **RC1.** Identify actions beyond the predefined process models. An unexpected disruption, such as
  a dropped part or an unavailable resource, can put the system in an undefined state that
  model-based replanning cannot leave.
- **RC2.** Guarantee that LLM-proposed actions are valid transitions, physically feasible, and safe
  under process and safety requirements.
- **Contributions.** (1) A neuro-symbolic MAS framework for recovering from disruptions outside
  predefined behavior. (2) An LLM approach that proposes recovery actions from the current state and
  resource capabilities. (3) DES-based validation that forms a valid recovery sequence before execution.

## Formal model (Section 3)

- **FSA** $G=(X,E,Tr,x_0,X_m)$ with generated and marked languages $L(G)$, $L_m(G)$. A deadlock is
  a state with no enabled event.
- **Atomic propositions** $AP$, labeling $\ell:X\to 2^{AP}$, trace
  $\pi(s)=\ell(x_0)\cdots\ell(x_m)$. **LTL$_f$** specifications $\Phi$, each compiled to a DFA
  $\mathcal{A}_\varphi$.
- **Plants.** Each resource is $G_i=(X_i,E_i,Tr_i,x_{0,i},X_{m,i})$ with task-level events, and the
  joint plant is $G=\Vert_i G_i$.
- **Unexpected faults.** These are not in the model. Once observed, the fault event and state are
  added, $\phi_{j,r_i}\in E_{f,r_i}$ and $x_{F,r_i}\in X_{f,r_i}$, with
  $Tr_{r_i}(x_{j,r_i},\phi_{j,r_i})=x_{F,r_i}$ (Eqs. 2–3). They cannot be prevented once the
  physical condition occurs.
- **Recovery events.** A recovery event $e^R_k\notin E_i$ is a proposed transition $x\to x'$ with no
  predefined implementation. Before it enters the plant, it must be feasible and safe.
- **Feasibility.** It uses a finite set of parameterized primitives $\mathcal{P}_i$, resource
  condition $v_i\in V_i$, parameter domain $\Theta_{i,\rho}$, assignment $\theta_\rho$, composition
  $PC_k=\langle\rho_1(\theta_1),\dots,\rho_h(\theta_h)\rangle$ (Eq. 4), and resource-condition
  transition $\mu_i$.
- **Assumptions.** A.1: the post-fault state and relevant resource conditions are observable, so
  detection and estimation are out of scope. A.2: every resource provides finite parameterized
  primitives with known preconditions, constraints and effects.
- **Problem.**
  - Eq. 5: recovery string $s^R=e^R_1\cdots e^R_m$.
  - Eq. 6: $\exists s_N\in E^*: s_F s^R s_N\in L_m(G^R)$.
  - Eq. 7 (feasibility): $\forall k\,\exists PC_k: \mu_{i(k)}(v_{i(k)},PC_k)=v'_{i(k)}$.
  - Eq. 8 (safety): $\forall\varphi\in\Phi: \pi(s_F s^R s_N)\in\mathcal{L}(\mathcal{A}_\varphi)$.
- **Illustrative example (Figs. 2, 8).** Two robots and an assembly-order rule. A fault leaves a
  deadlock. In the recovery, $r_1$ returns to $x_{0,r_1}$ and nominal events then complete its
  placement, so nominal work redoes the interrupted task.

## Architecture (Sections 4–5)

- **Agents.**
  - PA: neural generation of the recovery outline.
  - RA: feasibility validation, then primitive composition.
  - CCA: safety validation, then the safety monitor.
- **Candidate sets.** $R_{\text{neural}}$, $R_{\text{feasible}}$ (Eq. 7), $R_{\text{safe}}$ (Eqs. 6
  and 8), and $R^*=R_{\text{feasible}}\cap R_{\text{safe}}$ (Eq. 9). An accepted path is
  $x_F\to x^R_1\to\cdots\to x^R_m$ (Eq. 10).
- **Incremental synthesis (Section 5.1, Fig. 5).**
  - The PA updates the knowledge base $\mathcal{K}$ with the failure context, observed conditions,
    remaining goals and constraints, then starts at $x^R_0=x_F$.
  - The LLM proposes $e^R_k$ from the projected state $x^R_{k-1}$ and the resource capabilities.
    Its output schema gives the assigned resource and intended start and end states.
  - Accepted events extend the prefix $s^R_k=s^R_{k-1}e^R_k$ (Eq. 11). Rejections return feedback.
  - Synthesis continues until the projected state connects to the remaining product plan.
  - Fig. 5's knowledge base holds: the current system state (resource, product, fault states,
    goals and marked states, constraints and guards, observations); resource capabilities
    (reachable locations, workspace bounds); and the output schema.
- **Feasibility validation (Algorithm 1).**
  - Backward derivation collects primitives of transitions $Tr_i(x,e)=x'$ with $\nu_i(x')=v'_i$.
  - Forward validation requires
    $\forall\rho\,\exists\theta_\rho\in\Theta_{i,\rho}: C_\rho(v_i,v'_i,\theta_\rho)=1$ (Eq. 12).
- **Safety validation.** The CCA extends $\pi(s_F s^R_{k-1}e^R_k)=\pi(s_F s^R_{k-1})\ell(x^R_k)$
  (Eq. 13) and checks that an accepting state stays reachable in each monitor.
- **Primitive composition (Section 5.2).** The RA's LLM orders primitives and grounds parameters
  from $\mathcal{K}$, then checks structure, bindings and Eq. 7.
- **Safety monitoring (Section 5.3).** The RA supplies physical evidence (trajectories, geometry),
  which is mapped to the predefined APs. A local parallel composition covers recovery
  alternatives, running tasks and the steps needed for completion, with monitors starting from
  $s_F$.

## Case study (Section 6)

- **Cell (Gazebo).** KMR iiwa, a conveyor, ur5e-1 to ur5e-4, M1 and M2, Storage, a four-zone buffer,
  a printer, Exit, and NIST Task Board 1 parts.
- **Primitives (Table 2).**

  | Resources | Primitives |
  |---|---|
  | UR5e robots | `detect_parts`, `compute_pick_targets`, `compute_place_targets`, `move_to_named_pose`, `move_cartesian`, `move_relative`, `grasp_part`, `release_part` |
  | KMR | the same, plus `move_base` |
  | M1, M2 | `select_program`, `start_job`, `pause_job`, `resume_job`, `stop_job` |
  | Conveyor, buffer | `move_relative` |
  | Printer, Storage, Exit | none |

- **Failure scenarios.** Conveyor breakdown, robot breakdown (ur5e-1), machine breakdown, and part
  slippage.
- **Evaluation.** Performance, model comparison, ablation, discussion and limitations.

## How the current implementation maps to the paper (October 19, 2026)

- **Validation-off drafts** produce $R_{\text{neural}}$ samples, which feed the ablation and model
  comparison. RC2 checks (Eqs. 7, 8, 12, 13) are off in these runs.
- **Prompt and $\mathcal{K}$.** The Failure section holds the observed fault and conditions (A.1).
  The Resources and Parts sections hold the projected state $x^R_{k-1}$. A resource's
  `capabilities` lists reachable locations, workspace bounds, positions it moves to, and processes.
  The Knowledge Base section holds the lookups into $\mathcal{K}$. `expected_end_state` is the
  intended end state $v'_i$ and the part conditions: declared variables only (the model's state
  variables plus `pose` from primitive effects and `processCompleted` from product effects). The RA
  later finds the primitive parameters $\theta$ (Eq. 12); the LLM never writes them.
- **`neighbor_end_state`** (October 20; named `place_end_state` until October 21): an event may state
  how moving its part changes another resource, its neighbor at the shared state, for example a buffer
  slot. It is that resource's part $\nu_j(x^R_k)$ of the intended end state in the
  joint plant $G=\Vert_i G_i$. That resource does not act, so it has no composition (Eq. 7 covers
  $i(k)$ only); its values enter the projected state and the handoff condition $H$, and should enter
  the CCA labels (Eq. 13) once the check reads them. The nominal models already change other resources
  in one event (a delivery sets the buffer's slot).
- **Eq. 6** maps to the bounded search for connecting nominal events (at most 6) and the handoff
  check. Full completion to a marked state, and Eq. 8 over $s_N$, are not checked yet (CCA).
- **"Connected to the remaining product plan" is not defined in the paper.** The implementation
  uses:
  - replaced work $W=\bigcup_i B_i$, where $B_i$ is the longest prefix of resource $i$'s remaining
    tasks that needs a faulted resource;
  - each part's next remaining step $\tau^*_p$;
  - the handoff condition $H$, built from the guards of the $\tau^*_p$ and the entries that handle
    no part.

  The paper may want this definition.
- **Termination differs.** With validation off, the LLM ends the outline with `recovery_complete`
  and code only records whether nominal work can connect. The paper describes termination as
  "until connected", which is a code criterion. Decide which rule the validated mode uses.

## Scope guard for implementation

- **Outline stage.** It proposes task-level recovery events and their end states, stated as
  outcomes. Primitives appear only at composition (Section 5.2).
- **Feasibility belongs to RA validation** (Algorithm 1, Eq. 7), and **safety belongs to CCA
  validation** (Eqs. 8, 13). Do not add other judgment checks or heuristics to the prompt or the
  PA. With validation off, code only projects state and records results.
- **No failure-type, resource-type or scenario logic.** Faults are unexpected but observable (A.1).
  Facts come from measurements and declarations, stated the same way for every resource.
- **Out of scope:** process-chain propagation (future work), optimization or scheduling objectives,
  human-in-the-loop steps, and fault detection or state estimation (A.1).
- **Prompt wording uses plain terms.** Each term maps to a paper concept (state $\nu_i(x)$,
  capabilities, $\mathcal{K}$, $e^R_k$, $x^R_k$, the handoff), without DES jargon in the prompt
  itself.
