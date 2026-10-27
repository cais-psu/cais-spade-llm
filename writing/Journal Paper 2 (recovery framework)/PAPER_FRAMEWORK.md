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
  - CCA: safety and nonblocking validation (Eqs. 6, 8, 13), then the safety monitor (decided by the user,
    October 25; see "Nonblocking validation").
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
- **Feasibility validation, revised wording (decided by the user, October 24–26; to be written into the
  paper).**
  - Backward derivation: Algorithm 1 reads "primitives whose declared effects produce $v'_i$". It goes
    from each changed condition $c$ of $v'_i$ to $P_c$, the primitives of resource $i(k)$ whose declared
    effects make the stated value. A recovery event is a new transition, so derivation never goes
    through nominal events.
  - Eq. 12 becomes
    $\forall c\,\exists\rho\in P_c\,\exists\theta_\rho\in\Theta_{i,\rho}: C_\rho(v_i,v'_i,\theta_\rho)=1$:
    one feasible producer per condition, with no order between them and no chaining through a failed
    precondition (the RA is a verifier, not a planner).
  - A task-level state label that no primitive produces (`resource_state`) is not derived.
  - Location, slot and custody have no producing effect (primitives are not symbolically connectable).
    They are decided by the acting resource's capability: a known place, kinematic reach over its own
    chain, and navigation for a mobile base.
  - Delayed completion needs a definition in the paper: a machine's job completes after the remaining
    processing time, $(1-\text{processing fraction})\times\text{processing time}$, with the part's
    `processCompleted`, as the machine's observed, uncontrollable `complete_job` outcome while running.
  - Candidates are checked in the LLM's order and the first feasible one is applied
    ($R_{\text{neural}}\cap R_{\text{feasible}}$). Facts return as feedback only when every candidate
    fails.
  - An unresolved search (NEEDS_CONTEXT) is a rejection, like INFEASIBLE: a finite search does not
    prove a pose reachable. A name the model does not declare is rejected, because Algorithm 1 has
    nothing to derive from it.
- **Safety validation.** The CCA extends $\pi(s_F s^R_{k-1}e^R_k)=\pi(s_F s^R_{k-1})\ell(x^R_k)$
  (Eq. 13) and checks that an accepting state stays reachable in each monitor.
- **Nonblocking validation (Eq. 6; decided by the user, October 25; to be written into the paper).**
  - Who: the CCA. In supervisory control, admissible behavior must be both safe and nonblocking: it
    must never produce an illegal string and must always be able to complete its task (reach a marked
    state). Both are properties of the composed plant $G^R=\Vert_i G_i^R$, which only the CCA holds.
    RA validation is local by construction (Algorithm 1 and Eq. 12 judge whether resource $i(k)$ can
    realize $e^R_k$ with its own primitives, not whether other resources' remaining tasks stay
    executable), and the PA generates the recovery, so it does not validate its own proposals. The
    paper already defines $R_{\text{safe}}$ by Eqs. 6 and 8.
  - When: once, when the LLM declares the recovery complete. Intermediate states $x^R_k$ are generally
    not connected to the nominal plan (a recovery pauses jobs, unloads parts or moves resources aside
    first), so a nominal continuation after every event would reject valid recoveries. Requiring a
    continuation through further recovery events would mean searching for the recovery itself, the
    LLM's role (RC1), which is also why primitives are not symbolically connectable. Per event the CCA
    enforces only Eq. 13. There is no per-event progress check (user, October 25): it would be a
    distance-to-goal heuristic, the kind of hidden rule removed on October 22.
  - What: whether the remaining planned events $s_N$ lead from $x^R_m$ to a marked state of $G^R$. If
    not, the declaration is rejected and the CCA returns what is blocked as feedback (facts only): the
    planned task that cannot start, or the part that cannot reach its remaining requirement from where
    it is, never the conditions they need (user, October 26: those describe the end state other
    resources must reach, close to an answer); synthesis continues from $x^R_m$. The PA supplies the remaining plan and
    the handoff condition $H$, and ends synthesis only when the CCA accepts.
  - A part left where no nominal event acts on it blocks $s_N$ even when $H$ holds (user, October 26).
    Example: the trimmed KET4 left on M2's staging tray, which in the nominal models holds only M2's
    parts. The check runs on the nominal models, with each part's remaining requirements as the goal;
    their declared domains and guards decide that the part is blocked, never a rule about places, and
    the refusal states the part, where it is and its remaining requirement.
  - Not a specification: completion is expressed by the marked states of the resource and product
    models. The specifications in $\Phi$ are safety properties checked on prefixes by monitors;
    completion is a liveness requirement already encoded by $L_m(G^R)$, and stating it as a
    specification would need a failure-specific goal formula.
  - Paper edits this implies: the CCA's role reads "safety and nonblocking validation (Eqs. 6, 8,
    13)"; Fig. 5 and Section 5.1 end synthesis when the CCA accepts Eq. 6, not when "the projected
    state connects to the remaining product plan"; the paper defines the remaining planned events
    $s_N$ (see below).
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
- **Pose places** (user, October 24). A part released where no declared place names the spot rests at
  the resource's stated release `pose`: its location is empty and its observed pose is that pose, as
  part conditions otherwise follow from custody. Eq. 12 checks that the pose is reachable and that a
  recorded surface supports the part there with nothing overlapping it.
- **`neighbor_end_state`** (October 20; named `place_end_state` until October 21): an event may state
  how moving its part changes another resource, its neighbor at the shared state, for example a buffer
  slot. It is that resource's part $\nu_j(x^R_k)$ of the intended end state in the
  joint plant $G=\Vert_i G_i$. That resource does not act, so it has no composition (Eq. 7 covers
  $i(k)$ only); its values enter the projected state and the handoff condition $H$, and should enter
  the CCA labels (Eq. 13) once the check reads them. The nominal models already change other resources
  in one event (a delivery sets the buffer's slot).
- **Algorithm 1 and Eq. 12** map to `recovery_framework/ra_feasibility.py` (`check_event`), as revised
  above. It runs behind the session flag `check_ra_feasibility` (off by default; an Offline-only switch on
  the Recovery outline card). Known limits that answer NEEDS_CONTEXT (pose-place support, the KMR base
  search, tilted grasps) are left until a checked run hits one.
- **Eq. 6** maps to the CCA's nonblocking check at `recovery_complete` (built October 26, behind the
  session flag `check_nonblocking`, off by default; `recovery_framework/cca_nonblocking.py`). It
  composes the runtime resource models (the ones normal production and Eq. 13 admission use) for the
  parts the recovery touched or the remaining plan names, starting from $x^R_m$, and accepts when
  $x^R_m$ lies in the composition's nonblocking region: every such part can still reach its remaining
  requirement and every resource it uses can return to rest, whatever completions occur. The search is
  exhaustive (no step bound, no NEEDS_CONTEXT); a refusal names a resource left in a state its model
  does not have, a part that cannot reach its remaining requirement from where it is, or a resource
  that cannot carry on. The runtime models are capability models, not the scenario's routes, so a part
  left off its usual route is accepted when some resource can carry it on. Without the flag code only
  records the older diagnostic: a bounded search (at most 6 nominal events) toward the handoff
  condition $H$. Eq. 8 over $s_N$ is not checked: the selected predefined specifications use physical
  propositions the composition cannot evaluate.
- **Eq. 13** maps to the CCA's per-event safety check (built October 26, behind the session flag
  `check_cca_safety`, which needs `check_ra_feasibility`; `recovery_framework/cca_safety.py`). Candidates
  are checked in the LLM's order, RA first, and the first one both pass is applied (Eq. 9). The
  predefined specifications are compiled and grounded as the nominal physical gate grounds them, and
  their monitors start at $s_F$, replayed from the capture's recorded transitions. Labels $\ell(x^R_k)$
  come from one reader per kind of condition: the projected `processCompleted` ledger, and physical
  evidence for a resource in a region or a part entering one (unmoved resources keep their recorded
  collision boxes; the acting robot is placed by forward kinematics at the RA's reach configuration for
  each work point and at its end pose). An event is judged during it, after it, and once more for each
  uncontrollable event its end state enables, as if that event then happened on its own. A refusal
  states the broken specification and the facts of the step that breaks it. Decided by the user on
  October 26, to be written into the paper:
  - Eq. 13 also covers the uncontrollable events $x^R_k$ enables (a running job that completes on its
    own): the CCA cannot prevent them, so a safe state must stay safe under each of them
    (controllability).
  - The outline uses the RA's reach configuration as the robot's posture; the executed trajectory is
    checked again at composition (Section 5.3).
  - Captures are taken with the CCA bypassed, so a failure state may already break a specification
    (Part slippage: ur5e-3's arm, waiting with KET4 at buffer zone 4, reaches into the board area while
    ur5e-4 is over it). The check then stops unless an assumption is declared; for Part slippage the
    user chose to assume the CCA was on and ur5e-3 waited clear of the board area until it moves.
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
  "until connected", which is a code criterion. Decided October 25 (user): in the validated mode the
  LLM declares completion and the CCA accepts or rejects it by Eq. 6; the paper's Fig. 5 wording
  changes to match.

## Scope guard for implementation

- **Outline stage.** It proposes task-level recovery events and their end states, stated as
  outcomes. Primitives appear only at composition (Section 5.2).
- **Feasibility belongs to RA validation** (Algorithm 1, Eq. 7), and **safety and nonblocking belong
  to CCA validation** (Eqs. 6, 8, 13). Do not add other judgment checks or heuristics to the prompt or the
  PA. With validation off, code only projects state and records results.
- **No failure-type, resource-type or scenario logic.** Faults are unexpected but observable (A.1).
  Facts come from measurements and declarations, stated the same way for every resource.
- **Out of scope:** process-chain propagation (future work), optimization or scheduling objectives,
  human-in-the-loop steps, and fault detection or state estimation (A.1).
- **Prompt wording uses plain terms.** Each term maps to a paper concept (state $\nu_i(x)$,
  capabilities, $\mathcal{K}$, $e^R_k$, $x^R_k$, the handoff), without DES jargon in the prompt
  itself.
