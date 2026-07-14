# Architecture Boundaries

This file explains where runtime responsibilities live. It is for coding and
debugging after the project is installed. Installation steps belong in
[`README.md`](../README.md); coding rules belong in [`AGENTS.md`](../AGENTS.md).

## Entry Points

- `cais_spade_llm/__main__.py` routes package execution into the UI entry path.
- `cais_spade_llm/ui_main.py` is the application entry point for the NiceGUI UI
  and the headless SPADE agent path. It owns command-line flags, startup
  preparation, shutdown handling, and process cleanup.
- `cais_spade_llm/ui/app.py` registers the NiceGUI shell and the five UI routes:
  `/`, `/control`, `/products`, `/resources`, and `/safety`.
- `Makefile` keeps common local commands: `install`, `run`, `headless`, and
  `bootstrap-gazebo`.

## UI To Runtime

- `cais_spade_llm/ui/bridge.py` owns `SystemBridge`, the public UI-to-runtime
  surface. UI pages should call existing `SystemBridge` methods unless the user
  explicitly asks for a public interface change.
- `SystemBridge` coordinates config files, product order files, safety intent
  preview state, bundle state, agent startup, ROS2 launch/process control,
  `digital twin`, hardware status, function recording, and UI-facing runtime
  status.
- Helper code may live behind `SystemBridge`; keep the public UI surface stable
  and preserve existing actions, resources, states, predicates, and user-facing
  terms.
- `cais_spade_llm/ui/ros2_processes.py` contains ROS2 command rendering, domain
  ID helpers, workspace paths, and launch prerequisite checks used by the bridge.

## Agents And Execution

- `cais_spade_llm/agent_creator.py` builds and wires the SPADE agents.
- `cais_spade_llm/agents/intelligent_product/` owns product planning and recovery.
- `cais_spade_llm/agents/central_controller/` owns plan safety validation and
  runtime safety monitoring.
- `cais_spade_llm/agents/resource_agent/` owns robot-agent execution dispatch.
- `cais_spade_llm/resources/robot/` owns robot tasks, primitives, and controller
  adapters. Robot positions and capabilities should come from JSON manifests,
  not hardcoded constants.

## ROS2 And `digital twin`

- `ros2/cais_lab_robotics/launch/`, `config/`, `rviz/`, `worlds/`, and `scripts/`
  are the repo source files for this project's ROS2 integration.
- `make bootstrap-gazebo` copies those repo files into `~/ros2_ws`, builds the
  workspace, and installs the config/RViz assets used at runtime. Edits made only
  in `~/ros2_ws` are outside this git repository.
- Hardware MoveIt/RViz is the operator surface for `digital twin`; Gazebo is the
  passive mirror unless the user asks for a different architecture.
- UR5e hardware motion uses the RTDE trajectory server and the RG2 bridge in
  `ros2/cais_lab_robotics/scripts/`; xArm6 uses the xArm hardware driver and MoveIt
  path.
- After launch, script, config, world, or RViz edits, run `make bootstrap-gazebo`
  before checking installed workspace behavior.

## Safety Validation And Recovery

- Safety logic is split across `cais_spade_llm/agents/central_controller/`,
  `cais_spade_llm/agents/intelligent_product/`, `cais_spade_llm/specification/`,
  and UI bridge preview/status code.
- Keep validation-stage checks separate from downstream runtime dispatch gating.
- When changing recovery behavior, preserve existing action, resource, state,
  predicate, and bridge-event names exactly.

### Recovery validation authority

| Stage | Authority | Responsibility |
| --- | --- | --- |
| LLM proposal generation | LLM called by PA | In `pure_llm`, authors exactly three one-action candidates and `selected_candidate_index`. In `neurosymbolic`, authors one to `candidate_proposal_budget` materially distinct next-event candidates and does not select one. |
| Syntax and grounding validation | PA | Checks the response shape, exact resource/part/location/named-pose/predicate bindings, and allowed state fields. |
| Transition feasibility | PA | Checks the candidate start against the PA projected state and requires a holder, location, occupancy, gripper, or active-blocker effect. |
| Physical feasibility | Responsible live RA | Refreshes `get_recovery_snapshot()` and applies that RA's `check_recovery_physical_feasibility()`. Missing validation capability fails closed. |
| Local plant model | Responsible live RA | Privately supplies its task-level extended finite automaton with exact state variables, finite domains, current valuation, event guards and updates, controllable/observable classifications, marked conditions, and descriptor fingerprint. The descriptor is not included in the LLM prompt. |
| Candidate safety | Live CCA | Applies `validate_outline_macro_recovery_safety()` with the CCA's loaded safety rules and current monitor state. |
| Model-based selection | PA | In `neurosymbolic`, composes RA-declared local models with product state and compares only PA/RA/CCA-valid projected states using open recovery obligations and recovery-relevant admissible enabled events. |
| Candidate commit | PA | Rechecks the PA state fingerprint. `pure_llm` commits only the valid LLM-selected candidate; `neurosymbolic` commits only one unique nondominated candidate. It never substitutes an invalid candidate. |
| Post-outline safety generation | CCA | Generates the safety bundle for the complete accepted outline. This is separate from candidate validation. |
| Runtime safety gating | CCA | Supervises dispatch and execution-time events. A passed outline candidate is not an execution authorization. |
| Primitive execution | Responsible RA | Generates and executes the resource-owned primitive program after the outline and runtime gates pass. |

The PA sends `recovery_outline_physical_validate` to the candidate's exact
`resource_jid`. The RA replies with `recovery_outline_physical_validated`, its
fresh snapshot, snapshot fingerprint, local recovery DES descriptor, and descriptor
fingerprint. PA-valid and RA-valid candidates are
then sent to the CCA with `recovery_outline_safety_validate`; the CCA replies with
`recovery_outline_safety_validated`, findings, rule identifiers, condition
identifiers, and a safety-rule fingerprint. Requests and replies are correlated
by request id, recovery session id, turn index, PA state fingerprint, and sender
JID. A timeout, malformed reply, wrong sender, stale reply, or unavailable agent
fails closed.

### `pure_llm` and `neurosymbolic`

`pure_llm` is the compatibility baseline: one LLM call authors exactly three
one-action candidates and chooses `selected_candidate_index`. All three still
pass through PA, responsible-RA, and CCA validation; an invalid selected
candidate causes revision instead of substitution.

`neurosymbolic` keeps LLM generation necessary because the recovery event
alphabet is incomplete, but removes selection authority from the LLM. The
manufacturing plant is written conventionally as

`G = G_P || (||_{r in R} G_r)`,

where `G_P` is the product model and every responsible RA owns its local extended
finite automaton `G_r`. The supervised plant is `S/G`; PA does not infer the
semantics of `G_r` from event names, descriptions, resource names, or resource
types. The RA descriptor supplies the variables, domains, valuation, local
alphabet, guards, updates, controllability, observability, and local marked
conditions used by the comparison. Robot alphabets come from task-level
`RobotTaskProgram` transitions rather than controller, observation,
geometry-computation, or primitive-generation functions. Continuous pose and
workspace evidence remain outside the finite DES state and are evaluated by
the RA physical validator. `RobotAgent` declares `gripper_state` because its
task-level transitions and physical validator use it with `held_part`.
`PrintingAgent` expresses `pause_job`, `resume_job`, and `cancel_job` through
`resource_state` only. Its runtime snapshot and primitive execution still keep
`job_state` and `active_job`, but those fields are not part of its private DES
descriptor. Other ResourceAgents declare only the exact finite variables their
own enabledness, marked conditions, safety checks, or physical validator need.
They must implement the descriptor or provide an explicit initialization
descriptor; absence fails closed.

The LLM request contains only compact live symbolic state, observations needed
for grounding, resource capabilities, recovery goals, safety rules, and exact
current validation findings. RA event alphabets, guards, updates, marked
conditions, and descriptor fingerprints remain private in session state and
debug result artifacts.

The structured response schema contains the union of RA-declared state-field
names needed to parse candidates for different resources in one response. That
union is not authorization: PA checks each candidate against only the exact
descriptor owned by its `resource_jid`, including the declared `resource` or
`part` scope. For example, `gripper_state` is valid for a RobotAgent that
declares it, while `job_state` is not accepted for either that robot or the
current PrintingAgent private DES descriptor. Values of resource-specific
fields must belong to the declaring RA's exact finite domain. Adding another
ResourceAgent does not add a PA field rule: its descriptor is the authority.
Accepted declared fields are propagated exactly through PA state, projected RA
snapshots, and CCA pre/end-state projection.

The LLM may author any nonempty exact `event_name` and may optionally author a
new nonempty `resource_state` or `part_state` in a candidate end state. These
state values remain uninterpreted labels and do not modify the live RA
descriptor. A selected value is stored only in the recovery session's projected
state, so the following candidate must match it exactly; it is discarded with
that session. A new event or state label contributes no transition meaning or
selection progress by itself. PA requires another concrete declared effect, or
an exact match that clears a supplied recovery obligation. PA, RA, CCA, and the
selector never infer motion, acquisition, release, safety, or progress from the
text of an authored name.

For composed plant state `q`, `O(q)` is the exact set of open PA recovery goals,
continuation/reentry obligations, and CCA safety-condition identifiers.
`Q_m^R` denotes recovery-compatible marked states. `Gamma^R_{S/G}(q)` is the
CCA-admissible subset of RA-declared controllable events that are enabled by
their exact guards and backward-relevant to `O(q)` and `Q_m^R`. A validated LLM
candidate progresses when it reduces `O(q)` without adding an obligation, or
preserves `O(q)` and enables at least one previously disabled recovery-relevant
event in the successor. The artifact also records the exact RA event consumed by
the proposed transition. An executed event commonly disables itself while
enabling its guarded continuation, so literal enabled-set inclusion would reject
that ordinary DES pattern.

It is enabled through
`cais_spade_llm/initialization/recovery_outline_experiment_settings.json`:

```json
{
  "enabled": true,
  "recovery_selection_mode": "neurosymbolic",
  "action_horizon": 1,
  "candidate_count": "adaptive",
  "candidate_proposal_budget": 5
}
```

`candidate_proposal_budget` is a maximum rather than a required candidate
count. The checked-in Case 3 dry-run setting currently selects `neurosymbolic`;
`pure_llm` remains available as the controlled baseline.

Candidate `i` dominates candidate `j` when
`O(q'_i) ⊆ O(q'_j)` and `Gamma^R_{S/G}(q'_i) ⊇ Gamma^R_{S/G}(q'_j)`, with at least one strict
relation. One unique nondominated candidate is committed. No progressing
candidate returns `need_revision`; multiple incomparable nondominated
candidates, including equivalent successor proposals, return
`selection_ambiguous` and append nothing. The following LLM request receives the
exact obligation and enabled-event evidence and asks for one representative per
equivalent successor class. Three unresolved revisions under unchanged plant,
RA-descriptor, and supervisor fingerprints terminate as `selection_unresolved`.
The active ambiguity constraint remains in effect if that representative is
rejected by PA, RA, or CCA; the following request receives both the current
finding and the compact one-representative constraint. It clears after a
transition commit or an authoritative fingerprint change.
Candidate position,
`event_name`, `rationale`, resource name, token ordering, and randomness are not
tie breakers. PA, RA, and CCA do not synthesize a fallback event.

CCA returns `safety_dfa_states_before` and `safety_dfa_states_after` for every
candidate. The first outline transition starts from the live CCA DFA state;
later unexecuted transitions start from the selected projected DFA state. This
projection uses a temporary monitor and does not mutate the live CCA monitor.
Rule and live-state fingerprints fail closed when stale.

Outline validation is snapshot-specific. It establishes symbolic transition
feasibility, resource-reported physical feasibility, and rule-level safety for
the supplied state projection; it does not guarantee that a later primitive,
MoveIt plan, collision check, trajectory, Gazebo run, or hardware execution will
succeed.
The `neurosymbolic` method is not BFS, a global recovery-path search, or a formal
nonblocking supervisor over a complete event alphabet. Incomparable successors
remain explicit unless a separate declared objective is introduced.

## Verification Boundary

- Python-only changes usually need `poetry check`,
  `poetry run python -m compileall -q cais_spade_llm ros2`, and a targeted import
  or CLI smoke check.
- Entrypoint changes need `poetry run python -m cais_spade_llm.ui_main --help`.
- ROS2 launch/script/RViz changes need `make bootstrap-gazebo` before installed
  workspace checks.
- Use `test/` for focused or temporary tests when a feature needs them.
