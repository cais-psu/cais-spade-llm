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
| Transition feasibility | PA | Checks exact `expected_start_state` agreement and validates the proposed successor against the Product-owned `part_traceability` invariant, including the responsible RA's exact carried-part location. |
| Recovery admission | PA | Rejects `no_state_change` and `label_only_state_change`. This concrete-effect policy is separate from standard DES transition feasibility. |
| Physical feasibility | Responsible live RA | Refreshes `get_recovery_snapshot()` and applies that RA's `check_recovery_physical_feasibility()`. Missing validation capability fails closed. |
| Local plant model | Responsible live RA | Privately supplies its task-level extended finite automaton with exact state variables, finite domains, current valuation, event guards and updates, controllable/observable classifications, marked conditions, and descriptor fingerprint. The descriptor is not included in the LLM prompt. |
| Candidate safety | Live CCA | Applies `validate_outline_macro_recovery_safety()` with the CCA's loaded safety rules and current monitor state. |
| Model-based selection | PA | In `neurosymbolic`, composes RA-declared local models with product state and compares only PA/RA/CCA-valid projected states using open recovery obligations, recovery-relevant enabled events, and exact nominal-reentry enabled events. |
| Candidate commit | PA | Rechecks the PA state fingerprint. `pure_llm` commits only the valid LLM-selected candidate; `neurosymbolic` commits the lexicographically preferred symbolic successor, using a stable exact-effect identifier only when the symbolic evidence is tied. It never substitutes an invalid candidate. |
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
the RA physical validator. `gripper_state` remains binary RobotAgent-private
runtime and physical-validation evidence. It is not a recovery DES variable,
is not accepted in an LLM candidate, and is not stored in the accepted symbolic
transition trace. For projected, unexecuted outline transitions, RobotAgent may
derive private `gripper_state` evidence from explicit `held_part`.
`PrintingAgent` expresses `pause_job`, `resume_job`, and `cancel_job` through
`resource_state` only. Its runtime snapshot and primitive execution still keep
`job_state` and `active_job`, but those fields are not part of its private DES
descriptor. Other ResourceAgents declare only the exact finite variables their
own enabledness, marked conditions, safety checks, or physical validator need.
They must implement the descriptor or provide an explicit initialization
descriptor; absence fails closed.

The LLM request contains only compact live symbolic state, observations needed
for grounding, resource capabilities, recovery goals, safety rules, and exact
current validation findings. Nominal `origin_location` remains private to
nominal-reentry evaluation and debug evidence; it is not proposal guidance in
the outline prompt. RA event alphabets, guards, updates, marked conditions, and
descriptor fingerprints also remain private in session state and debug result
artifacts. Available named poses, reachability, current state, goals, and active
safety rules are instance facts that constrain validity, not scripted recovery
answers.

The structured response schema contains the union of RA-declared state-field
names needed to parse candidates for different resources in one response. That
union is not authorization: PA checks each candidate against only the exact
descriptor owned by its `resource_jid`, including the declared `resource` or
`part` scope. `gripper_state` and `job_state` are not accepted candidate fields
for the current RobotAgent and PrintingAgent descriptors. Values of
resource-specific fields must belong to the declaring RA's exact finite domain. Adding another
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

### PA transition feasibility and recovery admission

Let `q_k` be the PA's authoritative projected symbolic state at outline turn
`k`. An LLM candidate declares `expected_start_state` and
`expected_end_state`. The candidate is enabled only when every exact field it
declares in `expected_start_state` equals the corresponding field in `q_k`.
The LLM-authored `expected_end_state` is then copied as the proposed successor;
PA does not fill a missing `part_location` from `held_part`.

The successor is legal only when it satisfies the Product-owned
`part_traceability` invariant:

- every candidate with `part_name` supplies `held_part`, `part_state`, and
  `part_location` in both state objects;
- an unknown current part location may be represented by a null start
  `part_location`;
- acquisition or release must produce a non-null, supplied end
  `part_location`;
- whenever the responsible resource holds the part, end `part_location` equals
  that RA's exact declared carried-part location;
- the projected part holder agrees with explicit `held_part`; and
- two projected resources cannot hold the same part.

Failures of exact start agreement or `part_traceability` are reported as
`transition_feasibility`. Schema, token, field-scope, and RA-domain failures are
reported as `syntax_and_grounding_validation`. A successor that produces no
state change, or changes only an uninterpreted state label without a concrete
effect or exact recovery-condition satisfaction, is rejected separately as
`recovery_admission`.

An accepted LLM-authored event therefore adds one validated recovery-session
transition `(q_k, event_name, q_{k+1})` to the nominal transition relation used
by the outline. New `resource_state` and `part_state` values remain
uninterpreted exact labels; their meaning is not inferred from their text.

Custody changes remain atomic symbolic transitions. Acquisition must explicitly
change `held_part` from null to `part_name` and set `part_location` to the
responsible resource's declared carried-part location. Placement or release is
a later transition that changes `held_part` from `part_name` to null and sets
`part_location` to the concrete destination. A direct null-to-null holder
transition that moves `part_location` is rejected as
`part_relocation_without_carrier`; primitive generation cannot use that row to
hide acquisition, transport, and release inside one unvalidated macro-event.

Corrective prompt evidence is validation-derived and resource-scoped. After
`part_relocation_without_carrier`, PA calls the responsible resource's existing
`ResourceProfile.carried_entity_location_builder` and includes the resulting
exact token only when that same RA descriptor permits it in the
`part_location` domain. The same scoped correction follows
`held_part_location_mismatch` or `missing_acquisition_location`; the token is
not displayed globally. Resources without a declared carried-part location
receive no custody token and fail closed when custody is required. Applicable carrier and workspace findings survive
unrelated turns while their exact state and physical evidence remain unchanged;
they clear immediately when custody, observation, or workspace evidence changes.
Rejected rationale, event names, candidate ordering, and accepted transition
stacks are not carried into later LLM requests.

Recovery completion is separate from transition feasibility. A part recovery
goal at `goal_location` remains open while the part has a projected holder or
any projected resource reports `held_part == part_name`. Exact supplied
part-state conditions must also hold. State labels such as `placed` or
`assembled` never bypass location and custody consistency, while a newly
authored state label is permitted when no exact final state label was supplied.

For composed plant state `q`, `O(q)` is the exact set of open PA recovery goals,
continuation/reentry obligations, and CCA safety-condition identifiers.
`Q_m^R` denotes recovery-compatible marked states. Recovery enabledness is the
intersection

\[
\Gamma^R_{S/G}(q)
= \Gamma^R_G(q) \cap \Gamma^R_{\mathrm{RA}}(q) \cap \Gamma^R_S(q).
\]

`Gamma^R_G(q)` contains the backward-relevant RA-declared controllable event
instances whose exact symbolic guards hold. `Gamma^R_RA(q)` contains the same
bound instances accepted by the responsible RA's physical validator.
`Gamma^R_S(q)` contains the instances admitted by the CCA safety projection.
The RA check uses exact affected resources, parts, locations, observations, and
the current projected state. RA and CCA enabledness replies are correlated and
fingerprinted; unavailable, stale, malformed, or mismatched evidence fails
closed. This keeps continuous workspace evidence out of the finite DES state
without allowing a symbolically enabled but physically unreachable event to
count as recovery progress.

`Gamma^N_{S/G}(q)` is the CCA-admissible set of exact nominal-reentry events
whose verified task guards hold for affected resources and parts on their
continuation paths. It is derived from verified plan tasks, current task status,
RA task guards, exact holder/location state, and CCA projection. Unrelated
unfinished nominal work is excluded and neither event names nor location text
are interpreted.

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

A validated LLM candidate progresses in exactly one of three ways:

1. it reduces `O(q)` without introducing another obligation;
2. it preserves `O(q)` and strictly expands `Gamma^R_{S/G}(q)`; or
3. it preserves both `O(q)` and `Gamma^R_{S/G}(q)` and strictly expands
   `Gamma^N_{S/G}(q)`.

Selection is lexicographic in that same order. If candidates remain symbolically
identical or incomparable, the smallest existing exact-effect `candidate_id` is
committed as a reproducible representative. That last rule is arbitrary and
does not claim operational superiority. Candidate position, `event_name`,
`rationale`, resource type, location wording, distance, duration, energy,
execution effort, and randomness are not selection criteria. No valid
progressing candidate returns `need_revision`; three unchanged-fingerprint
turns with no valid progressing candidate terminate as `selection_unresolved`.
PA, RA, and CCA do not synthesize a fallback event.

After a transition is committed, later LLM requests contain the authoritative
projected state and only rejected PA/RA/CCA findings that remain applicable to
that state and the same observation and capability evidence. A finding such as
`workspace_unreachable` therefore survives an unrelated successful transition,
but is removed when its affected observation, holder/location state,
capability fingerprint, or physical result changes. Accepted event names,
rationales, selection comparisons, and transition traces remain debug-only and
are not prompt history.

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
The `neurosymbolic` method is logical/symbolic supervisory selection, not BFS,
a global recovery-path search, optimal supervisory control, or a formal
nonblocking supervisor over a complete event alphabet. Cost-based optimal
supervisory control remains a future comparison mode. This separation follows
DES work that treats logical requirements separately from optional cost or
throughput objectives; representative comparisons include
[Automatica](https://www.sciencedirect.com/science/article/pii/S0005109824001274),
[IFAC optimal DES control](https://www.sciencedirect.com/science/article/pii/S1474667017512197),
and [IFAC throughput control](https://www.sciencedirect.com/science/article/pii/S1474667015374036).

## Verification Boundary

- Python-only changes usually need `poetry check`,
  `poetry run python -m compileall -q cais_spade_llm ros2`, and a targeted import
  or CLI smoke check.
- Entrypoint changes need `poetry run python -m cais_spade_llm.ui_main --help`.
- ROS2 launch/script/RViz changes need `make bootstrap-gazebo` before installed
  workspace checks.
- Use `test/` for focused or temporary tests when a feature needs them.
