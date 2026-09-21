# Journal Validation and Selector Handoff

## Purpose

This note records the journal-readiness review of transition feasibility,
physical feasibility, safety validation, and model-based selection. It is a
handoff for implementing and evaluating the remaining work on another machine.

The current architecture is suitable for a paper whose central claim is safe
admissibility of LLM-authored recovery transitions. It should not be presented
as global recovery-path search, optimal supervisory control, complete physical
executability, or universal safety.

This note does not record any runtime change. The only required selector code
change identified by the review is the strict-expansion correction described
below.

## KMR delivery handoff (2026-09-21)

The nominal delivery increment registers KMR with participating Storage/M1
contexts and routes the explicit Storage-to-M1 order through Start System and
the existing PA/RA/CCA task protocol. It changes no selector ranking or recovery
admission rules. Compact machine openings and fixed-joint grasp attachment are
simulation assumptions. Read the dated delivery verification in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) before making execution claims.

Active `lg_slippage` settings and resource bindings are removed; retain the
historical evidence below without presenting it as an available NIST scenario.
[Adaptive matching](ADAPTIVE_REQUIREMENT_CAPABILITY_MATCHING.md) remains planned.
Failure injection and recovery execution remain future increments. The
strict-expansion correction and its tests below remain outstanding.

The delivery acceptance recorded three acknowledged Gazebo tasks in
`20260921T194635_5b778116`, with M1 loaded, KMR empty, and no machining or
assembly completion. Its recorded acknowledgements reproduce the final custody.
The separate [active Stop check](../../cais_spade_llm/monitor/recovery_gazebo_runs/20260921T200209_5d5b706e/control_verification.json)
observed the executing arm goal terminate after Stop System, with zero task
commits, Storage inventory retained, KMR empty, and Gazebo retained. See the
linked implementation plan for current test results and reach-check scope.
This adds no selector or recovery-success evidence.

## Historical UI/settings handoff (2026-09-21)

The UI/settings phase is implemented separately from the selector correction.
**projects → recovery-framework → setup** now saves the selected Product Order,
permitted resources, Safety, execution mode, existing recovery settings, and
one optional failure scenario in `recovery_framework_setup.json`. Products,
Resources, and Safety keep their definition editors/views; **run** displays the
saved selections and retains **Start System**/**Stop System**.

Part slippage binds exact NIST components to permitted manipulators and their
capability tasks. It supports `before_execute`, `after_execute_before_commit`,
once-per-run occurrence, a configured drop pose, and an optional second holder.
The examples `ur5e-3` / `KET4_Square_4mm` and `ur5e-4` / `gear_large` do not
assert shared eligibility or prescribe recovery. The existing `lg_slippage` /
`LG` file is unchanged. All four documented scenarios can be saved as
**execution not integrated**. Start System rejects unsupported selected failures
and resource restrictions instead of ignoring them.

These settings establish no failure observations, task completion, PA/RA/CCA
approval, physical feasibility, or recovery success. Historical recovery artifacts
are not rewritten when setup changes. Missing run configuration remains `not recorded`.

The following increments are **nominal Start System integration**, **failure
injection**, then **recovery behavior**. The nominal increment must record the
actual setup with each run and establish acknowledged KMR Storage-to-M1 handling
before treating that case as Gazebo execution evidence. The strict-expansion
correction and its regression tests below remain outstanding and separate from
these UI changes. Preserve the existing historical verification in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Validation Authority at a Glance

| Stage | Authority | Actual responsibility |
| --- | --- | --- |
| Transition feasibility | PA | Checks exact declared start-state agreement, `part_traceability`, and state freshness. |
| Recovery admission | PA | Rejects `no_state_change` and `label_only_state_change`. |
| Physical feasibility | Responsible RA | Performs a resource-specific outline-level pre-check using a fresh or explicitly projected snapshot. |
| Safety | CCA | Evaluates projected event/state atomic propositions against the loaded safety DFAs. |
| Model-based selection | PA | Compares only PA/RA/CCA-valid projected successors by obligations, recovery-enabled events, and nominal-reentry events. |

## Transition Feasibility

### Current behavior

For a candidate transition from projected state `q_k`:

1. Every field declared in `expected_start_state` must exactly equal the
   corresponding PA-projected resource or part field.
2. A part-affecting candidate must preserve the Product-owned
   `part_traceability` invariant.
3. Acquisition and release require an explicit, non-null end
   `part_location`.
4. A held part must use the responsible RA's exact declared carried-part
   location.
5. Product and resource holder facts must agree, and two resources cannot hold
   the same part.
6. `part_location` cannot change without an explicit carrier/control relation.
7. A fresh RA snapshot and the PA state fingerprint prevent stale validation
   results from being committed.

An accepted LLM-authored `event_name`, `resource_state`, or `part_state` is an
exact symbol. Its wording does not supply motion, custody, safety, or progress
semantics.

### Journal-valid claim

Describe this stage as PA-owned symbolic transition admissibility for a
recovery-session successor. Do not equate it with physical feasibility, safety,
runtime authorization, or guaranteed membership in a complete pre-existing RA
event alphabet.

### Limitations and disclosure

- Only declared start-state fields are compared. Some resource-only fields are
  optional, so this is an exact partial valuation rather than full-state
  equality.
- Constraint-derived feedback can reveal an exact valid token, such as the
  responsible RA's carried-part location. This is deliberate validator
  guidance, not independent LLM discovery of that answer.
- Validation returns the first applicable finding, so revision feedback is
  ordered rather than exhaustive.
- Novel state labels remain recovery-session symbols and do not modify the RA's
  private descriptor.

No transition-feasibility runtime change is required for the safe-admissibility
claim. The paper must retain these boundaries.

## Physical Feasibility

### Current behavior

The PA sends the grounded candidate to the exact RA named by `resource_jid`.
The RA refreshes `get_recovery_snapshot()`, validates the candidate, and returns
snapshot and descriptor fingerprints. A base `ResourceAgent` without a
resource-specific implementation rejects with
`resource_validation_unavailable`.

`RobotAgent` currently checks:

- named-pose capability presence and exact availability;
- explicit resource unavailability;
- supported or concretely grounded resource targets;
- holder and gripper occupancy conflicts;
- availability of a grounded acquisition source;
- whether a part-affecting operation controls the required part; and
- whether a supplied Cartesian pose lies inside configured axis-aligned
  workspace bounds.

`PrintingAgent` currently rejects empty/out material and error/fault bed states.

### Journal-valid claim

Call this an RA-reported, outline-level physical feasibility pre-check. It does
not run inverse kinematics, collision detection, MoveIt planning, trajectory
validation, Gazebo execution, or hardware execution.

### Concerns to disclose or investigate

- Robot and printer validators have materially different coverage. Report the
  implemented checks per resource instead of aggregating them as equivalent.
- Robot workspace validation is an axis-aligned bounds check, not complete
  reachability. A candidate with no pose-dependent requirement can pass without
  a geometric check.
- Missing pose axes are not rejected by the bounds helper; only supplied axes
  are checked.
- `RobotAgent` gives the literal `resource_state` value `idle` a special abstract
  recovery allowance. This is an answer-specific semantic exception and should
  either be removed in favor of `supported_recovery_states` or explicitly
  declared as a RobotAgent-owned reserved state in the paper.
- Later unexecuted transitions use an intentional hybrid snapshot: projected
  dynamic state over fresh live capability evidence.

Removing the `idle` exception is recommended for consistency with the paper's
claim that state-label wording is not interpreted. It is separate from the
required selector correction and should be handled as an independently scoped
change.

## Safety Validation

### Current behavior

CCA receives the candidate, pre-transition state, projected successor, live
running atomic propositions, loaded safety rules, and current or projected DFA
states. It then:

1. Maps the candidate event and current/projected state to rule atomic
   propositions.
2. Creates a temporary monitor without mutating the live CCA monitor.
3. Evaluates each loaded DFA from its current projected state.
4. Rejects a candidate that enters a recognized violation state with
   `safety_rule_violation`.
5. Returns before/after DFA states and CCA-admissible nominal-reentry events.
6. Uses rule and live-state fingerprints to reject stale validation evidence.

### Journal-valid claim

Use the phrase "safe with respect to the loaded and successfully mapped safety
rules." Do not claim that `is_safe=True` proves absence of every manufacturing
hazard.

### Concerns to disclose or investigate

- No active rules produces `is_safe=True` by vacuous acceptance.
- A candidate that maps to no candidate or predicted-state atomic propositions
  is allowed by the monitor.
- A rule without a usable `dfa_dot` and recovery AP mapping is excluded from
  outline validation.
- Selector coverage is limited to the implemented recovery event/state selector
  modes; an unsupported selector can fail to activate an intended AP.
- A missing DFA transition stutters in the current state.
- Overlapping DFA guards use the first matching transition, so malformed or
  nondeterministic DFAs can be order-sensitive.
- Violation-state discovery assumes the expected DOT encoding, including the
  recognized `true` self-loop pattern.

Before journal evaluation, report rule/AP mapping coverage and distinguish
"no rules configured" from "rules were expected but could not be mapped."
DFA determinism, violation-state recognition, and AP coverage checks are strong
research-hardening improvements but do not need to be presented as part of the
current selector correction.

## Model-Based Selector

### Current behavior

The checked-in `neurosymbolic` configuration uses a one-action horizon. The LLM
proposes one to `candidate_proposal_budget` candidates and is prohibited from
selecting or ranking them.

The selector first excludes every candidate that fails PA, RA, or CCA
validation. For each valid projected successor `q'_i`, it computes:

- `O(q'_i)`: remaining exact recovery, continuation, reentry, and CCA safety
  obligations;
- `Gamma^R(q'_i)`: backward-relevant controllable RA-declared recovery event
  instances whose exact guards hold and which pass responsible-RA physical and
  CCA safety filtering; and
- `Gamma^N(q'_i)`: exact pending nominal-reentry events whose task guards and
  CCA projection are admissible.

The selector then prefers:

1. candidates with the fewest remaining obligations;
2. candidates whose `Gamma^R` set strictly contains another candidate's set;
3. candidates whose `Gamma^N` set strictly contains another candidate's set;
   and
4. the smallest exact-effect `candidate_id` when surviving candidates are
   equivalent or incomparable.

The final fingerprint rule is deterministic but arbitrary. It explicitly makes
no operational-superiority claim. Candidate position, `event_name`, rationale,
distance, duration, energy, and execution effort are not selection criteria.

"Model-based" means that private RA models and CCA projections are used to
compare consequences and future enabledness. It does not require the
LLM-authored candidate itself to be an existing RA-declared event.

### Required strict-expansion correction

The paper states that an unchanged obligation set progresses through recovery
enabledness only when:

```text
Gamma^R(q'_i) is a strict superset of Gamma^R(q)
```

The implementation currently tests only whether at least one new event exists:

```python
bool(enabled_events_after - current_enabled_events)
```

That permits an event-set swap:

```text
before = {event_A}
after  = {event_B}
```

even though `after` is not a strict expansion of `before`.

Update the recovery progression condition to require:

```python
enabled_events_after > current_enabled_events
```

The nominal-reentry condition has the same mismatch. When obligations and the
recovery-enabled set are unchanged, require:

```python
nominal_reentry_events_after > current_nominal_reentry_events
```

Do not change the later candidate-to-candidate strict-superset dominance or the
stable exact-effect representative rule.

### Required regression tests

Add focused cases for both `Gamma^R` and `Gamma^N`:

| Before | After | Expected progress |
| --- | --- | --- |
| `{event_A}` | `{event_A, event_B}` | Yes: genuine strict expansion |
| `{event_A}` | `{event_B}` | No: event swap |
| `{event_A, event_B}` | `{event_A}` | No: event loss |
| `{event_A}` | `{event_A}` | No: unchanged set |

Retain and rerun the existing tests for obligation clearing, name/order
invariance, RA-declared guard enabling, nominal-reentry expansion, incomparable
candidates, and equivalent-successor representative selection.

### Selector limitations for the paper

- The selector is one-step and can reject a necessary temporary detour that
  does not immediately reduce obligations or expand modeled enabledness.
- Every obligation is currently counted equally; severity and priority are not
  weighted.
- Results depend on the completeness and granularity of the RA descriptors and
  safety AP mappings.
- Only candidates generated by the LLM can be compared.
- Physical and safety evidence inherits the coverage limitations of the RA and
  CCA validators.
- No cost, throughput, distance, time, energy, or resource-load objective is
  implemented.

These limitations are compatible with a paper claiming one-step logical and
symbolic supervisory selection. They are incompatible with claims of an
optimal, unbiased, globally nonblocking, or shortest-path recovery policy.

## Required Work Before Submission

1. Correct the two strict-expansion progression predicates.
2. Add the recovery and nominal event-set regression tests.
3. Update the manuscript so physical feasibility and safety are described at
   their actual authority and coverage boundaries.
4. Add threats to validity covering candidate-generation dependence, one-step
   selection, model completeness, validator coverage, equal obligation
   weighting, and the arbitrary representative.
5. Report how often candidates are rejected at each validation stage, how often
   the stable representative is used, and how often no progressing candidate is
   found.
6. For a `pure_llm` comparison, use matched candidate pools where possible or
   disclose differences in candidate count and prompt instructions.

## Optional Future Improvements

- Weighted recovery obligations.
- Cost-, time-, energy-, or throughput-aware selection.
- Multi-step lookahead or global recovery-path search.
- Explicit unresolved-choice handling instead of immediate fingerprint
  selection.
- Stronger robot reachability, IK, collision, and trajectory pre-validation.
- Uniform physical-validation contracts and coverage reporting across RAs.
- Safety-rule completeness checks and fail-closed handling for expected but
  unusable rule mappings.

## Implementation Paths

- Selector implementation:
  `cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_outline_generation.py`
- Existing selector and validation regression coverage:
  `test/test_case3_recovery_dryrun.py`. Extend this coverage for the outstanding
  correction; `test/test_neurosymbolic_recovery_selection.py` is not present.
- UI/settings regression tests:
  `test/test_recovery_setup.py` and `test/test_project_pages.py`
- RA physical validators:
  `cais_spade_llm/agents/resource_agent/resource_agent.py`,
  `cais_spade_llm/agents/resource_agent/robot_agent.py`, and
  `cais_spade_llm/agents/resource_agent/printing_agent.py`
- CCA outline safety projection:
  `cais_spade_llm/agents/central_controller/outline_macro_safety.py`
- Shared DFA behavior:
  `cais_spade_llm/agents/central_controller/base_safety_checker.py`
- Journal working notes:
  `writing/Journal Paper 2 (recovery framework)/README.md`
- Architecture boundary:
  `docs/architecture.md`

## School-Laptop Workflow

The handoff note must be committed and pushed, or transferred through another
sync mechanism, before it will appear on the school laptop. After synchronizing
the repository:

```bash
cd /path/to/cais-spade-llm
git status --short
git switch -c fix/neurosymbolic-strict-expansion
```

Implement the selector correction and focused tests without mixing in the
optional physical or safety hardening work.

## Verification Commands for the Later Code Change

Run the focused selector suite:

```bash
poetry run pytest -q test/test_case3_recovery_dryrun.py
```

Run the project-required Python checks:

```bash
poetry check
poetry run python -m compileall -q cais_spade_llm ros2
```

Inspect the final patch:

```bash
git diff --check
git status --short
git diff -- \
  cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_outline_generation.py \
  test/test_case3_recovery_dryrun.py \
  "writing/Journal Paper 2 (recovery framework)/JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md"
```

Do not commit `.env` files, credentials, generated recovery artifacts, or
unrelated local changes.
