# Recovery framework roadmap, including UI dry runs and trustworthy prompt inspection

**Status: paper-aligned RA derivation, CCA outline checks, and diagnostic evidence
implemented in the working tree on 2026-09-29; remaining milestones are identified
below.** Source baseline: `c485e51`. This is the sole current roadmap for Journal
Paper 2. Software checks do not establish live recovery or hardware safety.

## Summary

Implement and evaluate formal validation, deterministic safety, local parallel
composition, four recovery scenarios, and UI dry-run testing. Use Gazebo first;
hardware experiments remain later work.

A recorded prompt in the UI must be the complete application-controlled request
actually submitted to the model. Expected recovery answers remain outside model
inputs. Prompts are read-only in the UI. Dry runs support reviewed fixtures and
saved Gazebo failure snapshots.

Use the [formal reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md) for model
semantics and claim boundaries, [the document index](README.md) for supporting
references, and [implementation history](IMPLEMENTATION_HISTORY.md) for prior
results. The supplied manuscript excerpt is authoritative for feasibility and
outline safety; its exact text is preserved in the
[formal reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#supplied-manuscript-excerpt).
Selector policy, deterministic safety construction, and local composition are
separate implementation decisions. PA schema checks remain preparation, rather
than a new paper-level validation contribution.

## Current status and dependencies

| Area | Observed baseline | Remaining milestone |
| --- | --- | --- |
| Nominal Gazebo execution | Retained 11-part run with 195 matching transitions; machining uses simulated processing and gears start preprinted. | Establish a fresh baseline with active rules and CCA bypass disabled before recovery experiments. |
| Feasibility and DFA validation | Union of all successor-support primitives; witnesses for every distinct primitive; shared strict DFA transitions, prefix continuation checks, and detailed findings. | Extend physical witnesses and resource contracts to all four larger-setup scenarios; selector correction remains a separate work item. |
| Recovery safety generation | Some formula construction is deterministic, but an LLM still selects rule/event involvement. | 2: deterministic applicability, bindings, APs, formulas, and coverage. |
| CCA composition | Global plan FSA and existing active-window validation paths exist. | 3: direct local composition with joint completion checks. |
| Diagnostic UI | Reusable `scenario_runner`, frozen dependencies, explicit live/replay modes, immutable per-attempt capture, RA/CCA evidence, and input inspection. | Reviewed fixtures/snapshots and resource adapters for the four larger-setup failures; deterministic `safety` mode depends on milestone 2. |
| Physical failure scenarios | Setup can save the four failures; startup rejects selected failures as execution not integrated. | 5: injection, recovery execution, observation, and resumption. |
| Paper evaluation | Protocol specified here; no new trials collected by this change. | Experimental design after the relevant acceptance gates. |

Checked-in settings select `safety_none.txt`, `diagnostic_cca_bypass=true`,
`neurosymbolic`, `action_horizon=1`, `candidate_count="adaptive"`, and
`candidate_proposal_budget=5`. Historical bypass runs establish execution evidence,
not active-rule CCA safety. See [the retained run](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/README.md).

The RA → CCA → UI implementation slice is available for saved-context inspection.
Next, supply reviewed staging/grasp/transport contracts and failure snapshots for
the larger setup, including physical evidence that the current validators report
as `NEEDS_CONTEXT`. Milestone 2 establishes the rule contract for milestone 3.
Milestone 4's remaining scenario fixtures can proceed alongside 2–3; its current
`safety` stage still records the existing LLM-selected grounding method. Milestone
5 depends on the relevant capabilities and validated safety paths. Collect
confirmatory trials only after all gates relevant to their study pass.

Preserve `SystemBridge`, exact project symbols, current evidence, and the
PA/RA/CCA authority split. Positions, capabilities, and resource bindings remain
configuration-driven. Do not change the separate Spec2Primitives subsystem for
this roadmap.

## Milestone 1: Formal validation contract

Retain `G = G_P || (||_{r in R} G_r)` and the
[stage contract](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#formal-model-and-authority):

| Stage | Authority | Required behavior |
| --- | --- | --- |
| Symbolic feasibility | PA | Exact declared-start agreement, valid successor effects, `part_traceability`, and fresh state. |
| Primitive feasibility | Responsible RA | Establish primitive support and parameter/capability witnesses; distinguish `FEASIBLE`, `INFEASIBLE`, and `NEEDS_CONTEXT`. |
| Primitive composition | Responsible RA | Check sequence ordering, compatible parameters, and achievement of the complete intended successor. |
| Safety | CCA | Evaluate applicable LTLf DFAs using projected history, running events, persistent state, and pending obligations. |
| Execution and commit | RA, PA, CCA | Revalidate relevant evidence and commit only acknowledged, observed effects. |

### Implemented paper-aligned validation slice

PA proposes new recovery events from projected conditions and public capabilities,
without receiving the complete private RA transition model. RA derives support
from its own successor valuations; neither the proposed event nor its starting
condition needs to be a predefined task transition.

- Collect the union of primitives from **all** matching successor transitions.
  Deduplicate by primitive identity and retain every source transition, binding,
  and covered valuation field. Require a parameter/capability witness for every
  collected primitive. A known empty union is rejected; missing descriptors,
  observations, evaluators, or incompletely searched domains remain `NEEDS_CONTEXT`.
- Return the union and evidence to PA and composition. PA verifies the descriptor,
  exact proposal, sources, and primitive identities. Feasible support does not
  establish compatible ordering or complete product effects; those remain explicit
  composition obligations. Private primitive preconditions and conditional steps
  remain in the contract evidence.
- A `release_part` effect can clear `held_part`; it cannot establish staging by
  itself. Composition needs a bound target and motion to that same target. Motion
  away invalidates the earlier landing fact. Actual support/contact and final
  part location still require physical evidence and observed execution.
- CCA deterministically projects each successor label, including relevant running
  tasks and persistent state. Every applicable DFA must take one unambiguous
  transition and retain a path to acceptance; an unfinished prefix need not already
  be accepting. Empty steps remain in the trace. Missing rules, mappings, required
  fields, or malformed transitions cannot authorize admission.
- Rejected candidates leave the accepted prefix, projected state, and monitor
  state unchanged. Records retain each rule, requirement, label, transition,
  acceptance/reachability finding, and rejection reason. Outline checks do not
  certify the intermediates of an ungenerated primitive composition.

**Verified scope:** parameterized/novel successor matching, primitive-union
semantics, missing witnesses, airborne-release rejection, DFA continuation and
malformed transitions, and evidence inspection. The
[contract audit](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#resource-coverage-and-remaining-limitations)
identifies physical and resource-specific gaps. Existing runtime interfaces are
retained. PA schema checks remain implementation preparation.

### Separate selector correction (planned)

Correct strict-expansion selection: event swaps cannot count as progress.
Incorporate CCA admissibility within the documented enabled-event sets rather
than an additional undocumented progression rule. Preserve arbitrary but stable
exact-effect tie resolution and the one-action horizon; make no cost-optimality
claim. This correction is not part of the manuscript's feasibility algorithm.

**Acceptance:** strict expansion, event swaps, event loss, unchanged sets,
obligation changes, and nominal reentry have regression coverage.

## Milestone 2: Deterministic safety construction

Use `reviewed templates + exact resource bindings → AP mappings → LTLf → DFA`.
Remove LLM decisions from authoritative applicability, event selection,
grounding, and formula construction. Keep LLM recovery-action and primitive
proposals.

The reviewed rule catalog covers machine access/interlocks, exclusive M1/M2
`workholding` access through front access and `side_access`, staging/buffer
capacity, Conveyor handoffs, KMR docking/arm/base conditions, assembly-region
exclusion, persistent occupancy, and recovery/resumption obligations. Bind from
resource declarations and observations, never from the expected recovery answer.

Preserve event/state APs, predicted successor facts, empty trace steps, and
finite-trace completion semantics. Nonaccepting prefixes with a valid accepting
continuation remain pending obligations. Missing required evidence or unsupported
rules block admission with an explicit reason. Version the templates and record
AP/rule coverage with each bundle.

**Acceptance:** identical inputs produce identical safety artifacts, all required
rules have grounded coverage, and authoritative safety construction makes no
LLM calls. Treat construction reproducibility separately from the completeness
of the reviewed hazard model.

## Milestone 3: Local parallel composition

Keep CCA as the decision authority. Build local products directly from factored
models; do not build a global FSA and then slice it. Determine scope from shared
events, resource participation, guard/update dependencies, product custody,
task prerequisites, and safety APs. Expand scope when dependencies cross its
boundary; omitted resource behavior cannot be assumed safe or fixed without
validated evidence.

Preserve shared-event synchronization, atomic custody, running tasks, persistent
occupancy, and pending obligations. Check immediate safety and joint safe
completion. Separate successful rule checks do not establish a compatible joint
continuation. Preserve the event clock and empty/end-step semantics; dropping
an event is not automatically equivalent to stuttering.

Cache by model, rule, relevant-state, and scope fingerprints and invalidate on
change. Start with the existing 200,000-state limit and a 10-second validation
deadline. Exhaustion is inconclusive and holds affected dispatch, not proof of
safety or infeasibility. Strong coupling may still require a large product.
Completion and failure observations are consumed; only controllable starts are
gated. Recheck the snapshot and reservations before committing concurrent starts.

**Interfaces:** retain `SystemBridge` and existing PA/RA/CCA protocol identities.
Add an internal factored-plan backend and result evidence for checked scope,
dependencies, fingerprints, explored states, timing, and inconclusive reasons.
Keep global and existing active-window backends as comparisons. Do not equate a
local accepting continuation with a general global nonblocking guarantee.

**Acceptance:** small-model checks agree with global validation for safety and
completion, coupled conflicts are detected, and unrelated resource growth does
not unnecessarily enlarge products. Test open obligations across scope changes,
concurrent starts, shared handoffs, and stale cached results.

## Milestone 4: UI dry runs and exact prompt evidence

Use the existing **recovery → Test a failure scenario** panel. Extend its
[diagnostic service](../../cais_spade_llm/recovery_framework/diagnostics.py)
now launches the reusable
[scenario runner](../../cais_spade_llm/recovery_framework/scenario_runner.py).
The former Case 3 test file retains its regression cases and imports that runner.
The page reads the authoritative roadmap/formal sections; README compatibility
headings remain short links.

The following is the full milestone contract. Its request recording, input
freezing, cancellation, explicit replay, and validation-evidence foundations are
implemented. Currently selectable saved fixtures are the legacy `lg_slippage`
and `move_home_failure` contexts. They do not constitute reviewed fixtures for
all four larger-setup scenarios; that acceptance gate remains open.

### Available inspection now

Open **recovery-framework → recovery → Test a failure scenario**. The existing
`lg_slippage` scenario pairs with `test/fixtures/case3_recovery/runtime_context.json`;
select `outline` first. Inspect input provenance and dependencies before running.
Live calls are the default. Missing physical readiness or saved CCA history is an
unresolved validation result, not an automatic fixture approval. Saved active-rule
contexts need `recovery_safety_context.safety_dfa_states` and an explicit
`running_aps` list. Explicit replay requires a JSON array of response objects and
is labeled as mocked. The larger ur5e-3/ur5e-4 scenario still needs its own matching
snapshot and contracts.

After a run, select its record to inspect exact requests/responses and each
candidate's detailed RA/CCA evidence. Requests from cancelled or failed attempts
remain available even when there is no completed stage result.

### Workflow

1. Select one of the four failure scenarios and a variant.
2. Select a reviewed fixture or saved Gazebo failure snapshot.
3. Inspect the starting state, observations, goals, capabilities, safety
   configuration, model/settings, and provenance.
4. Select `outline`, `primitive`, `safety`, or `full`; default to `outline`.
5. Run or cancel the diagnostic.
6. Inspect every turn's request, response, validation findings, selected
   transition, and projected successor.

Use live model calls by default. Fixture-response replay requires explicit
selection and visible labeling. API errors never silently substitute scripted
answers. Deterministic safety mode shows construction inputs and generated
artifacts without presenting them as an LLM conversation.

### Shared behavior and isolation

Dry runs and runtime use the same prompt builders, schemas, selection logic, and
validators. Supply observations through snapshot-backed adapters and label
observed, synthetic, and mocked evidence. Missing physical evidence remains
`NEEDS_CONTEXT`; fixtures cannot silently make every action feasible.

Dry runs do not dispatch robot actions, change live plant state, or complete
production tasks. Snapshot tools expose only declared non-executing context.
Freeze inputs and all referenced dependencies when the run is created, and
actually read those copies throughout execution. A copied manifest that still
reads mutable live files is insufficient.

Permit accepted-outline reuse only when inputs, prompt/schema, settings, and
validator fingerprints match. Preserve one active diagnostic job, cancellation,
partial results, and progress across page navigation. Viewing or refreshing a
record must not send a new model request.

### Exact request and response records

Capture every application-level attempt immediately at the provider-call
boundary, including all message roles and exact contents, tool definitions,
response schemas, model settings, and explicit generation parameters. Capture
all tool rounds, assistant tool calls, tool results, and application retries.
Record raw returned responses before parsing, plus errors and available response
metadata. Exclude sensitive transport credentials.

Persist immutable records keyed by run, stage, turn, and attempt. Record a
prepared request separately from invocation and response outcomes. The UI reads
these artifacts directly and presents raw records separately from parsed
outputs, selected transitions, and validation summaries. Provide the complete
record for download; visual summaries must not replace it.

A prepared preview says **not sent**. Missing capture says **not captured**.
Reconstructed prompts, filename guesses, and `latest` files cannot stand in for
an exact historical request. Existing legacy records retain their provenance
and capture limitations. Capture scope and current limitations are in the
[formal reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#prompt-evidence-and-information-boundaries).

### Answer-leakage controls

Keep expected recovery sequences, golden responses, and scoring criteria in
evaluator-only data. Prompt builders and context tools must not expose that data.
Explicit response replay is loaded separately from operational inputs and is
labeled as scripted; replay responses are never an outline-prompt ingredient.
Outline inputs contain operational facts, legitimate goals, capabilities,
safety constraints, and applicable validation feedback, not a prescribed route
or assignment from the evaluator's answer.

Primitive generation may receive the current run's validated outline and the
responsible resource's primitive catalog, not a golden primitive sequence.
Keep private RA models and selector internals outside model-facing fields unless
explicitly part of the documented method. Exact constraint-derived feedback is
legitimate guidance and must remain visible in the record.

Record operator guidance explicitly and distinguish assisted runs from
unassisted evaluation. Audit prompts and tool results together. Use the term
**controlled information exposure and answer-leakage prevention**; do not claim
universal absence of bias. Preserve formal tokens when varying input order.

**Acceptance:** all four scenarios can be tested before live recovery exists;
transport-spy tests prove displayed payloads equal actual call arguments;
evaluator-only canaries never reach messages or tool results; no diagnostic
moves equipment. Cover malformed responses, retries, cancellation, missing
capture, frozen-dependency changes, and checkpoint mismatches.

## Milestone 5: Four physical failure scenarios

Use once-per-run, state-based injections reached through acknowledged execution.
Record the actual failure state and distinguish configured drop targets from
observations. These recoveries are evaluation hypotheses, not prompt material
or scripted runtime answers. Alternative validated sequences may succeed.

| Failure and checkpoint | Recovery behavior | Negative variants |
| --- | --- | --- |
| **Conveyor breakdown:** after `ur5e-1` acquires the completed part, before Conveyor release. | `ur5e-1` stages the completed part in the M1 staging tray. KMR collects it and transports it directly to Buffer For Machined parts for `ur5e-3` to retrieve. | Unavailable KMR, blocked route, occupied staging/buffer. |
| **Machining station handling robot breakdown (`ur5e-1`):** completed part remains in M1 before acquisition. | KMR docks at M1, unloads the completed part through `side_access`, and places it on the operational Conveyor for delivery to Buffer For Machined parts. | Blocked access, missing interlock evidence, unavailable loading space. |
| **Machining breakdown during part processing (M1):** at 50% of configured simulated processing duration. | `ur5e-1` or KMR extracts recoverable WIP from M1. KMR transports it to the M2 staging tray, and `ur5e-2` loads M2 to complete the remaining operation after any required configuration change. | Unsafe extraction, incompatible configuration, unavailable M2, unrecoverable WIP. |
| **Part slippage:** `ur5e-3` holds its part while `ur5e-4` holds another; inject and observe a drop into `ur5e-4`'s region. | `ur5e-4` stages its current part, retrieves the slipped part, completes `ur5e-3`'s interrupted task, and resumes its original sequence. | Unreachable part, unavailable staging, unsuitable part condition. |

Preserve the failed robot's actual pose, WIP identity/progress, custody, and both
interrupted-task obligations. Implement missing extraction, staging,
configuration-change, and route support through resource-owned capabilities.
Validate new KMR recovery routes without preloading them as hidden nominal
alternatives. Simulated machining progress does not establish material-removal
physics or hardware resumability.

**Acceptance:** observe recovery and nominal resumption, or a justified
rejection/hold without false completion or custody loss. A blocked reference
route is not proof that every alternative is infeasible. Preserve unsuccessful
attempts and incomplete observations.

## Predefined task-level DES audit

For the claim that a failure needs synthesis, separately establish **no recovery
path within the predefined task-level DES** from the actual failure valuation to
all remaining goals, including both interrupted tasks. An RA's private
successor/effect model is used to derive primitive support; it is not evidence
that a complete task-level recovery route already exists.

The diagnostic runner writes `task_des_audit.json` using a bounded exhaustive
search of a supplied explicit task model. Bind the model's exact failure valuation
to `failure_snapshot_fingerprint(runtime_context)` in
`predefined_task_des_audit.snapshot_fingerprint`. Include `model` (`states`,
`transitions`, `events`), `failure_valuation`, `goal_conditions`, and an explicit
`complete` model-construction assertion. Results are:

- `path_found`: report the legitimate modeled continuation without suppressing it.
- `no_path`: complete exploration of a model explicitly declared complete.
- `inconclusive`: absent/incomplete model, unmatched/stale failure snapshot, or
  exhausted 200,000-state/10-second search budget.

Current larger-setup audit (2026-09-29):

| Scenario | Result | Missing evidence |
| --- | --- | --- |
| Conveyor breakdown | `inconclusive` | Exact post-acquisition failure snapshot and complete task graph with staging/buffer capacity. |
| ur5e-1 breakdown | `inconclusive` | Exact in-M1 failure valuation and complete task graph including access/interlock facts. |
| M1 processing breakdown | `inconclusive` | 50% WIP snapshot and complete task graph with progress, configuration and resumption obligations. |
| ur5e-3 part slippage | `inconclusive` | Matching ur5e-3/ur5e-4 snapshot and complete task graph preserving both interrupted tasks; legacy LG/MCP fixtures are different conditions. |

No symbolic-unsolvability claim is established for these four scenarios yet.
Keep baseline audits outside model prompts and do not remove valid nominal
alternatives to obtain a desired result.

## Experimental design

Separate four studies so their effects can be assessed independently:

| Study | Comparison and measurements |
| --- | --- |
| Recovery effectiveness | No recovery, DES-only, `pure_llm`, `neurosymbolic`; recovery success, resumed work, validation-stage rejection, revisions, latency, primitive/execution failures, custody errors. |
| Composition scalability | Global, existing active-window, local; construction time, explored states, peak memory, decision latency, cache reuse, disagreements, timeouts. |
| Safety construction | Existing LLM-selected grounding versus deterministic grounding on identical recorded inputs; required-rule coverage, AP correctness, repeatability, false acceptance/rejection against reviewed cases, construction time. |
| Prompt integrity | Dry-run/runtime payload parity, every-round capture, evaluator-data leakage, and sensitivity to input ordering with exact symbols retained. |

### Recovery trials

Use four scenarios × one feasible and one deliberately blocked condition × four
methods. Run three pilot repetitions, followed by ten confirmatory repetitions:
**96 pilot trials and 320 recorded trials**. Pilots are not confirmatory data.

Use `KET4_Square_4mm` for initial machining/transport and add `gear_large` as
`ur5e-4`'s interrupted part for slippage. Initial blocked conditions are unavailable
KMR, blocked `side_access`, incompatible M2 configuration, and a slipped part
unreachable by permitted resources, respectively. Freeze exact configurations
and reference labels before collection; include other negative variants in
regression coverage first. Score validated alternative recoveries on their
observed outcome rather than requiring the reference sequence.

For controlled LLM comparisons, add exactly-three-proposals support in both
modes, keeping adaptive settings unchanged outside experiments. Use matched
models, settings, seeds where supported, and budgets. Compare selectors
separately on identical recorded candidate pools; disclose candidate-generation
and actual-count differences in end-to-end runs.

Use shared limits of 30 proposal turns and 30 minutes after injection. Budget
exhaustion is recorded separately from infeasibility. Freeze configurations
after pilot revisions and before collecting confirmatory trials. Establish a
fresh nominal Gazebo baseline with active rules and CCA bypass disabled first.

### Scaling, safety construction, and prompt integrity

Benchmark 2, 4, 8, 12, 24, and 48 resource factors using documented subsets and
synthetic extensions. Include sparse and tightly coupled dependencies, ten
repetitions per configuration, and identical 60-second/2-GiB benchmark limits.
Count global construction cost as well as checking cost. These offline limits
are separate from the 10-second runtime validation deadline.

Use identical recorded contexts/outlines for safety-construction comparisons.
Include valid, invalid, and missing-context cases, unmapped APs, stale evidence,
open obligations, and boundary conflicts. Repeat identical deterministic inputs
to establish artifact reproducibility. Prompt-integrity tests cover initial
requests, tools, retries, error paths, and fixture/Gazebo provenance.

All executed methods retain deterministic safety gating. No recovery retains
normal stop behavior; DES-only uses the modeled recovery available to it. Run
removed-validation treatments only through offline replay. Keep dry-run
validation, Gazebo execution, and hardware outcomes distinct.

### Reporting and evidence

Record baseline/code versions, scenario and variant, exact inputs and snapshot
origin, seeds, model and prompt/schema settings, safety bundle, composition
backend, budgets, per-attempt requests/responses, stage findings, selected
actions, controller acknowledgements, observed outcomes, and wall/simulation
time. Preserve failed, cancelled, and incomplete runs with their true status.

Randomize method order within matched conditions. Report denominators, 95%
uncertainty intervals, variability, and all failures/timeouts. Use binary
success intervals and resampled latency/difference intervals; distinguish
successful-run latency from time-limited runs. Exclude pilots from confirmatory
results and identify every exclusion rather than silently dropping runs.
No recovery-success or throughput claim follows from a validated projection alone.

## Verification and handoff

Documentation acceptance requires one roadmap, working index/reference links,
preserved history, and the exact README headings `Working Thesis`, `Experiment
Plan`, and `Neurosymbolic Selection Method` retained as compatibility links.
The UI now reads the authoritative sections directly.

Extend the nearest existing diagnostics, UI, feasibility, and recovery suites.
Tests must cover every scenario, request/tool round, malformed response,
cancellation, snapshot change, missing capture, checkpoint mismatch, and
absence of robot dispatch. Add transport-spy and evaluator-data-canary checks.
Test strict expansion, local/global agreement, coupled obligations, concurrent
starts, stale evidence, and deterministic rule coverage.

For Python changes run `poetry check` and
`poetry run python -m compileall -q cais_spade_llm ros2`. Run
`poetry run python -m cais_spade_llm.ui_main --help` for entrypoint changes.
Run `make bootstrap-gazebo` before installed-workspace checks for ROS2
launch/script/RViz changes. Always run focused changed-feature tests and
`git diff --check`.

The historical planning baseline passed 36 feasibility/DFA and 14 diagnostics/UI
checks. The implemented validation/inspection slice now passes **244 focused
tests**, plus the final artifact-label checks, Poetry, compilation, and CLI checks;
[verification details](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#verification-and-claim-limits)
are recorded in the formal reference.
These are software validation results; no live Gazebo or hardware recovery was
performed. Local composition, deterministic safety construction, four-scenario
execution, and confirmatory experiments remain separate acceptance gates.
