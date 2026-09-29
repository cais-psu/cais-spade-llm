# Journal formal validation and selection reference

**Implementation reference, 2026-09-29, working tree based on `c485e51`.**
The supplied manuscript excerpt is authoritative for feasibility and outline
safety. Its [supplied text](#supplied-manuscript-excerpt) is preserved below.
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) remains the sole roadmap for
acceptance gates, scenarios, UI dry runs, and experiments.

The excerpt ends at the heading for safety monitor construction; no unprovided
manuscript content is inferred. Selector policy, deterministic construction, and
local composition are separate decisions. PA schema checking is implementation
preparation rather than an additional paper contribution. Prior narratives and
superseded claims remain in [implementation history](IMPLEMENTATION_HISTORY.md).
Preserve exact identifiers, fields, predicates, events, resources, and states.

## Formal model and authority

Retain the plant notation:

`G = G_P || (||_{r in R} G_r)`

`G_P` is the product automaton; each RA owns its local extended finite automaton
`G_r`. Its descriptor declares exact variables, finite domains, valuation,
events, guards, updates, controllable/observable classifications, marked
conditions, and a descriptor fingerprint. `S/G` denotes CCA supervision of the
plant. This notation does not assert that the implementation has enumerated a
complete plant or synthesized an optimal nonblocking supervisor.

Task-level resource transitions, saved function programs, and controller
primitives are distinct representations. Continuous pose, workspace, and motion
evidence belongs to the responsible RA's physical checks rather than an
inferred meaning of a finite-state label. Product identity, custody, and
completed operations remain authoritative throughout recovery.

| Stage | Authority and input | Result and claim boundary |
| --- | --- | --- |
| Proposal generation | PA invokes the LLM with documented model-facing context. | A candidate is a proposal, not an authorized action. |
| Syntax and grounding | PA checks the candidate against the responsible RA's declared field scopes and exact bindings. | Shape, binding, and domain findings are `syntax_and_grounding_validation`. |
| Symbolic transition feasibility | PA checks the prepared start/end valuation and Product-owned `part_traceability`. The responsible RA also checks its local transition conditions. | Symbolic admissibility is separate from physical realizability. |
| Primitive feasibility | Responsible RA evaluates fresh/projected state, its private model, primitive support, parameters, and capability evidence. | `FEASIBLE`, `INFEASIBLE`, or `NEEDS_CONTEXT`; no guarantee of a complete executable sequence. |
| Primitive composition | Responsible RA validates the generated program, ordering, parameter compatibility, and complete intended effect. | Executable composition still requires dispatch-time checks and observations. |
| Safety projection | Live CCA checks applicable rules, APs, running tasks, and current/projected DFA states. | Safe with respect to the loaded, applicable, successfully mapped rules and supplied evidence. |
| Model-based selection | PA compares validated successors and their remaining obligations/enabled events. | One-step logical selection; no distance, time, energy, or throughput optimum. |
| Execution and commit | CCA gates controllable dispatch; responsible RA executes; PA/resource state commits require matching evidence. | Projected effects and controller acknowledgement alone do not establish unobserved work. |

`recovery_outline_physical_validate` addresses the exact responsible
`resource_jid`; RA refreshes its snapshot and descriptor. RA-valid candidates
are sent through `recovery_outline_safety_validate`. Existing request/session/turn
identifiers, sender checks, and state/snapshot/descriptor/rule fingerprints bind
evidence to that candidate. Missing, malformed, stale, or timed-out evidence
must not become permission to execute. Preserve the existing protocol identities
and `SystemBridge` interface.

## Paper contract

Synthesis starts at `x_0^R = x_F`. PA proposes `e_k^R` from projected conditions
and public capabilities. It does not supply the complete private RA transition
model to the proposal LLM. After RA feasibility and CCA safety succeed, append
`s_k^R = s_{k-1}^R e_k^R`, with `s_0^R = ε`; rejection leaves the accepted state
and prefix unchanged.

RA's backward derivation collects

\[
\mathcal{P}_{i,k}^{R}
= \bigcup_{Tr_i(x,e)=x',\;\nu_i(x')=v_i'} \mathcal{P}_i(e).
\]

Its source transitions are support evidence, not competing PA recovery proposals
or a required execution sequence. Forward validation requires

\[
\forall\rho\in\mathcal{P}_{i,k}^{R},\quad
\exists\boldsymbol{\theta}_{\rho}\in\Theta_{i,\rho}:\;
C_{\rho}(v_i,v_i',\boldsymbol{\theta}_{\rho})=1.
\]

The implementation's `NEEDS_CONTEXT` distinguishes unavailable evidence from a
known failed condition; neither result permits admission.

CCA uses the deterministically projected `ℓ(x_k^R)` to extend
`π(s_F s^R_{k-1} e_k^R) = π(s_F s^R_{k-1}) ℓ(x_k^R)` and checks accepting-state
reachability from every updated applicable monitor. This is prefix validation,
not final acceptance or a guarantee that independently possible continuations
have a compatible joint realization.

## Symbolic transition feasibility

At outline turn `k`, `q_k` is the PA-projected symbolic state. In the current
incremental candidate schema the LLM authors `expected_end_state`; PA derives
`expected_start_state` from the projected state in `_derive_candidate_outline_task`.
The validator then compares every declared start field exactly. Do not describe
the current model-facing schema as requiring the LLM to author both objects.
These checks cover a declared partial valuation, not automatic full-state equality.

An accepted candidate contributes a recovery-session transition
`(q_k, event_name, q_{k+1})`. A new `event_name` need not already be in a nominal
RA event alphabet. Its spelling supplies no motion, acquisition, release,
progress, or safety semantics. Syntactically accepted new state labels likewise
provide no authority to extend a live descriptor or invent physical capabilities.
The RA successor-support check below can reject them.

For part-affecting transitions, preserve `held_part`, `part_state`, and
`part_location` consistently with the responsible resource and Product state:

- Acquisition and release require an explicit, non-null destination location.
- A held part uses its RA's exact declared carried-part location.
- Product and resource holder facts agree; two resources cannot hold one part.
- A part cannot relocate without an explicit carrier/control relation.
- Reaching a goal location while the part remains held does not establish
  recovery completion or clear an unfinished processing/assembly obligation.

Constraint-derived feedback may provide an exact valid token after a failed
check. This is explicit validator guidance, not independent discovery by the
LLM. Record its source and invalidate it when the underlying evidence changes.

`no_state_change` and `label_only_state_change` are selection/admission findings,
not axioms of DES transition feasibility. Current regression coverage shows a
label-only candidate passing PA syntax and symbolic checks, then failing RA
primitive support with `unsupported_successor_condition`; the separate selector
classifier can also report `label_only_state_change`. Do not restore the older
blanket claim that the PA transition validator rejects every label-only change.

## Primitive feasibility and composition

### Current implementation

[ResourceAgent](../../cais_spade_llm/agents/resource_agent/resource_agent.py)
checks its transition conditions, calls
[`validate_primitive_support`](../../cais_spade_llm/resources/recovery_feasibility.py),
and, after support succeeds, applies the resource-specific physical pre-check.
The support checker performs two distinct operations:

1. Match all modeled successor valuations against the proposal's public
   resource-scoped `expected_end_state` fields, including parameter–effect
   bindings and preserved fields. The proposed event and its actual starting
   condition need not be predefined. Collect all contributing transitions.
2. Form the **union** of their primitive identities; require at least one
   demonstrated parameter/capability witness for **each** distinct primitive.
   A feasible subset or one feasible transition alternative is insufficient.

`primitive_support` retains `matching_transitions`, source `contributors`,
parameter bindings and attempts, `valuation_coverage`, `covered_valuation_fields`,
`deferred_valuation_fields`, and primitive/transition `composition_contract`.
Deduplication does not erase sources or imply that their parameter bindings
are mutually compatible. Conditional step guards remain composition evidence.
Product fields are explicitly deferred to complete-effect composition checks;
resource successor matching must not imply that a part has been staged.

Known empty support is rejected. Missing support metadata/evaluators or
incompletely searched parameter domains remain `NEEDS_CONTEXT` when no witness
exists. Parameter enumeration is bounded at 4,096 assignments per support check;
truncation without a witness is unresolved, not proof of infeasibility. A known
infeasible primitive rejects the union even if another primitive lacks evidence.
No matching successor produces `unsupported_successor_condition`.

PA independently reconstructs the expected union from the bound RA descriptor
and verifies matching transitions, primitive identities, proposal and descriptor
fingerprints, and witnesses. The evidence feeds composition without exposing the
complete private model in the outline prompt.

| Status | Meaning |
| --- | --- |
| `FEASIBLE` | The declared primitive-support condition has a demonstrated witness under the supplied evidence. |
| `INFEASIBLE` | A required primitive or known empty support is ruled out within that validator's coverage. |
| `NEEDS_CONTEXT` | Required capability, observation, planning, or parameter evidence is unavailable or incomplete. |

This support check is not a proof of full physical execution. Individual
primitive witnesses may not form one compatible ordered sequence. Composition
must establish consistent bindings, preconditions, intermediate effects, and
the complete intended successor before execution.

### Staging and airborne-release example

Suppose `ur5e-4` holds `gear_large` when `ur5e-3`'s part slips into its region.
PA can propose an unforeseen event to stage `gear_large`, with the observed
held-part start and a proposed released/staged successor. No predefined staging
recovery event is required for PA to make the proposal.

RA derives all primitive support matching that resource successor. For example,
`release_part` may contribute the effect `held_part = null`. That effect alone
does not establish `part_location = M1 staging tray` or any other staging target.
Required motion/target primitives must have capability witnesses; missing target
pose, reachability, capacity, or interlock evidence remains `NEEDS_CONTEXT`.

Composition must bind a suitable destination, compute its target, move to that
same target with consistent coordinates, satisfy release conditions, and achieve
the **complete** resource/product successor. The trace validator rejects release
without a landing fact, a landing assembled from inconsistent coordinates, and
release after moving away from the previously established target. Execution still
needs acknowledged motion and eventual observation of the released part; a
symbolic landing fact does not prove stable contact or actual placement.

For the subsequent slipped-part grasp, backward support identifies primitives
associated with the intended acquired-part valuation. Forward capability checks
need the specific part's observed pose and evidence that the robot can reach and
grasp it. Composition establishes approach-before-grasp, shared bindings, and
intermediate conditions. The primitive union is an unordered support set, not an
automatically synthesized approach-and-grasp plan.

### Resource coverage and remaining limitations

Contract audit for the four-scenario roadmap:

| Primitive/resource family | Available contract/evidence | Remaining admission evidence |
| --- | --- | --- |
| Robot target computation, motion, `grasp_part`, `release_part` | Parameter schemas, private prerequisites/effects, parameter–effect bindings, trace facts, target occupancy/custody, witness attempts; release landing invalidation is enforced. | Collision/IK trajectories, calibrated grasp and stable placement/contact observations where required. |
| Conveyor handoff and Buffer For Machined parts | Nominal resource programs and configured locations. | Novel handoff contracts, exact loading space/capacity and custody observations for the failure routes. |
| KMR docking, arm motion, `side_access` unload and transport | Nominal KMR primitives/programs and informal local contracts. | Full RA successor descriptor and non-executing capability witnesses for new docking/access/transport routes; nominal executability alone is insufficient. |
| M1 extraction and M2 continuation | Nominal machining/program/configuration metadata. | Recoverable WIP/progress, stopped/interlocked access, extraction support, M2 compatibility and observed resumption effects. |
| ur5e-3/ur5e-4 slippage | Shared robot contracts and legacy saved-context regression support. | Matching two-robot failure snapshot, reachable slipped part, suitable staging, and both interrupted-task obligations. |

Missing contracts/evidence hold admission; this change does not invent capabilities
or populate complete physical witnesses for all four scenarios. Configured nominal
programs are retained even if a future baseline audit finds an existing recovery.

Robot checks include named-pose support, availability, occupancy/custody,
grounded targets, and configured workspace bounds. The current evaluator returns
`NEEDS_CONTEXT` for `move_relative`, `move_joints`, and `rotate_joint` when live
robot-model planning is required. Bounds checks are not complete reachability,
IK, collision, trajectory, or grasp validation. Report the actual evaluator
coverage for each resource rather than treating robot, printer, machine,
Conveyor, and KMR checks as equivalent.

RobotAgent's physical pre-check still contains an abstract `idle` allowance.
The primitive-support gate now precedes it, so this allowance alone does not
establish a supported successor. Its eventual declaration-based replacement
belongs to formal-contract alignment; document the remaining exception rather
than claiming state wording is never inspected anywhere in current code.

Later unexecuted outline transitions intentionally combine projected dynamic
facts with fresh capability evidence. Keep the projection, live observations,
and provenance distinct. Neither a dry-run approval nor simulated attachment
establishes physical grasp contact or hardware execution.

## Safety validation and supervision

### Current outline safety behavior

[Outline safety projection](../../cais_spade_llm/agents/central_controller/outline_macro_safety.py)
uses pre-transition resources/parts, projected successors, running APs, loaded
rules, and `safety_dfa_states_before`. Temporary monitor evaluation does not
mutate the live CCA. Accepted transitions return `safety_dfa_states_after` for
the next projected step; stale live-state/rule fingerprints invalidate evidence.

Applicable rule scopes include `nominal`, `recovery`, `bridge`, and `both`.
For those rules, missing `dfa_dot`, unprojected propositions, or unsupported
AP selector evaluators raise errors rather than silently excluding the rule.
A constant rule with no APs can be valid; an empty AP valuation is still a trace
step and must be evaluated.

The tested `OnlineSafetyMonitor.online_safety_validation` path rejects missing
or ambiguous matching steps and states without a satisfiable path to acceptance.
It distinguishes accepting states, nonaccepting prefixes with accepting
continuations, and nonaccepting cycles with no such continuation. Multiple
accepting states and accepting `true` self-loops are valid. These are current
checks, not merely optional future hardening.

`BaseSafetyChecker._delta` and outline validation now use the same deterministic
transition evaluation. Missing or ambiguous successors are rejected; there is no
implicit stutter or first-edge choice. The online monitor records malformed
committed history and holds subsequent admission rather than authorizing from a
stale monitor state. Accepted empty labels still advance each applicable DFA.

Records include all evaluated requirements (including passed checks), projected
labels, matched guards, source/target states, acceptance and accepting-state
reachability, and failure reasons. Rejected projections do not mutate monitor
states or the PA's accepted prefix/projected state.

Unknown rule scopes, missing required rule IDs/DFAs, missing AP mappings, and
missing state fields are errors. A **present null-valued field** is a declared
formal valuation and differs from an absent observation field. Physical evidence
remains the RA's responsibility; neither null nor a symbolic AP establishes it.
No configured rules supplies no hazard coverage and is recorded as
`no_applicable_rules`. The experiment gate requires a reviewed active rule set
and `diagnostic_cca_bypass=false`; historical `safety_none.txt` runs cannot serve
as safety-validation evidence. Outline safety does not prove safety throughout
an ungenerated primitive sequence.

### Event/state and finite-trace semantics

Retain both `ap_event` and `ap_state`: events cover task execution/entry,
whereas state APs retain occupancy between actions. Candidate checks include
current running events, persistent facts, and predicted successor facts.
Committed successor facts replace superseded state facts rather than accumulating
contradictory states indefinitely. Keep unrelated running-task facts that remain
relevant to the checked rule.

Use the existing finite-trace terminal empty-step convention consistently across
validation and supervision. Rule satisfaction on a prefix, an outstanding response
obligation, and acceptance when the plan ends are different conditions. A local
projection cannot simply erase empty valuations or steps whose event name does
not occur in a rule.

The existing plan model is `P = (X, E, T, x0, Xm)` with runtime product state
`(x, q_vec, sig)`. The supervisor's modeled winning set `W` describes a safe
accepting continuation. `preventive` and `reactive` use precomputed product
data; `truly_reactive` searches from the live state. Modes affect enforcement;
record the selected mode instead of assuming a precomputed winning set is always
used. A continuation in a supplied plan model is not an adversarial guarantee
against every future physical failure.

### Target deterministic rules and local composition

The [roadmap](IMPLEMENTATION_PLAN.md#milestone-2-deterministic-safety-construction)
requires deterministic applicability, exact binding, AP construction, LTLf, and
DFA generation from reviewed rules. Current post-outline generation already
constructs some formulas deterministically but still asks the LLM which source
rules/events are involved. Removing that selection authority remains planned.

Local products must include dependency closure or validated boundary contracts
for shared events, guard/update reads, custody, task prerequisites, and rule APs.
Preserve a joint accepting continuation for coupled obligations. Separate local
winning sets cannot establish global nonblockingness without the required
coordination conditions; see the [coordination-control reference](https://arxiv.org/abs/1307.4332).
Bounded or missing-evidence results remain inconclusive and cannot authorize
dispatch. The local backend and its performance claims remain unimplemented.

## Neurosymbolic Selection Method

### Formal target retained from the journal notes

`pure_llm` proposes exactly three one-action candidates and selects an index,
subject to PA/RA/CCA validation. `neurosymbolic` uses LLM proposals without LLM
ranking, compares validated projected successors, and uses a one-action horizon.
The current configuration is adaptive with proposal budget five; exactly-three
controlled comparisons require the planned evaluation option.

For current state `q`, `O(q)` contains exact recovery goals, continuation blockers,
reentry requirements, and CCA safety-condition identifiers. `Q_m^R` denotes
recovery-compatible marked states. Retain:

\[
\Gamma^R_{S/G}(q)
= \Gamma^R_G(q) \cap \Gamma^R_{\mathrm{RA}}(q) \cap \Gamma^R_S(q).
\]

`Gamma^R_G(q)` contains backward-relevant, bound RA-declared controllable events
whose exact guards hold. `Gamma^R_RA(q)` retains events physically supported by
the responsible RA, and `Gamma^R_S(q)` retains CCA-admissible events. Every
validated proposal `e_i` produces `q'_i = delta(q, e_i)`. Evidence is specific to
the bound resource, part, location, observations, and model/state fingerprints.

`Gamma^N_{S/G}(q)` contains exact nominal-reentry events with verified task guards
and CCA-admissible projections on the affected continuation paths. Unrelated
unfinished nominal tasks are not reentry progress.

A candidate progresses through the first satisfied case:

1. `O(q'_i)` strictly decreases without introducing an obligation.
2. `O(q'_i)` is unchanged and `Gamma^R_{S/G}(q'_i)` strictly contains
   `Gamma^R_{S/G}(q)`.
3. Both are unchanged and `Gamma^N_{S/G}(q'_i)` strictly contains
   `Gamma^N_{S/G}(q)`.

Candidate comparison follows this lexicographic preference. For equivalent or
incomparable survivors, the smallest existing exact-effect `candidate_id` is a
stable but arbitrary representative. Candidate position, `event_name`, rationale,
time, distance, energy, and effort supply no operational-superiority claim.
No progressing candidate requests revision; unchanged-evidence selection limits
must remain distinguishable from proof of infeasibility.

### Current mismatch and required correction

The current progression predicate still uses `bool(newly_enabled)` and
`bool(newly_enabled_nominal_reentry_events)`. It also has a separate
`bool(newly_cca_admissible_goal_recovery_events)` progression branch. Consequently,
current behavior is not identical to the three formal cases above.

Require `enabled_events_after > current_enabled_events` for strict recovery
expansion and `nominal_reentry_events_after > current_nominal_reentry_events`
for strict nominal expansion under the corresponding unchanged-set conditions.
Account for CCA admissibility within the defined enabled-event sets instead of
an independent undocumented progression route. Retain subsequent strict-superset
candidate dominance and exact-effect representative selection.

| Before | After | Strict expansion |
| --- | --- | --- |
| `{event_A}` | `{event_A, event_B}` | Yes |
| `{event_A}` | `{event_B}` | No: swap |
| `{event_A, event_B}` | `{event_A}` | No: loss |
| `{event_A}` | `{event_A}` | No: unchanged |

Extend the existing recovery tests for both enabled-event sets, CCA filtering,
obligation changes, incomparable successors, and nominal reentry. Do not delete
coverage for currently distinct validation stages when correcting selection.

The method remains one-step receding-horizon symbolic selection. It can reject
necessary temporary detours, depends on proposal/model coverage, and treats
obligations without cost or severity weighting. It is not global recovery search,
optimal supervisory control, guaranteed recovery, or a proof of full executability.

## Prompt evidence and information boundaries

### Implemented capture and inspection

The recovery UI renders saved `outline`, `primitive`, `safety`, and `full`
diagnostics from the reusable `scenario_runner` service. Live and dry-run calls
share prompt/schema builders, validation, selection and the structured provider
adapter. Inputs and referenced dependencies are fingerprinted, copied, verified,
and read from frozen paths. Observations carry synthetic/observed provenance;
in-process transport and explicit fixture-response replay remain visibly mocked.
One job, cancellation, partial evidence, and outline fingerprint checks persist
across navigation. Live/replay source participates in the fingerprint, so a
mocked outline cannot be reused as live validation evidence. Live snapshots with
active rules must include `recovery_safety_context.safety_dfa_states` and an
explicit `running_aps` list from the saved CCA history; missing history holds
admission. Configured injection targets and observation templates are not
promoted to observed part locations.

The shared adapter writes an immutable request immediately before every actual
application-level provider attempt and a raw response before parsing. It records
roles/content, tool definitions, tool calls/results, schemas, explicit settings,
application retries, malformed responses and errors. Each record has run, stage,
turn, call, tool-round and attempt identity. Transport credentials and SDK-private
HTTP retries are outside application-level capture. A record-storage failure
stops generation and cannot cause a successful provider call to be repeated.

The UI reads these files directly with bounded, read-only previews/downloads.
Candidate details include backward source transitions, primitive witnesses and
parameter attempts, missing evidence, CCA requirement/label/DFA checks, selection,
and projected changes. Passed, failed, unresolved/unavailable, skipped and
unrecorded checks remain distinct. A prepared preview says **not sent**; absent
exact capture says **not captured**. Legacy summaries and `latest` files cannot
prove a historical outbound payload.

Only legacy reviewed contexts are currently available; the four larger-setup
scenario fixtures/adapters remain a roadmap gate. Deterministic safety
**construction** is not implemented: current safety-generation requests are
recorded as real model calls. The deterministic outline DFA check is a separate
validator and makes no model call.

The current incremental outline prompt includes a compact **Accepted Transition
Prefix (already applied; do not repeat)** with accepted identifiers/event names;
it does not expose the full accepted state objects in that block. The older
README claim that accepted event names never enter later prompts is superseded.
Any retained same-run history must be documented as model input and visible in
the exact record; it is distinct from an evaluator's expected solution.

### Source-of-truth contract and exposure policy

Use the same prompt builders, schemas, validation/selection functions, and
provider-call capture for runtime and diagnostics. Persist each actual application
attempt at the call boundary, including all roles, exact text, schemas, tool
messages, settings, retries, raw returned output before parsing, and errors.
Separate immutable request/response evidence from readable summaries and parsed
objects. The UI reads that evidence directly. Prepared previews say **not sent**;
missing exact capture says **not captured**. Never upgrade inferred legacy text
or a `latest` copy into an exact historical request.

| Information | Model-facing boundary |
| --- | --- |
| Actual observations, current/projected state, legitimate goals, capability vocabulary, safety constraints, applicable validator feedback | Allowed with source/provenance; no fabricated observations or prescribed recovery hidden in descriptions. |
| Accepted same-run history | Disclose the exact stage-specific fields actually supplied; never substitute a pre-authored reference trace. |
| Validated outline and responsible resource's primitive catalog | Allowed for primitive generation to realize that run's accepted transition. |
| Complete private RA descriptors, selector rankings, and unrelated internal evidence | Keep outside prompts unless explicitly part of the documented method. |
| Expected recovery sequences, golden responses/programs, evaluator labels, and scoring rules | Evaluator-only; not exposed by prompt/context tools. Explicit replay is loaded separately and labeled. |
| Operator guidance | Record explicitly; assisted trials are distinct from unassisted evaluation. |
| Fixture responses | Explicit replay only; label as scripted, never silently substitute on live API failure. |

Capture tools as well as initial prompts: a clean first request is insufficient
if a context response leaks an expected recovery. Transport-spy equality checks,
evaluator-data canaries, and order-variation tests establish specific information
controls, not universal absence of bias. Dry runs use frozen dependencies and
snapshot-backed, non-executing adapters; `NEEDS_CONTEXT` remains an honest outcome.

## Implementation and regression references

| Concern | Existing source and coverage |
| --- | --- |
| Candidate preparation and selection | [multi_turn.py](../../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn.py), [outline selection](../../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_outline_generation.py), [recovery tests](../../test/test_case3_recovery_dryrun.py). |
| Primitive support and evidence handoff | [feasibility implementation](../../cais_spade_llm/resources/recovery_feasibility.py), [feasibility tests](../../test/test_recovery_feasibility.py). |
| CCA projection and DFA semantics | [outline projection](../../cais_spade_llm/agents/central_controller/outline_macro_safety.py), [online monitor](../../cais_spade_llm/agents/central_controller/online_safety_monitor.py), [shared checker](../../cais_spade_llm/agents/central_controller/base_safety_checker.py). |
| Plan products and supervision | [plan validator](../../cais_spade_llm/agents/central_controller/plan_safety_validator.py), [online supervisor](../../cais_spade_llm/agents/central_controller/online_safety_supervisor.py), [event/state AP design](../../cais_spade_llm/specification/safety/ap_state_ap_event_mutex_design.md). |
| Safety construction | [post-outline generation](../../cais_spade_llm/agents/central_controller/recovery_safety_generation.py), [SafetyLogic](../../cais_spade_llm/agents/central_controller/safety_logic.py). |
| Requests and UI evidence | [immutable recorder](../../cais_spade_llm/agents/shared_information/llm_request_records.py), [recording tests](../../test/test_recovery_request_records.py), [structured-call wrapper](../../cais_spade_llm/agents/shared_information/llm_agent.py), [artifact writer](../../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/recovery_artifacts.py), [recovery page](../../cais_spade_llm/ui/pages/recovery.py), [diagnostic tests](../../test/test_recovery_diagnostics.py), [project-page tests](../../test/test_project_pages.py). |

## Verification and claim limits

Verification on 2026-09-29:

| Check | Result |
| --- | --- |
| Focused suites: `test_recovery_feasibility`, `test_case3_recovery_dryrun`, `test_recovery_diagnostics`, `test_recovery_request_records`, `test_recovery_task_des_audit`, `test_project_pages`, `test_place_insert_release_only`, `test_gazebo_resource_programs` | **244 passed** (116.54 s). |
| Final request/artifact labeling checks | 10 passed; explicit not-sent summaries verified after the final wording change. |
| `poetry check` | Passed; existing Poetry metadata deprecation warnings remain. |
| `poetry run python -m compileall -q cais_spade_llm ros2` | Passed. |
| UI and scenario-runner `--help` | Passed without live ROS2 initialization. |
| Ruff on the new recorder/audit/runner and revised diagnostics/evidence-reader modules | Passed. |
| Manuscript text/hash, local document links, README anchors, `git diff --check` | Passed. |
| Original Case 3 regression preservation | All 58 prior tests retained; 5 additional cases. |

Tests use fixtures, transport spies, and saved observations; no live model,
Gazebo recovery, or hardware trial was performed. Regression coverage
includes primitive unions and novel/parameterized successors; missing witnesses;
airborne release, motion-away, axis/target-kind binding; DFA continuation, empty labels and malformed
steps; rejection without accepted-state changes; exact transport payloads, retries,
tool responses, malformed JSON, recording failure, evaluator/private-data canaries;
frozen dependencies, stale checkpoint rejection, cancellation, and UI evidence.

The separate [task-DES audit](../../cais_spade_llm/recovery_framework/task_des_audit.py)
reports modeled paths, complete explored absence, or inconclusive evidence. The
[four-scenario audit table](IMPLEMENTATION_PLAN.md#predefined-task-level-des-audit)
is currently inconclusive for every larger-setup scenario. Private RA primitive
support is not a predefined task-level recovery plan, and modeled alternatives
must not be suppressed. No new live Gazebo or hardware execution, complete
physical grasp/placement proof, deterministic construction, local-composition
performance, or experimental recovery-success result is claimed.

## Supplied manuscript excerpt

Source: user attachment `Pasted text.txt`, supplied in this conversation.
This exact excerpt governs feasibility and outline safety; surrounding paper
sections were not supplied. The text follows, including its unfinished figure
caption and final heading; trailing spaces on four lines have been removed.
The SHA-256 below identifies the original attachment bytes.

SHA-256: `1801901c7af1b94994143e472e6eabaaf66da1ad27fe98d1afacbc97c1c69513`.

```latex

\begin{figure}[pos=t]
\smallskip
\smallskip
    \captionsetup{belowskip=-1pt}
    \includegraphics[width=.48\textwidth]{figures/figure05_recovery_outline.png}
    \caption{Recovery task synthesis from the observed post-fault state. The~\gls{pa} incrementally generates recovery events, while the~\glspl{ra} and~\gls{cca} validate their feasibility and safety. The completed outline reconnects to the remaining product plan and proceeds to primitive composition and safety synthesis for execution.}
    \label{fig:recovery-outline}
        \vspace{-20pt}
\end{figure}


This section details recovery task synthesis, primitive composition, and safety monitor construction, which connect the observed post-fault condition to the remaining product plan.

\subsection{Recovery Task Synthesis}
\label{subsec:task-synthesis}

The neuro-symbolic architecture introduced in the previous section defines recovery through neural candidate generation followed by symbolic validation. This subsection details how these components are coordinated to construct the recovery string $s^R$ at runtime. As shown in~\Cref{fig:recovery-outline}, the procedure grounds the post-fault recovery context, incrementally synthesizes the recovery outline, and passes the completed outline to primitive composition and safety synthesis.

Recovery needs to account for the physical conditions produced by the fault because the expected nominal state no longer describes the system. The \gls{pa} therefore updates the knowledge base $\mathcal{K}$ with the failure context, observed resource and product conditions, remaining goals, and constraints, initializing synthesis at $x_0^R=x_F$.

The~\gls{pa} uses the~\gls{llm} to propose each recovery event $e_k^R$ using the projected state $x_{k-1}^R$ and available resource capabilities. A common output schema specifies the assigned resource and intended start and end states. These states provide conditions for validation even when the transition is absent from $Tr_i$. Candidates passing the~\gls{ra}'s feasibility validation and the~\gls{cca}'s safety validation extend the recovery prefix:
\vspace{-0.3\baselineskip}
\begin{equation}
s_k^R=s_{k-1}^Re_k^R,\qquad s_0^R=\epsilon,
\label{eq:recovery-prefix-update}
\end{equation}
where $\epsilon$ is the empty string. The projected state advances to $x_k^R$. Rejected candidates return feedback for revision without changing the accepted prefix.

Synthesis continues until the projected state is connected to the remaining product plan. The completed outline specifies task-level recovery behavior. These events still require executable primitive sequences and a safety monitor covering interactions with normal tasks and recovery tasks. The outline is therefore passed to the~\glspl{ra} for primitive composition and to the~\gls{cca} for safety synthesis. These procedures are detailed in their respective subsections.

\subsubsection{Feasibility Validation}
\label{subsubsec:feasibility-validation}

\begin{algorithm}
\caption{Capability-Based Feasibility Validation}
\label{alg:feasibility_validation}
\begin{algorithmic}[1]
\algrenewcommand\algorithmicindent{1em}
\footnotesize
\Procedure{FeasibilityValidation}
{$Tr_i,\mathcal{P}_i,v_i,v_i'$}
    \Statex \textit{Backward derivation}
    \State $\mathcal{P}_{i,k}^{R} \gets \emptyset$
    \For{$Tr_i(x,e)=x'$}
        \If{$\nu_i(x')=v_i'$}
            \State $\mathcal{P}_{i,k}^{R}
            \gets
            \mathcal{P}_{i,k}^{R}
            \cup
            \mathcal{P}_i(e)$
        \EndIf
    \EndFor
    \If{$\mathcal{P}_{i,k}^{R}=\emptyset$}
        \State \Return $\mathrm{false}$
    \EndIf

    \Statex \textit{Forward validation}

    \For{$\rho\in\mathcal{P}_{i,k}^{R}$}
        \If{$\nexists\,
        \boldsymbol{\theta}_{\rho}\in\Theta_{i,\rho}
        \;:\;
        C_{\rho}(v_i,v_i',\boldsymbol{\theta}_{\rho})=1$}
            \State \Return $\mathrm{false}$
        \EndIf
    \EndFor

    \State \Return $(\mathrm{true},\mathcal{P}_{i,k}^{R})$

\EndProcedure
\end{algorithmic}
\end{algorithm}

Each candidate recovery event must be supported by the assigned resource’s capabilities before it can extend the recovery outline. Therefore, feasibility validation checks whether the assigned resource can support a proposed recovery event under the projected recovery conditions.~\Cref{alg:feasibility_validation} combines backward derivation of required primitives with forward validation of their capability constraints.

Backward derivation identifies transitions \(Tr_i(x,e)=x'\) whose successor states have the resource condition \(v_i'\). Let \(\mathcal{P}_i(e)\subseteq\mathcal{P}_i\) be the primitives associated with a predefined event \(e\). For candidate recovery event \(e_k^R\), the primitives associated with transitions are collected in \(\mathcal{P}_{i,k}^{R}\subseteq\mathcal{P}_i\). This step identifies primitives for the intended recovery outcome without requiring the proposed recovery transition itself to be predefined. Identifying primitive support does not ensure that the corresponding capabilities are usable. Forward validation therefore requires
\vspace{-0.3\baselineskip}
\begin{equation}
\forall\rho\in\mathcal{P}_{i,k}^{R},\quad
\exists\boldsymbol{\theta}_{\rho}\in\Theta_{i,\rho}:
C_{\rho}(v_i,v_i',\boldsymbol{\theta}_{\rho})=1,
\label{eq:recovery-forward-feasibility}
\end{equation}
where $C_\rho$ checks whether the parameter assignment satisfies the resource's capability constraints, such as whether a required location is reachable. The candidate is rejected when the selected primitive lacks a satisfying parameter assignment. This check identifies whether the candidate event is supported.

\subsubsection{Safety Validation}
\label{subsubsec:safety-validation}

While recovery task synthesis generates actions to restore a path to task completion, these actions can introduce unsafe interactions with normal tasks and other recovery tasks. For example, a proposed recovery event can lead the system to an unsafe state by violating mutual-exclusion requirements or precedence constraints. The~\gls{cca} therefore evaluates each proposed event against the~\gls{ltlf} specifications in $\Phi$ using their corresponding~\gls{dfa} $\mathcal{A}_{\varphi}$.

For each candidate $e_k^R$, the~\gls{cca} evaluates the proposed effects using the projected resource and product conditions. The projected successor $x_k^R$ incorporates the candidate's effects. The \gls{cca} deterministically evaluates $\ell(x_k^R)$ from these conditions, extending the proposition trace as
\vspace{-0.3\baselineskip}
\begin{equation}
\pi(s_Fs_{k-1}^Re_k^R)
=
\pi(s_Fs_{k-1}^R)\,\ell(x_k^R).
\label{eq:recovery-safety-trace-update}
\end{equation}

The resulting label advances each applicable~\gls{dfa} monitor from the state reached after processing $\pi(s_Fs_{k-1}^R)$. The~\gls{cca} checks whether an accepting state remains reachable from each updated monitor state. If this condition fails for any applicable specification, the candidate is rejected. For rejected candidates, the~\gls{cca} returns the corresponding requirement and projected conditions as feedback for revision.

\subsection{Primitive Composition}
\label{subsec:primitive_composition}

\begin{figure}[pos=t]
\smallskip
\smallskip
    \captionsetup{belowskip=-1pt}
    \includegraphics[width=.48\textwidth]{figures/figure06_primitive-composition.png}
    \caption{dd}
    \label{fig:primitive-composition}
        \vspace{-20pt}
\end{figure}

Primitive composition translates validated recovery events into executable routines for the assigned resources. Primitives specify local execution conditions and effects but do not encode the task-specific dependencies needed to achieve a recovery outcome. The~\gls{ra} therefore uses the~\gls{llm} to determine primitive ordering and parameter assignments from the recovery context. Related work has also explored \gls{llm}-based program composition and code reuse~\cite{liang2023code,singh2023progprompt}.

As illustrated in~\Cref{fig:primitive-composition}, the~\gls{ra} constructs the context for each $e_k^R$ using its start condition $v_i$, intended end condition $v_i'$, and relevant product information. The~\gls{llm} dynamically retrieves primitives and predefined function decompositions from $\mathcal{K}$. Using this information, the~\gls{ra} constructs $\mathit{PC}_k$ by selecting and ordering primitives from $\mathcal{P}_i$ and grounding their parameters.

The~\gls{ra} then checks the generated composition's structure, parameter bindings, and declared primitives against the projected resource conditions. The composition is required to achieve the intended condition according to~\eqref{eq:recovery_sequence_feasibility}. The resulting program thus provides the executable behavior needed for recovery.

\subsection{Safety Monitor Construction and Validation}
\label{subsec:runtime-verification}

```
