# Recovery framework roadmap, including UI dry runs and trustworthy prompt inspection

**Status reviewed on 2026-10-08:** nominal and recovery physical admission now
share `RecoveryCompositionAdmission`, its admission lock, command ledger,
reservations and physical execution history. `EnvironmentAdmission` retains
nominal eligibility and joint-completion authority through `commit_prepared`.
Exact prepared command ownership and authenticated feedback plumbing are
implemented. The current target is the user's four-step model-based Gazebo
method: fixed AP definitions, recovery events mapped to known primitives,
intermediate motion and modeled failure AP evaluation, and local composition with
relevant nominal operations. Stronger physical guarantees and automatic nominal
resumption are explicitly outside this implementation. Background Gazebo checks captured 11 controllers
and all 12 configured resources with both predefined specifications active and
`diagnostic_cca_bypass=false`. The supplied single-motion path now reaches an
allowed common composition graph with 67 instantiated rules and a committed
CCA grant. In `acceptance-11`, the native controller executed the exact authorized
trajectory and CCA committed authenticated observed completion once. This passes
the first empty-custody supplied `move_cartesian` gate under explicit models;
`physical_execution_verified` remains false. Earlier failed feedback and its
reservation remain recorded without retroactive completion. Both dedicated Gazebo
launches are stopped. Live occupied-region rejection, ongoing nominal composition,
modeled failure branches and retained history across successive executions remain
unfinished. Full `part_slippage` recovery and the four failure scenarios have not
been established.

**Roadmap maintenance requirement:** after every implementation or verification
update, synchronize the current status and dependencies in this file and append
a dated entry containing changes, checks performed, evidence paths, remaining
blockers and the next acceptance gate. This is the single current roadmap; retain
historical results as dated evidence rather than treating them as current results.

The accepted implementation grounds the existing CCA path for nominal and
recovery execution under explicit motion and failure-model assumptions, preserving
the Gazebo plant and actuator behavior. CCA must still evaluate intermediate motion,
include relevant nominal work, bind grants to exact prepared commands, and retain
observed monitor history. Missing model inputs remain `NEEDS_CONTEXT`. Model-based
admission does not assert proven bounds on every physical deviation, contact or
stopping motion. Use background Gazebo on school WSL. Actual recovery-event
generation for the four failure scenarios follows validation of this CCA path;
automatic nominal resumption and full physical guarantees are later milestones.
No new Python modules are required.

**Historical status reviewed on 2026-10-05 (superseded where stated above):
paper-aligned RA derivation, CCA outline checks,
and diagnostic evidence are implemented; local parallel composition is implemented
for nominal execution through `EnvironmentRuntime`. The new offline slice implements
reusable reviewed LTLf checking over primitive observations, automatic grounding
from the complete frozen scene, and synthetic KMR recovery evidence. Existing
structured `ap_event` / `ap_state` descriptors connect through explicit task/state
evidence without changing their identities or substituting occupancy. Bounded offline
local parallel composition now connects these observations to generated recovery
events and included already-running work. The `validated` route now registers complete
recovery sequences with CCA and requires an event-specific composition grant before
RA invokes primitives. Predefined AP meanings, formulas, and scopes are now a
reviewed input assumption. A selectable document compiles deterministically and
CCA uses the same fixed definitions for grounded nominal/recovery admission;
predefined safety never calls the LLM to replace them. This connection is verified
with explicit mock executors. Resource-owned non-dispatching preparation and the
recovery page's Prepare and check action now save a detached CCA result. Registered
`scene.resource_models` and owner-provided pure models support additional resources
without changes to AP evaluation, LTLf compilation, or local composition. Existing
UR5e controller preparation and KMR offline evidence remain supported. The supplied-motion
slice adds a read-only Gazebo attachment/state reader, registered preparation assembly,
and continuous joint-motion bounds with uncertainty in the same composition search.
Its focused software regressions pass; the dedicated Gazebo attempts remain
`NEEDS_CONTEXT`, so safe/conflicting live-motion acceptance is explicitly incomplete.
Active-rule Gazebo execution, nominal resumption, and the legacy
`receiving_region_entry` counterexample remain open.**

The current focused experiment uses three supplied mock `part_slippage` plans
and synthetic primitive observations with the unchanged predefined specifications.
The [computed report](../../test/fixtures/part_slippage/predefined_safety/REPORT.md)
records a mutex-only rejection, a precedence-only rejection, and satisfaction of
both rules for the correctly ordered trace. This offline experiment does not
depend on Gazebo preparation; it tests checking newly named recovery behavior.
The mock inputs use the existing offline observation contract and are not native
UR5e controller dispatch programs or newly collected LLM outputs.
The subsequent [Gazebo companions](../../test/fixtures/part_slippage/gazebo/README.md)
preserve those identities. Their live preflight captured all 12 configured resources
and resolved the static board envelope, but did not stage or execute any trial.
The requested output is now **one silent, captioned 20× video**, showing the
separately staged mutex, precedence and safe trials in sequence. Both predefined
specifications must be active together in every trial. The combined exporter is
implemented and tested with encoding fixtures; the live video remains unproduced.
Missing live evidence is not a mutex or precedence counterexample.
The 2026-09-29 source baseline was `c485e51`; the 2026-10-03 review inspected
`e64d694` and the new, untracked offline primitive safety files. This is the sole current
roadmap for Journal Paper 2. Software checks do not establish live recovery or
hardware safety.

**Prepare and check** now implements capture from registered resource owners,
preparation through their native contracts, and checking supported evidence using
a detached CCA snapshot. Synthetic machine, Conveyor, Buffer, seven-joint and
additional-resource providers exercise the common path. The first supplied candidates
use the existing UR5e controllers; configured fixed equipment and stationary KMR
now expose stamped physics observations through registered owners. Continuous
bounds preserve prepared joint interpolation and do not infer straight TCP paths.
The earlier live preparation attempts were blocked before guarded execution:
observed link motion, missing controller idle records and incomplete part geometry
blocked those sessions; a later retry also timed out reading the service. Those
single-motion blockers are superseded by the 2026-10-08 model-based acceptance
recorded above. Missing geometry, custody evidence, stationary
coverage, or supported motion interpolation remains `NEEDS_CONTEXT`. It cannot
install an execution proof, grant permission, or advance either monitor history.
The later part_slippage captures obtain the board envelope from its explicitly
configured static fixture models. An opt-in read-only controller query now reports
holding, active/pending goals, controller incarnation and command revision. A
dedicated launch with navigation disabled returned all 11 controller observations
covering the 17 configured action endpoints, alongside the complete physics
snapshot. This establishes current idle evidence, not future stationary coverage.
Fresh checkpoint validation and owner model evidence are exercised by the first
supplied-motion acceptance; prepared native custody execution remains open.
Pure continuous owner models now support explicit grasp,
carried-part geometry and release effects in synthetic checking; the live controller
does not yet supply or execute that complete contract.

## Summary

Verify newly generated recovery behavior while preserving existing specifications,
ongoing work, and valid nominal continuation. Assume that safety specifications,
AP meanings, and formulas are supplied and reviewed, as in the journal paper.
The LLM may propose recovery behavior; it does not discover or reinterpret these
selected specifications. Establish declared trace semantics,
resource-owned primitive effects, reusable formula evaluation, and deterministic
applicability, then bounded offline recovery composition using the existing local
parallel composition machinery and authoritative CCA admission. Evaluate supplied
mock recovery plans with synthetic evidence independently of live preparation.
Complete live preparation and execution-observation contracts before executing
four recovery scenarios. Use Gazebo for execution
validation; hardware experiments remain later work.

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
| Recovery safety generation | Legacy offline `receiving_region_entry` and `shared_area_mutex` use fixed DFAs and 1D observations. Automatic offline grounding now derives the complete population from `build_environment_models(scene)`: 12 resources and 66 mutex pairs, without participant lists or exclusions. Reviewed scopes and explicit physical/task/state evidence preserve AP meanings. Predefined mode compiles the supplied definitions unchanged and uses them throughout CCA; legacy natural-language mode retains LLM rule/event selection. | 2: resolve the legacy counterexample; validate reviewed hazard coverage and trusted evidence for live admission. |
| CCA composition | As of 2026-10-08, `RecoveryCompositionAdmission` shares nominal/recovery physical admission, command authorization, reservations and history; `EnvironmentAdmission.commit_prepared` retains nominal commitment. Local parallel composition is implemented for nominal `EnvironmentRuntime` execution, including dependency closure, joint completion analysis, caching, snapshot validation, and atomic CCA admission. A separate offline path composes generated recovery events with fixed running work and reviewed primitive observations using finite supplied start alternatives. The `validated` PA → RA → CCA route now registers the sequence and gates each requested start against the retained composition and observed history. Global plan FSA and active-window runtime paths remain. The first supplied empty-custody `move_cartesian` now passed live common composition, exact authorization, execution and one observed completion with both selected specifications active and bypass disabled. | 3: validate occupied-region rejection in Gazebo, then connect relevant moving nominal work, remaining recovery events, modeled failure AP observations and retained continuous history across live programs. Full physical guarantees and automatic nominal resumption are later milestones. `X` remains unsupported. |
| KMR recovery preparation | The synthetic `storage_interruption` fixture has four events and 22 primitives: stage `KET8_Square_8mm`, recover `KET4_Square_4mm`, then reacquire `KET8_Square_8mm`. Companions supply complete 12-resource evidence and two explicit timing alternatives with already-running `ur5e-3` withdrawal. Original fixture JSONs are retained. | Establish physical feasibility and later pending M1 delivery separately; the offline endpoint restores the Storage checkpoint. |
| Mock part_slippage safety experiment | Three supplied plans each contain seven newly identified events and 19 modeled primitives. The unchanged predefined mutex and strict precedence produce the expected separate violations and a complete safe trace. All 12 configured resources and 66 mutex pairs are included; event renaming preserves physical AP values. Separate Gazebo companions retain these identifiers; the combined 20× exporter preserves distinct trial evidence. | Complete native preparation, staged checkpoints and observed CCA-granted execution before publishing one video of the three trials with both specifications active. No live violation or successful recovery has been established. Actual LLM generation remains a separate evaluation. |
| Diagnostic UI | Reusable `scenario_runner`, frozen dependencies, explicit live/replay modes, immutable per-attempt capture, RA/CCA evidence, and input inspection. | Reviewed fixtures/snapshots and resource adapters for the four larger-setup failures; deterministic `safety` mode depends on milestone 2. |
| Physical failure scenarios | Simulation failure injection support is implemented; the [2026-10-02 validation notes](../../docs/conveyor_breakdown.md#validation-on-2026-10-02) record blocked checkpoint attempts. | 5: establish checkpoint acceptance with CCA enabled, then recovery execution, observation, and resumption. |
| Resource-owned preparation | Recovery-page Prepare and check uses common owner hooks and a registered entity reader. `scene.resource_models` appends trusted declarations and preserves existing owner identities; native programs stay in `resource_programs.resources`. Pure models provide supported physical effects independently of AP meanings. A synthetic additional resource produces 78 mutex pairs; equipment and seven-joint mocks use the same checking path. Existing UR5e `move_cartesian` planning remains in its controller. The read-only Gazebo service and configured owner readers capture the complete population; continuous polynomial/FK bounds feed the same composition path. Static constituent geometry resolves the empty board carrier envelope. The opt-in controller query establishes current idle ownership; synthetic continuous custody effects and explicit joint-error envelopes extend detached checking. | Exact prepared command ownership and authenticated feedback passed the first supplied live motion. Physical tracking, future stationary containment and stopping guarantees remain unavailable and outside this scope; the accepted path uses explicit models. Complete conflicting supplied-motion acceptance, then native grasp/release, observed carried-part transforms and required KMR/machine/Conveyor/Buffer effects. Registration and idle observations alone supply no physical execution guarantee. |
| Paper evaluation | Protocol specified here; no new trials collected by this change. | Experimental design after the relevant acceptance gates. |

The 2026-09-29 baseline settings selected `safety_none.txt`, `diagnostic_cca_bypass=true`,
`neurosymbolic`, `action_horizon=1`, `candidate_count="adaptive"`, and
`candidate_proposal_budget=5`. Historical bypass runs establish execution evidence,
not active-rule CCA safety. See [the retained run](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/README.md).

The RA → CCA → UI implementation slice is available for saved-context inspection.
Milestone 2 treats the reviewed AP/formula catalog as an input assumption, then
proceeds through trace semantics, resource-owned effects, deterministic compilation,
and automatic applicability. Safety-page compilation, saved artifacts, bundles,
and CCA preserve this same predefined source and its fingerprints. Automatic population and binding
construction now covers every configured resource, including additional registered
`scene.resource_models`, with no exclusion pruning. New supported actions enter
through trusted owner contracts. Missing declarations/providers/evidence remain
unavailable; no shared AP or composition branch dispatches on a new resource name.
Task/state evidence adds boundaries to the same frozen physical trace; pending
requirements and explicit finite-trace completion remain visible. The legacy receiving-region
counterexample and full reviewed hazard coverage remain open. Nominal local parallel
composition already exists. The offline milestone 3 path now uses grounded primitive
evidence and replayed monitor history for finite generated recovery alternatives;
CCA now adds the authoritative generated-event admission gate on the existing
`validated` route. Owner-supplied live geometry, trajectories, timing, acknowledgements,
and observation support remain required; missing evidence blocks admission. The
runtime recovery path still builds a global FSA after inserting recovery macros.

Supply reviewed staging/grasp/transport contracts and failure snapshots alongside
this work, including physical evidence that the current validators report as
`NEEDS_CONTEXT`. Milestone 4's remaining scenario fixtures can proceed alongside
2–3; its current `safety` stage still records the existing LLM-selected grounding
method. Milestone 5 depends on the relevant capabilities and validated safety
paths. The supplied mock `part_slippage` comparison uses the existing offline
checker and does not require live observation, failure injection, or an LLM call.
For later live trials, first verify non-dispatching preparation through resource-owned contracts,
including all configured resources and unchanged CCA histories. The reusable path
is tested with synthetic providers. Supplied recovery events and programs are the test
inputs; neither failure injection nor recovery generation is needed for this next gate.
The dedicated live session collected observations but did not establish either a safe
moving candidate or a definite mutex counterexample. Complete those two checks and
the missing-evidence negative case before guarded execution.
For the requested three-case part_slippage recording, use separately staged,
observed post-slippage checkpoints and preserve the original mock identities.
Keep both specifications in the same composition for all three cases. Publish
one 20× video with clearly labeled trial boundaries; keep each run's original
CCA decisions and observations separately. Encoding tests do not establish any
of those live outcomes.
The staging does not demonstrate the drop. Missing live preparation/admission
support must leave acceptance incomplete; publish no substitute safety-rejection
video for `NEEDS_CONTEXT`.
Establish a fresh nominal Gazebo baseline with active rules and
`diagnostic_cca_bypass=false` before recovery/resumption experiments. Collect
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

### Implemented offline primitive checking (2026-10-03)

The [specifications and evidence contract](../../cais_spade_llm/specification/safety/primitive_observation_safety.md)
define `receiving_region_entry` and `shared_area_mutex` with explicit resource,
part, and region bindings. The
[offline CCA checker](../../cais_spade_llm/agents/central_controller/primitive_program_safety.py)
instantiates concrete bindings and the two fixed specification DFAs deterministically,
without an LLM call. It does not provide arbitrary LTLf formula compilation.

- [Resource-owned primitive observations](../../cais_spade_llm/resources/primitive_observations.py)
  model supplied `compute_place_targets`, `move_cartesian`, and `release_part`
  evidence under configurable 1D geometry. Joint trajectories share one time axis;
  checking includes the initial state, boundary crossings, and intervals between
  crossings. Closed intervals count contact as occupancy.
- `BaseSafetyChecker.transition_evidence` checks AP valuations separately for each
  `rule_id`. Missing evidence remains `NEEDS_CONTEXT`; rejected or unresolved
  candidates preserve the supplied DFA history. Accepted modeled snapshots retain
  occupancy, custody, and deposited-part containment between programs.
- Applicable bindings and resource-model evidence are supplied by the caller.
  Omitted requirements are not discovered or checked. Capacity checking and
  runtime admission integration remain deferred. This legacy checker can
  inspect supported nominal or generated recovery primitive programs offline;
  passing does not authorize execution.

The implemented 2026-10-04 offline extension separates these four stages:

1. Freeze one joint observation trace: initial state, primitive/trajectory/custody
   boundaries, region contact, and representative observations between boundaries.
   `X` means the next observation in this declared clock. Version and fingerprint
   the clock, scope, geometry, AP meanings, and compiled formulas. Continue only
   through contiguous slices of that same trace; completion adds no observation.
2. Add opt-in `geometry.dimension: 3` through
   [resource-owned 3D effects](../../cais_spade_llm/resources/primitive_observations_3d.py):
   fixed-orientation axis-aligned envelopes, piecewise-linear XYZ trajectories,
   separate KMR base/tool motion, release, stationary deposition, and reacquisition.
   Preserve `processCompleted`; modeled placement never establishes `assembly`.
3. Add [reviewed formula evaluation](../../cais_spade_llm/agents/central_controller/reviewed_primitive_program_safety.py)
   using `LTLfParser`/MONA without LLM calls or preview writes. Fixed AP evaluators
   bind exact identifiers; formulas may use `G`, `F`, `X`, and `U`. Results distinguish
   `prefix_checked`, `satisfied`, `violated`, and `unavailable`, with pending rule IDs.
   Satisfaction requires explicit completion at the actual trace end. A checked
   prefix does not establish a physically feasible joint continuation.
4. The first reviewed slice derived all 10 `shared_area_mutex` pairs from the four
   `ur5e` resources and `KMR` in its caller-declared study population. It required every selected catalog specification,
   participant, and motion/stationary witness. Apply the separate requirement
   “KET4_Square_4mm may enter assembly_board-v1 only after its trim with result: square is completed.”
   Ground entry from part geometry and completion from the exact process/result
   record in an explicitly complete checkpoint ledger. Missing evidence is unresolved.

The subsequent automatic grounding slice adds
[`validate_grounded_primitive_program_safety`](../../cais_spade_llm/agents/central_controller/offline_safety_grounding.py).
It freezes the supplied scene and uses every `build_environment_models(scene)` key:
all 12 resources generate 66 mutex instances. Reviewed `requirement_scopes` supply
the applicable region, part, process, and result; callers supply no resource pairs,
participant allowlists, or exclusion declarations. Every configured resource needs
geometry, a snapshot, and complete motion/stationary evidence. Fixed equipment uses
`stationary_only: true` without invented gripper states; resources declaring
`held_part` cannot use that flag to omit custody evidence. A declaration-only builder
extension permits an additional configured robot without inventing a nominal handling
assignment; 13 resources generate 78 mutex instances with the same catalog/scopes.

Existing structured APs retain their exact `full` and `label`. Explicit associations
ground task APs from complete task execution intervals and state APs from resource-owned
state evidence. Physical APs continue to use primitive observations. `place_approach`
never becomes an occupancy predicate. The combined clock includes physical contacts,
task start/end boundaries, and completed state updates jointly; task intervals are
start-inclusive and end-exclusive. `X` advances one observation on this declared clock.
Its separate version and fingerprint reject incompatible continuation records.
Per-AP results expose descriptors, bindings, source evidence, observation indices,
and Boolean values; physical counterexamples preserve recovery event and primitive-step
references. Formula evaluation delegates to the existing reviewed monitor. Both
earlier public checking entrypoints retain their contracts.

Trace preparation is also available internally without a verdict: composition
receives complete observations and per-`rule_id` valuations, including unsafe
branches. Its optional supplied decision times join the boundary set before
midpoint construction. The standalone checker keeps its existing `X`, continuation,
and explicit-completion meanings; the new composition path restricts formulas to
Boolean combinations of `G`, `F`, and `U` through parsed-formula inspection.

The [synthetic companion](../../test/fixtures/KMR_assembly_board-v1_recovery/storage_interruption/README.md)
retains all four original recovery events and 22 primitive parameters/references
over `[0, 22]`. Recovery restores KMR carrying `KET8_Square_8mm` at `Storage`;
`move_to_resource(M1) → place_approach → place_release` remains pending. This is
preparation for offline checking, not physical recovery or nominal-resumption evidence.
The legacy entrypoint and DFA-state meanings remain compatible. The new part-entry
predicate does not resolve or reinterpret the legacy mismatch below. Safety-page
requirements, selection, and approvals remain unchanged.

### Unresolved receiving-region counterexample

The specification's requirement is:

> An incoming part must not begin entering an applicable resource's receiving region while that resource contains another part.

Its `ap_event/physical_observation/receiving_region_entry` meaning is:

> The bound resource's incoming tool/part begins occupying the bound receiving region while carrying the bound part. Boundary contact counts as occupancy; initial occupancy does not create an entry.

These wordings are retained exactly. The current predicate detects entry from
combined robot/tool-or-carried-part occupancy. When the tool initially overlaps
the receiving region but the carried part starts outside, the part can enter
without a new occupancy edge, leaving `ap001=False` and returning `is_safe=True`.

The 2026-10-03 review reproduced this with `KMR` moving from `x=1.1` to `x=1.3`
over `[0.0, 1.0]`, resource footprint `[-0.2, 0.2]`, `attachment_offset=-0.3`,
and part `p` initially at `x=0.8` with footprint `[-0.05, 0.05]`. `M2` contains
`q`, and its receiving region is `[1.0, 1.2]`. At `time=0.75`, `p` reaches
`x=0.95` and touches the occupied region, but the checker reports no entry event.

Resolve this mismatch against the written incoming-part requirement and add
regression coverage before accepting this rule as complete. Preserve the exact
identifiers and make the requirement/AP decision explicit; do not silently
change either wording. The existing 98 passing tests do not cover this
counterexample. The separate part-specific AP in the new path does not fix it.

### Remaining deterministic construction and acceptance

Use `supplied reviewed AP meanings + LTLf + requirement scopes → deterministic DFA compilation → grounded bindings`.
The [predefined document](../../cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt)
contains separate assembly-board mutex and gear_small-before-KET4 entry specifications.
The Safety page displays their exact descriptors, bindings, supplied LTLf, and
compiled DFAs under **Compile DFA**. Its complete text hash covers definitions and
bindings; product geometry has a separate validated fingerprint. New review and
explicit activation remain required; narrower historical previews confer no approval.

`G !(ap001 & ap002)` expands to all 66 resource pairs in the current scene without
an allowlist. Labels remain local to each instance. Precedence uses
`(!ap001 U (ap002 & !ap001)) | G !ap001`, with
`ap_event/physical_observation/part_region_entry` bound to `KET4_Square_4mm` and
`assembly_board-v1`, and `ap_state/processCompleted/process_target_completed`
bound to `gear_small`, `assembly`, and `Gear_Plate/Gear_Shaft_1`. The latter target
is checked against configured product geometry. Completion must precede entry;
first simultaneous completion and entry fail. Initial occupancy creates no entry.
`process_result_completed` retains its existing meaning.

Keep LLM recovery-action and primitive proposals. In predefined mode CCA associates
recovery scopes with these fixed definitions, never with replacement generated rules.
Invalid source/artifacts/compiler support block admission without a generation
fallback. The [legacy generation path](../../cais_spade_llm/agents/central_controller/recovery_safety_generation.py)
still uses `ask_llm_structured` for legacy natural-language selections. Automatic
grounding covers the complete configured population for every supplied requirement;
it does not discover missing hazards. See [the predefined contract](../../docs/predefined_safety.md).

The [resource-owned preparation contract](../../docs/gazebo_safety_preparation.md) retains
UR5e's exact `x`, `y`, `z`, `speed`, and quaternion parameters separately from
the physical observation evidence. Its configured joint identities, trajectory points, derivatives,
nanosecond timing, custody, checkpoint/configuration fingerprints and original
event/step references are saved. A joint plan alone never establishes linear TCP
motion. Legacy evidence retains its separate fixed-orientation Cartesian and resource-envelope
coverage. The new optional `continuous_motion` owner effect preserves spline
positions, derivatives and nanosecond timing; rational polynomial bounds and
outward-rounded interval FK enclose configured link envelopes. Physical occupancy
is definite or possible, with unresolved alternatives retained rather than treated
as false. Resource names and AP labels do not select this model. Version 3 owner
effects add explicit grasp/release boundaries and local carried-part envelopes.
Custody, geometry, inventories and deposited stationary coverage must agree;
these effects preserve `processCompleted`. The continuous custody clock is
`continuous_physical_boundaries_v2`. Supplied joint-error envelopes enlarge the
checked motion but do not prove that a live controller stays within those bounds.
Unresolved live helpers, grasp/release preparation, named poses, changing commanded
tool orientation and unsupported interpolation remain `NEEDS_CONTEXT`.
The common [resource_safety_preparation.py](../../cais_spade_llm/resources/resource_safety_preparation.py)
holds registration, pose/timing validation, model descriptors and evidence helpers.
The old `ur5e_safety_preparation.py` module is removed; controller-dependent
operations live in the existing controller. `RobotAgent` delegates under its motion
lock; fixed equipment needs no joints or gripper fields. `ResourceAgent` defaults
remain `NEEDS_CONTEXT`.

Optional keyword-only `primitive_models` flows through observation, reviewed
checking, automatic grounding and composition. Trusted owner callbacks validate
native parameters and supply complete fixed-orientation body/part paths, timed
containment transfers and native state effects. They return no APs or verdicts.
Containment and gripper custody remain distinct; both transfer participants and
inventories must agree. New owner-modeled fixed equipment can work while its body
remains stationary, preserving legacy `stationary_only` behavior for existing callers.
Declared task effects, guards, exact supported programs and acknowledgement evidence
justify process/state changes; `dwell`, motion and task names cannot establish a
completion by themselves. Model identity/version/configuration and resulting effects
are fingerprinted with the trace; changed contracts invalidate continuation.
These effects remain detached projections and do not advance actual history.
The KMR fixtures and legacy `receiving_region_entry` meanings remain unchanged.

The full milestone's reviewed rule catalog must cover machine access/interlocks,
exclusive M1/M2 `workholding` access through front access and `side_access`, staging/buffer
capacity, Conveyor handoffs, KMR docking/arm/base conditions, assembly-region
exclusion, persistent occupancy, and recovery/resumption obligations. Bind from
resource declarations and observations, never from the expected recovery answer.

Continue preserving event/state APs, predicted successor facts, empty trace steps, and
finite-trace completion semantics when integrating recovery. Nonaccepting prefixes with a valid accepting
continuation remain pending obligations. Missing required evidence or unsupported
rules block admission with an explicit reason. Version the templates and record
AP/rule coverage with each bundle.

**Acceptance:** the receiving-region counterexample is resolved with regression
coverage; identical inputs produce identical safety artifacts; all required rules
have deterministic applicability and grounded coverage; and authoritative safety
construction makes no LLM calls. Connect the validated primitive safety evidence
to the existing local composition backend before claiming runtime integration.
Treat construction reproducibility separately from the completeness of the
reviewed hazard model. Capacity checking, arbitrary physical predicates, changing
orientations, controller feasibility, and complete live evidence remain deferred.
The opt-in 3D slice is conditional on complete, accurate configured envelopes and
trajectories; it does not establish physical feasibility or collision-free motion.

## Milestone 3: Local parallel composition

### Implemented nominal execution

Local parallel composition is implemented and connected to nominal execution
through [EnvironmentRuntime](../../cais_spade_llm/recovery_framework/environment_runtime.py).
PA uses `set_plans(..., compile_fsa=False)` and submits
`composition_backend="local_parallel"`. CCA checks candidates through
`EnvironmentAdmission.check(..., commit=False)` and performs atomic admission
with `commit=True` when RA requests permission before execution.

The [local composition backend](../../cais_spade_llm/agents/central_controller/local_composition.py),
[EnvironmentPlant](../../cais_spade_llm/recovery_framework/environment_composition.py),
and [EnvironmentAdmission](../../cais_spade_llm/recovery_framework/environment_admission.py)
implement dependency closure, running-task and joint completion analysis,
composition caching, monitor history, snapshot validation, and atomic grants.
Existing regression coverage exercises these behaviors. The 2026-10-03 review
did not complete fresh local-composition regression runs, so no new pass result
is attributed to that review. Fresh 2026-10-04 compatibility results are recorded
under Verification and handoff below.

### Remaining generated recovery integration and acceptance

The pure offline
[`analyze_grounded_recovery_composition`](../../cais_spade_llm/agents/central_controller/offline_recovery_composition.py)
now reuses `Scope`, `Analysis`, `Budget`, and `nonblocking_region`. It accepts frozen
grounding inputs, exact recovery event/primitive references, included running work,
finite fully supplied timing alternatives, completion conditions, and an optional
replayable prefix. It builds reachable shared prefixes directly, without compiling
a global FSA. All configured resources and applicable bindings remain in scope;
there is no exclusion pruning or scaling claim.

Predefined physical rules are evaluated on the grounded observation path, with
CCA-owned source bindings; the native task monitor does not interpret physical
descriptors. Interacting nominal starts also require a complete owner-supplied
joint proof. Missing nominal or recovery evidence blocks before a grant commits.
A trusted owner can register finite nominal behavior through `nominal_request`;
there is no default live trajectory provider. Plan/bundle native validation alone
cannot certify physical rules. `SystemBridge` and its public interface are unchanged.

Optional acknowledged product-effect evidence contributes timestamps to the joint
clock. Supported declared task effects may append exact process records; the
existing `place_insert` assembly record and synthetic `M1` `machine_part` trim
record use this path while preserving other process records and custody.
Owner-provided body/part trajectories, joint transfer boundaries and native state
effects join the same branch search through `primitive_models`. Registration adds
neither an AP meaning nor permission. Branches may project
these effects, but actual history requires validated acknowledgements and exact
observed effects. Task and physical monitors retain distinct histories within one
shared continuation. Unsupported effects or incompatible histories remain unavailable.

Whole recovery-event starts are controllable. Internal observations, primitive
progress, and completion are uncontrollable; unsafe forced successors remain in
the graph. Waiting is possible only where an explicit alternative supplies its
timing and complete stationary evidence. Every alternative uses fixed trajectories
and durations. Different mutex instances consume their own `rule_id` valuations,
even though their AP labels remain `ap001` and `ap002`.

The separate
[composition companion](../../test/fixtures/KMR_assembly_board-v1_recovery/storage_interruption/composition_evidence.json)
uses `[0, 26]`: immediate placement starts the four events at `0, 5, 10, 14`;
delayed placement uses `0, 5, 12, 16`. Included `ur5e-3` work withdraws by `12`.
The composition observes simultaneous effects jointly, consumes each observation
once, and adds no empty terminal observation. Replay retains pending obligations
and preceding observations within the same fingerprinted problem. `X` is unavailable
in this path until its branching clock is defined; standalone reviewed checking
continues to support it.

Completion requires the restored Storage checkpoint, completed included running
work, and accepting selected DFAs. Missing behavior needed to discharge a pending
`F` or `U` requirement is inconclusive. `KET8_Square_8mm` custody,
deposited `KET4_Square_4mm`, both process ledgers, and all three pending M1 delivery
task IDs survive. Restoring this checkpoint does not complete delivery. These
results establish bounded offline recovery composition, conditional on the supplied
evidence; they do not grant runtime admission or establish physical feasibility.

The inspected `ProductRecoveryController._dispatch_runtime_plan_validation_check`
in [product_recovery_controller.py](../../cais_spade_llm/agents/intelligent_product/product_recovery_controller.py)
still calls `compile_global_fsa()` after inserting recovery macros. The nominal
`EnvironmentPlant` resolves and enumerates declared resource-model events.
The new
[`RecoveryCompositionAdmission`](../../cais_spade_llm/agents/central_controller/recovery_composition_admission.py)
registers the complete sequence during `plan_safety_check`, binds exact dispatch
identities and primitive programs, and grants only the specific next winning start.
CCA recognizes registered generated tasks before nominal `EnvironmentRuntime`
matching. PA carries `recovery_composition_ref`; only the CCA-owned grant permits
RA's macro executor to proceed. An overall `allowed` analysis does not authorize
an unsafe immediate start. Full-resource proofs hold additional unmodeled starts.

The same branch search retains the main and active recovery-scope native monitors
alongside the grounded physical monitors, with separate states and clocks. Existing
scope routing supplies exact native event eligibility; out-of-scope completions add
no DFA tick. Pending obligations, replayed physical checkpoints, simultaneous
completion-order alternatives, and incompatible histories remain explicit. Only
validated acknowledgements and observations advance retained histories; predicted
endpoints cannot do so. Identity, state, revision, timing, and program changes are
rechecked under the admission lock. Re-registration and duplicate requests cannot
reset monitor history or execute a consumed task twice.

The new [Prepare and check adapter](../../cais_spade_llm/recovery_framework/gazebo_safety_preparation.py)
inspects a detached CCA snapshot through the same pure composition entrypoint.
It captures all configured owners, running work, revisions, ledgers, reservations
and monitor states; stale capture windows and changed programs or histories are
unavailable. It does not register an admission session, issue `allow`, synchronize
actual monitor history, or activate the selected specifications. Live preparation
acceptance remains a prerequisite for a later guarded-execution connection.
The CCA-owned `recovery_safety_preparation_provider` is now registered from trusted
`scene.safety_preparation` configuration. The short `GAZEBO_MOTION_SAFE` and
`GAZEBO_MOTION_CONFLICT` inputs retain their mock origin and native event/step
references; observations and planner outputs remain separate. Every saved report
has `dispatch_authorized=False`. The idle slice retains the initialized ledger,
rejects unprovided history and checks resource/controller activity before planning.

For `continuous_physical_boundaries_v1`, each uncertain interval represents possible
finite AP words. Numerical refinement points are not semantic clock ticks. DFA
state sets and uncontrollable unsafe alternatives remain in the same branch search;
only a continuation safe across all retained alternatives may be `allowed`.
Definite conflicts produce `held`; unresolved bounds or budget exhaustion produce
`inconclusive`. Rule-local AP labels, native monitor history and pending obligations
remain separate and require a common completion witness. `X` is still unsupported.
The existing deterministic 1D and piecewise-linear paths retain their clocks.

The resource-owned `prepare_recovery_composition_evidence` hook defaults to
`NEEDS_CONTEXT`. KMR exports retained physical pose and custody fields, but complete
live motion, timing, helper-output, and observation contracts are not supplied by
that export. Synthetic providers are confined to tests with explicitly enabled mock
executors. Registered executor overrides without the required evidence contract
remain blocked. See the [admission contract](../../docs/recovery_composition_admission.md)
for owner interfaces, supported clocks, execution feedback, and conditional claims.
CCA now resolves pure primitive models from registered owners, passes those models
to the same composition search, and fingerprints their descriptors. Executable
callbacks cannot arrive in serialized context; changed contracts invalidate a
registered proof. `RobotAgent` can delegate a registered macro to the common
grant-consuming executor only when its owner and initialized controller both
declare prepared execution support. As of 2026-10-08, the native controller and
owner adapters implement exact prepared command ownership and dispatch plumbing,
but do not provide justified tracking, stationary containment or stopping bounds;
physical execution coverage therefore still blocks live dispatch. Continuous observed-history
replay across recovery boundaries remains required; exact synthetic graph states
cannot stand in for measured Gazebo states.
Individual permission messages cannot authorize an atomic multi-resource start
edge; those choices remain inconclusive. Active-rule Gazebo execution and observed
recovery/resumption remain separate acceptance gates.

Keep CCA as the decision authority. Extend the existing direct factored-model
analysis; do not build a global FSA and then slice it. Determine scope from shared
events, resource participation, guard/update dependencies, product custody,
task prerequisites, and safety APs. Expand scope when dependencies cross its
boundary; omitted resource behavior cannot be assumed safe or fixed without
validated evidence.

Select dependencies through the primitive AP bindings as well as declared events,
and verify which generated-event boundaries are actually controllable. A controller
that cannot stop inside a primitive cannot use every observation as a dispatch
decision. Reconcile the frozen primitive observation clock with concurrent nominal
behavior explicitly, especially for `X`; do not drop intermediate observations.

Preserve shared-event synchronization, atomic custody, running tasks, persistent
occupancy, and pending obligations. Check immediate safety and joint safe
completion. Separate successful rule checks do not establish a compatible joint
continuation. Preserve the event clock and empty/end-step semantics; dropping
an event is not automatically equivalent to stuttering.

Cache by model, rule, relevant-state, and scope fingerprints and invalidate on
change. Current local `Budget` defaults are **20,000 states / 2 seconds**;
`EnvironmentAdmission` uses the run's `composition_budget` when supplied.
The earlier roadmap proposed **200,000 states / 10 seconds** for local validation;
those are not the current local defaults. This update does not change runtime
configuration. Exhaustion is inconclusive and holds affected dispatch, not proof
of safety or infeasibility. Strong coupling may still require a large product.
Completion and failure observations are consumed; only controllable starts are
gated. Recheck the snapshot and reservations before committing concurrent starts.

**Interfaces:** retain `SystemBridge` and existing PA/RA/CCA protocol identities.
Reuse the internal factored-plan backend and retain result evidence for checked
scope, dependencies, fingerprints, explored states, timing, and inconclusive reasons.
Keep global and existing active-window backends as comparisons. Do not equate a
local accepting continuation with a general global nonblocking guarantee.

**Acceptance:** extend local/global small-model agreement to generated recovery
behavior for safety and completion; detect coupled conflicts; and avoid enlarging
products unnecessarily for unrelated resource growth. Test open obligations across
scope changes, concurrent starts, shared handoffs, and stale cached results.
Retain active-rule nominal Gazebo execution and observed recovery/resumption as
separate acceptance gates, with CCA bypass disabled. Implemented nominal local
composition does not establish those execution outcomes.

For `storage_interruption`, preserve both parts' custody and `processCompleted`,
running work, all safety-derived obligations, and the original pending M1 delivery
task IDs. Verify a compatible continuation from the restored carrying checkpoint
and later observed delivery; recovery completion alone must not complete those tasks.

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
| **Part slippage:** `ur5e-4` drops `gear_small` into `ur5e-3`'s region. `KET4_Square_4mm` in `Buffer For Machined parts` is the desired starting condition, not an observed checkpoint. | Prepare recovery using the observed custody of `ur5e-3`, preserve its pending work, and recover `gear_small` under the separate mutex and precedence specifications. | Unreachable part, unavailable staging, unsuitable part condition. |

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
| ur5e-4 part slippage | `inconclusive` | Observe `gear_small` in `ur5e-3`'s region and the desired `KET4_Square_4mm` buffer condition; retain both resources' pending work. Legacy LG/MCP fixtures and earlier opposite-direction experiments remain historical evidence. |

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

Use `KET4_Square_4mm` for initial machining/transport. For the planned slippage
case, observe `ur5e-4` holding `gear_small` before a drop into `ur5e-3`'s region;
`KET4_Square_4mm` in `Buffer For Machined parts` is the desired starting condition
and has not been established by preparation tests. Retain earlier slippage
experiments as historical evidence. Initial blocked conditions are unavailable
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
are separate from the configurable local runtime validation budget, whose current
default is 2 seconds. Milestone 3 retains the earlier 10-second proposal separately.

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

The historical planning baseline retained in the 2026-09-29 roadmap passed
36 feasibility/DFA and 14 diagnostics/UI checks. Verification on **2026-09-29**
recorded **244 focused tests** for the implemented validation/inspection slice,
plus the final artifact-label checks, Poetry, compilation, and CLI checks;
[verification details](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#verification-and-claim-limits)
are recorded in the formal reference.
These historical results are software validation; no live Gazebo or hardware
recovery was performed in that verification.

The **2026-10-03** source review ran
`poetry run python -m pytest -q test/test_primitive_program_safety.py`:
**98 passed in 2.37 seconds**, using offline modeled evidence. A separate
reproduction exposed the receiving-region counterexample recorded in milestone 2.
The `test/test_local_composition.py` rerun and selected admission/composition cases
from `test/test_environment_capabilities.py` were interrupted; no fresh pass result
is claimed for them. No live Gazebo or hardware validation was performed in that
review. The two offline safety verification slices immediately below did not
rerun local composition.

Verification on **2026-10-04** ran
`poetry run python -m pytest -q test/test_primitive_program_safety.py`:
**186 passed in 6.95 seconds**, including the retained 98 legacy tests and 88 new
reviewed-checker cases. Sixteen formula cases include **5,440 comparisons** against
an independent finite-trace evaluator. The synthetic baseline produces **107 joint
observations** for **22 primitives** and satisfies all **10 mutex pairs plus KET4
trim precedence** at explicit trace completion. Negative cases cover concurrent
and internal conflicts, simultaneous contact, part-specific entry, missing trim,
incomplete evidence, unsupported effects, and invalid continuation. All 22 primitive
endpoints agree with the original saved physical expectations; M1 delivery remains
pending. Passing does not establish a feasible joint continuation or physical execution.

`poetry check` passed with existing metadata deprecation warnings; compilation of
`cais_spade_llm` and `ros2`, focused Ruff checks, JSON/local-reference checks, and
`git diff --check` passed. Protected-artifact hashes and the unrelated tracked diff
were preserved. The
[evidence contract](../../cais_spade_llm/specification/safety/primitive_observation_safety.md#conditional-soundness-and-evidence-limits)
states the clock-specific conditional soundness argument and model assumptions.
No LLM, runtime admission, local-composition integration, Gazebo execution, or
nominal resumption was performed in this verification.

Subsequent verification on **2026-10-04** for automatic offline grounding ran
the same safety suite: **259 passed in 26.23 seconds**, retaining the preceding
186 cases and adding 73 regressions. The unchanged 22-primitive behavior produces
107 observations for **66 mutex instances plus KET4 trim precedence**, now covering
every configured resource. Adding `ur5e-5` produces **78 mutex instances** and
detects its conflict without changing the requirement catalog or scopes. Tests
also cover fixed-resource conflicts and missing coverage, unchanged physical APs
under new recovery event identities, preserved `ap002` / `ap008` task/state
descriptors, complete ledgers, exact Boolean/string state semantics, joint
completion boundaries, `X`, pending requirements, and rejected continuation.

**53 relevant nominal declaration/projection tests passed**, including 10 new
resource/handling regressions. A broader nominal/UI run was interrupted in
unrelated UI tests and is not reported as passing. No local-composition rerun
was performed. `poetry check`, compilation, focused Ruff checks on changed source
and the safety suite, JSON/source-reference checks, and `git diff --check` passed.
Poetry retains its metadata deprecation warnings; the nominal test file's nine
pre-existing Ruff findings were verified unchanged. Original recovery JSONs,
Safety-page artifacts and selection, scene configuration, historical evidence,
and unrelated tracked changes remain preserved. All added physical/task/state
evidence is offline and synthetic; these results establish neither dispatch nor
actual recovery/resumption.

Subsequent verification on **2026-10-04** for bounded offline recovery composition
ran `poetry run python -m pytest -q test/test_offline_recovery_composition.py test/test_primitive_program_safety.py`:
**305 passed in 66.48 seconds**, including **46 new composition cases** and all
**259 existing safety cases**. Four formula cases compare every supplied timing
alternative with an independent finite-word evaluator. Tests cover the shared
initial starts, held immediate placement, explicitly permitted waiting, unavoidable
internal conflicts, per-`rule_id` AP values, pending `F`/`U`, exact observation
consumption, same-problem prefix replay, unavailable `X`, and budget exhaustion.
They also reject choice-dependent trajectories or unrelated task/state effects,
missing coverage, inconsistent custody, and completion claims based on copied
`current_state` or `resource_location` metadata. Distinct `des_event_id` and
`outline_id` symbols remain distinct.

All **13 existing nominal local-composition tests passed** in separate runs:
**12 passed in 1.87 seconds** in the restricted environment; the asynchronous
arrival/admission test hung there with an idle worker and a waiting event loop,
then **1 passed in 1.57 seconds** outside that sandbox. The restricted attempts
were interrupted and are not counted as successful runs. The historical
2026-10-03 interrupted results remain unchanged.

`poetry check` passed with its existing metadata deprecation warnings. Compilation
of `cais_spade_llm` and `ros2`, focused Ruff checks, JSON/source-hash checks,
local-link checks, and `git diff --check` passed. All **26 roadmap headings** and
unrelated tracked changes are preserved. Protected fixture JSONs, initialization,
Safety-page artifacts, and nominal runtime source retain their prior hashes;
only the authorized companion documentation changed among the protected files.

The synthetic immediate schedule's first mutex conflict is at
`time=11.407407407407407`; the supplied delayed schedule provides a completed
offline witness for the restored Storage checkpoint and included `ur5e-3`
withdrawal. All 22 KMR primitives retain their parameters and provenance. The
three original M1 delivery tasks remain pending. These results establish bounded
offline composition under supplied geometry, trajectories, start alternatives,
and evidence, with unchanged **20,000 states / 2 seconds** defaults. No LLM,
runtime admission, carrying-base extension, Gazebo execution, or M1 delivery
execution was performed.

Verification on **2026-10-05** for generated recovery admission ran the primitive
safety, offline composition, coordinator, PA → RA → CCA admission, nominal
acknowledgement, nominal local-composition, recovery feasibility, and recovery
delivery suites together: **566 passed, 8 deselected in 138.43 seconds**. The eight
deselected cases are the existing
`test_complete_run_page_uses_existing_start_stop_without_render_dispatch` UI group.
A preceding broader feasibility/delivery run recorded **178 passed and 5 failed**;
all five failures raised NiceGUI's `RuntimeError: Request is not set` at the
unchanged `ui/recovery_run.py:777`. These UI failures remain unresolved and are not
counted as passing. Restricted asynchronous runs that stalled were interrupted;
the reported combined run completed outside that sandbox with mock executors.

The combined run includes all **259 primitive safety cases**, **69 offline
composition cases**, and **13 nominal local-composition cases**. Tests establish
one shared native/physical completion witness, retained pending obligations,
exact scope clocks, replayed physical history, and rejection of incompatible
checkpoints. Admission tests cover the specific requested start, stale evidence,
program changes, unavailable live preparation, synthetic evidence on the live
route, duplicate requests and acknowledgements, and unmodeled concurrent starts.
The real nominal acknowledgement path contributes one committed main-monitor tick;
the recovery coordinator verifies and reuses its audit instead of consuming it twice.

The complete mock message-path test invokes all **22 KMR primitives** through
PA registration, RA permission requests, CCA decisions, and validated feedback.
Placement at time 10 is blocked without another primitive call; the supplied start
at time 12 receives permission after the required observed progress. The final
Storage checkpoint retains `KET8_Square_8mm` custody, deposited `KET4_Square_4mm`,
both process ledgers, and all three pending M1 delivery tasks. Fresh permission
request identities reject delayed replies, and CCA serializes complete-resource
proofs across products. Global plan FSA start conditions remain an additional gate.

After a final coordinator readability cleanup, **39 coordinator/nominal
acknowledgement tests passed in 30.07 seconds**, and **6 PA → RA → CCA integration
tests passed in 7.39 seconds**. `poetry check` passed with the existing metadata
deprecation warnings. Compilation of `cais_spade_llm` and `ros2`, focused Ruff
checks on the new composition/admission modules and affected test suites,
JSON/local-reference checks, and `git diff --check` passed. All **46 protected
files** retain their starting hashes, including original fixtures, Safety-page
artifacts/selection, `SystemBridge`, and unrelated changes; all **26 roadmap
headings** remain unchanged. Existing complexity findings in large agent/admission
handlers are outside the focused Ruff result.

These results establish the actual blocking gate with mock execution. The default
resource preparation hook returns `NEEDS_CONTEXT`; complete live motion, timing,
custody, and execution-observation support remain unavailable. No LLM call, Gazebo
trial, hardware trial, carrying-base extension, or M1 delivery execution was
performed. Active-rule Gazebo recovery and observed nominal resumption remain
separate acceptance gates. The default budget remains **20,000 states / 2 seconds**.

Nominal local parallel composition and bounded offline generated recovery composition
are implemented, with generated-event admission connected to the `validated` route
and verified using mock executors. The legacy two-specification
checker retains the recorded counterexample. The automatic grounding path establishes
reusable offline checks across the complete frozen configured population for supported
AP meanings and supplied reviewed scopes. Complete reviewed hazard coverage, trusted
live physical/task/state evidence, complete resource preparation and observation
contracts, active-rule Gazebo execution, observed recovery/resumption, and
confirmatory experiments remain separate acceptance gates.

Verification on **2026-10-05** for predefined specifications ran the compiler,
Safety-page preview, primitive grounding, offline composition, recovery coordinator,
nominal acknowledgement, PA → RA → CCA admission, predefined admission, and nominal
local-composition suites together: **520 passed in 150.38 seconds**. This includes
**290 primitive safety cases**, **98 composition cases**, **52 compiler/preview cases**,
and **13 nominal local-composition cases**. After adding the guard against restarting
physical history through another ProductAgent, **35 related admission tests passed
in 16.79 seconds**. The final startup-metadata/history checks passed **2 cases**.
Restricted asynchronous attempts stalled with an idle worker and waiting event
loop and were interrupted or timed out; the combined and related admission runs
completed outside that sandbox using mock executors.

The tests establish exact predefined formula/AP preservation with zero safety
LLM calls, deterministic compilation and fail-closed loading, all 66 configured
mutex instances plus the separate precedence instance, and a common native/physical
continuation. Prior gear_small completion permits KET4 entry; entry before or at
first completion fails. Wrong target, part, unacknowledged effects, changed source,
missing participant evidence, and incompatible history cannot authorize a start.
Nominal mock starts use the same coordinator, exact prepared task/program matching,
atomic grants, duplicate handling, and the existing acknowledgement audit. Actual
product effects must match declared effects and validated acknowledgements;
predictions never commit completion history. Native-only bundle validation explicitly
reports unavailable physical grounding instead of a safety pass or safety-free replan.

`poetry check` passed with existing metadata deprecation warnings. Compilation of
`cais_spade_llm` and `ros2`, focused Ruff, JSON/geometry-reference checks, local-link
checks, and `git diff --check` passed. All **206 protected files** retain their
starting hashes, including `SystemBridge`, selection, approvals, all saved preview
artifacts, and original recovery fixtures. All **26 roadmap headings** remain.
The new predefined document is selectable but has not been approved or activated.
No safety-generation LLM call, browser trial, Gazebo execution, new live trajectory
provider, carrying-base extension, or M1 delivery execution was performed. The
legacy `receiving_region_entry` mismatch and historical interrupted results remain.

Verification on **2026-10-05** for UR5e Prepare and check ran the new preparation
tests together with predefined compilation, previews, primitive grounding, offline
composition, recovery admission and nominal composition: **551 passed in 156.57s**.
After final trajectory-header, provenance, unresolved route-context, explicit-later-start,
and concurrent execution-feedback checks, the dedicated preparation suite passed
**33 tests in 6.13s**. These use mock ROS services
and registered test owners. They establish no-dispatch preparation, six-joint and
nanosecond preservation, all 12 resources / 66 mutex pairs, independent AP labels,
mutex conflicts, strict gear_small-before-KET4 precedence, missing/stale/changed
evidence rejection, explicit later starts, and unchanged histories/ledgers/pending
work. The NiceGUI component requires an explicit click; no browser or live execution
claim is made. The first sandbox UI run stalled in the asyncio selector; the
mock-only runs completed outside that restriction.

The broader controller suite reported **250 passed / 6 failed**. All six failures
were reproduced using the unchanged `HEAD` Cartesian execution method: five expect
collision checking to be disabled despite the current configured `true`, and one
requires the absent historical `recorded_two_part_validation/saved_waypoints_scene.json`.
The combined preparation/UI-setup run reported **91 passed / 10 failed**; the ten
failures occur in the untouched Run-page `client.ip` path with a test client lacking
an HTTP request (`RuntimeError: Request is not set`). These broader suites are not
claimed clean, and their fixtures or runtime settings were not changed to obtain
a pass.

`poetry check` and compilation of `cais_spade_llm` and `ros2` passed; Poetry retained
its existing metadata warnings. Focused Ruff, JSON/reference checks, local links,
and `git diff --check` passed. The 48-file preservation capture for this slice and
the earlier **206-file protected set** remain unchanged, including `SystemBridge`,
selection, approvals, previews and original fixtures. All **26 headings** remain.
No local `gzserver` was running, so no live capture/planning result was collected.
The current scene has no complete `safety_geometry` contract; fresh attachment
observations, full-resource Cartesian/envelope coverage and stationary/running-work
coverage also remain blocking gaps. No proof activation, grant, LLM call, failure
injection, Gazebo motion, or recovery/resumption trial was performed.

Verification on **2026-10-05** for extensible resource-owned preparation ran the
combined preparation, predefined safety, primitive grounding, offline composition,
CCA admission and nominal local-composition suites: **579 passed in 164.03s**.
Final affected product/native-history checks passed **111 tests** (53 deselected),
and registration/model checks passed **51 tests in 9.11s**. After preserving the
historical UR5e evidence contract, the complete owner/preparation suites passed
**81 tests in 11.62s**, including the explicit NiceGUI action. These counts are
overlapping runs, not additional independent trials. Existing program-contract
tests passed **21 tests in 2.43s**; focused environment capability/composition tests
passed **14 tests** (123 deselected).

The new tests establish a registered additional resource and its declared action,
78 mutex pairs, physical conflict detection without catalog changes, a seven-joint
mock through the same preparation adapter, and fixed-equipment evidence for `M1`
`machine_part` / `dwell`, `Conveyor` `advance_conveyor` / `move_relative`, and
`Buffer For Machined parts` `advance_part` / `move_relative`. They check matching
containment transfers, exact inventories and native state effects, declared trim
completion, pending `F` satisfaction in one native/physical composition witness,
parameterized `inventory.{part_name}` declarations, immutable actual histories,
and incompatible contract/history rejection. Compatibility fixes retain the
historical `assembly_board-v1` containment destination and disallow unsupported
legacy native completion fields. No command, grant, active proof, or live resource
provider was introduced.

The broader program/Resources-page run reported **111 passed / 2 failed** before
stopping. Both failures were reproduced with the pre-change environment-model
and program modules: `ReadOnlyBridge` lacks `get_conveyor_fault` in the untouched
Resources-page status reader. That suite is not claimed clean. Broader environment
inbox runs were interrupted without a completed result; no whole-suite pass is
claimed. Restricted asyncio runs also stalled; the focused mock-only acceptance
runs completed outside that restriction.

`poetry check` passed with the existing metadata warnings. Compilation of
`cais_spade_llm` and `ros2`, focused evidence/preparation Ruff, fixture JSON and
local-reference checks, changed-file whitespace checks, and `git diff --check`
passed. The **58-file protected capture** is unchanged, including `SystemBridge`,
Safety-page artifacts, initialization files and original fixtures. All **26 roadmap
headings** and earlier dated results remain. This establishes reusable preparation
and conditional offline checking. Complete live observation and motion contracts,
guarded execution, failure injection, and recovery/resumption remain later gates;
the legacy `receiving_region_entry` mismatch, composition's restriction on `X`,
and **20,000 states / 2 seconds** defaults are unchanged.


Verification on **2026-10-05** for supplied recovery motion ran the continuous-motion,
preparation/UI, resource-owner, primitive safety, predefined safety, offline recovery
composition, CCA admission and nominal local-composition suites: **620 passed in
174.27s**. These include between-point conflicts, outward-bound uncertainty, pending
`F`/`U`, rejected `X`, preserved event provenance and AP labels, unsupported evidence,
unchanged custody and histories, and no-dispatch preparation. Existing before/after/
simultaneous-completion precedence cases remain in the passing regressions. Counts
from earlier runs remain historical and overlap this run.

`make bootstrap-gazebo` built **16 packages** after adding `/GETRECOVERYSTATE` to
the attachment owner. The first build exposed a missing collision header; the first
live read exposed infinite ground-plane bounds in JSON. Both were corrected, and
installed observations subsequently reported **36 models and zero attachments**.
The dedicated headless session explicitly selected the predefined document and used
real registered owners and CCA without starting PA/RA dispatch behaviors. Both
supplied candidates saved **12-resource** checkpoints and returned `NEEDS_CONTEXT`:
UR5e link motion exceeded the declared idle limits, controller action servers had
not supplied idle-status records, and the `assembly_board-v1` carrier had no finite
collision envelope. Removing a required owner declaration also returned
`NEEDS_CONTEXT`. A later retry timed out on the read-only service. See the
[recorded evidence index](../../docs/gazebo_safety_preparation_2026-10-05.json)
for the immutable report paths and hashes.

**Live acceptance remains incomplete:** no prepared moving candidate reached a
live `allowed` result or a definite live mutex counterexample. No recovery
primitives, controller motion, attachment mutations, entity repositioning, failure
injection, grants or LLM calls were dispatched by Prepare and check. Precedence
has no new non-vacuous live result; empty-custody motion with no
`KET4_Square_4mm` entry would only check the no-entry case. Guarded execution and
recovery/resumption remain subsequent milestones. Do not substitute an offline
pass or a captured checkpoint for either missing live acceptance result.

Final affected continuous-motion, owner/preparation and UI checks passed **125 tests
in 16.53s** after adding stationary-component coverage, explicit quaternion validation,
owner hold-contract rejection and controller-configuration revalidation. This run
overlaps the 620-test run. `poetry check` passed with existing metadata warnings;
compilation of `cais_spade_llm`, `ros2` and the test runner, CLI help, focused Ruff,
JSON/report-fingerprint and local-link checks, and `git diff --check` passed.
The **57 existing protected files** checked against this turn's initial capture
are unchanged, including `SystemBridge`, Safety-page files, saved safety artifacts,
`cca.json` and original fixtures. All **26 roadmap headings** remain. The scene's
new preparation configuration changes no runtime budget or safety selection.

Verification on **2026-10-05** for the focused mock `part_slippage` experiment:
the existing primitive safety and predefined compiler suites passed **338 tests
in 39.48s**, including **17 new fixture/report regressions**. The standalone
runner checked all three complete traces against the same supplied document.
`PART_SLIPPAGE_MUTEX` violates only mutex at **349/66 s**;
`PART_SLIPPAGE_PRECEDENCE` violates only precedence at **415/66 s**;
`PART_SLIPPAGE_SAFE` satisfies both, with synthetic gear assembly completion at
**9 s** preceding actual modeled KET4 entry. The report retains the exact
counterexample events, primitive indices, AP bindings, and DFA transitions.
Each candidate has seven events and 19 primitives with complete 12-resource
evidence. Renaming every recovery event preserves physical AP values and
verdicts; missing evidence and forged completion cannot pass. Removing the
separate synthetic acknowledgement causes the previously safe trace to violate
precedence. Both parts are deposited, both robots end empty, KET4's trim ledger
is preserved, and its assembly task remains pending.

`poetry check` passed with the existing metadata warnings; `compileall` for
`cais_spade_llm`, `ros2`, and the runner, CLI help, and focused runner Ruff passed.
JSON, source/report hashes, local links, and `git diff --check` passed. Comparison
with the initial file capture confirms that existing runtime code, Safety-page
artifacts, and historical fixtures remain unchanged; only the existing safety
test file and this roadmap were extended, alongside the new demo files.
The evidence is explicitly synthetic and uses the existing offline primitive
observation contract, not native UR5e dispatch programs. This experiment calls
the pure grounded checker; it does not add an admission grant or a new
composition result. No LLM, Gazebo, or primitive execution was invoked. The
earlier live-preparation gaps, historical results, and legacy
`receiving_region_entry` counterexample remain unchanged.

Verification on **2026-10-05** for the requested three-case Gazebo recording:
the [live report](../../cais_spade_llm/monitor/recovery_gazebo_runs/part-slippage-20261006T021725Z/REPORT.md)
retains the separate mutex, precedence and safe prerequisite captures. The ROS
bootstrap completed **16 packages**; the native controllers reported ready and
the attachment owner returned **36 models with zero attachments**. Each final
capture includes all **12 configured resources**, 12 resource envelopes and
12 part envelopes. Explicit static constituent geometry resolves the empty
`assembly_board_v1` carrier envelope without guessing dimensions. Fixed equipment's
configured `static_body_and_idle_containment` contract now requires observed
static and idle evidence without robot fields.

All three attempts are **`NEEDS_CONTEXT`**: observed link motion and missing
controller goal-status records prevent a usable idle checkpoint; the first final
capture also exceeded the freshness window. CCA and the recovery resources have
no registered live execution-evidence providers. Continuous grasp/release,
carried-part effects, bound prepared execution and validated feedback are still
required. The new companions preserve seven events and 19 primitive references
per case, but contain no resolved native programs or prepared trajectories.
They never substitute the original synthetic parameters or one-second durations.
No post-slippage checkpoint was staged, no PA → RA → CCA execution trial occurred,
and no mutex/precedence counterexample or safe recovery was established. No
requested MP4 was produced. The owned Gazebo session was stopped after capture.

**611 tests passed**: 65 preparation checks, the isolated UI check, 504 safety/
composition/admission checks and 41 recording checks. The first preparation
suite was interrupted with 63 checks passed while its UI test hung; that test
also timed out alone in the restricted sandbox, then passed outside it in 1.51s.
Those partial/repeated runs are not added to the total. `poetry check` passed
with existing metadata warnings; compilation, both CLI help checks, focused
Ruff, JSON/report fingerprints, local links and `git diff --check` passed.
All 26 roadmap headings remain. Safety-page selection/approvals, `SystemBridge`,
`cca.json`, original fixtures, historical recordings and unrelated working-tree
changes are preserved. The legacy `receiving_region_entry` mismatch, `X`
restriction and **20,000 states / 2 seconds** defaults remain unchanged.

The **2026-10-05 live-connection follow-up** changes the requested output to one
silent, captioned 20× video containing the three separately staged trials, with
both fixed specifications active together. The combined exporter passed complete
decoding, timing and source-retention tests using synthetic encoding frames.
No requested live video exists yet. The [follow-up report](../../cais_spade_llm/monitor/recovery_gazebo_runs/part-slippage-live-20261006T030859Z/REPORT.md)
retains actual read-only Gazebo observations: all 11 controller rows covering
17 action endpoints, a complete physics snapshot, and empty observed attachments.
The configured population remains 12 resources. The successful read window was
0.234 seconds after cold-discovery failures; a diagnostic 15-second service timeout
permitted discovery without changing the observation freshness requirement.
The dedicated launch disabled Nav2 and enabled the opt-in controller reader;
default launch/controller behavior is preserved.

Trusted pure models now reach CCA's composition search without serialized callbacks,
and changed descriptors invalidate registration. Synthetic continuous custody
effects preserve part geometry, release/reacquisition ordering and process ledgers;
explicit joint-error envelopes widen motion bounds. Registered RobotAgent execution
requires both owner preparation/feedback and initialized controller support. That
live execution contract and compatible continuous observed-history replay are still
missing. No staged checkpoint, live safety rejection, recovery execution or
observed resumption is claimed, and no missing-evidence capture is published as a
successful trial.

Final selected suites passed **954 tests**: **212** preparation/owner/admission,
**478** safety/composition/nominal-admission, **221** Gazebo launch/resource, and
**43** recording tests. Earlier overlapping runs and corrected fixture/default
failures are excluded from that total. `make bootstrap-gazebo` completed **16
packages**. `poetry check` passed with existing metadata warnings; compilation,
UI/preflight CLI checks, focused evidence/admission Ruff, JSON/reference checks,
local links and `git diff --check` passed. The recorder module retains pre-existing
lint findings outside the new exporter. All **26 headings** and **77 protected
files** are preserved. The unchanged specifications, approvals, fixtures, runtime
authorities, legacy counterexample, `X` restriction and budget defaults remain.


**2026-10-08 consolidation and grounded CCA implementation:** the
[retained consolidation verification](../../cais_spade_llm/monitor/recovery_gazebo_runs/cca-consolidation-20261008T033602Z/verification.json)
records the shared coordinator, removal of five superseded production `live_*.py`
modules, exact nominal grant/command fixes and checks performed. Test groups in
that report overlap and must not be summed. Three pre-existing UI fixture failures
remain explicitly reported. Its background Gazebo run captured 11 controllers and
12 resources with `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm` active and bypass disabled. Observed UR
link motion prevented a complete idle checkpoint; no command or physical-history
entry was committed. Shutdown required forced termination, recorded separately.

Implementation now follows the accepted grounding sequence above. The first
blocking gate is a justified native tracking, stationary-containment and stopping
contract under unchanged Gazebo behavior. Existing position commands, sampled
stationarity and observed error maxima do not establish it. Record precise missing
evidence and retain `NEEDS_CONTEXT` wherever a guarantee cannot be established.
Continuous history and a recoverable hold are supporting implementation work;
neither is successful recovery or permission to resume. Full acceptance requires
actual grants, unchanged prepared execution, authenticated effects and valid nominal
continuation with both selected specifications active. Recovery-event generation
for the four failure scenarios remains downstream of that gate.

**2026-10-08 scope clarification (supersedes the full physical-guarantee gate
above):** the user confirmed that the required implementation is Safety Monitor
Construction and Validation (Modified), Steps 1–4, working in Gazebo under explicit
motion and failure models. Proving universal physical tracking/contact/stopping
bounds and implementing automatic nominal resumption are not prerequisites for
this scope. Keep model-based CCA decisions and observed execution distinct from
physical guarantees; never mark `physical_execution_verified` true for modeled
evidence. Retain exact command authorization, selected specifications and
`diagnostic_cca_bypass=false`. Optional runtime hold changes from the broader plan
are being removed; continuous-history replay and evidence-identity checks remain
relevant. The next acceptance gate is an actual CCA-checked modeled motion in
Gazebo, plus a rejected modeled conflict and retained observations.

**2026-10-08 model-based integration update:** continuous checkpoint version 2
now replays possible DFA states across contiguous modeled programs while retaining
the scalar checkpoint contract. Prepared-motion admission now requires an allowed
immediate start from the existing common composition engine; missing graphs,
wrong events and unfulfilled temporal continuations cannot authorize dispatch.
The optional runtime hold changes were removed after the scope clarification.
Owner model preparation separates declared interpolation/stationary assumptions
from `physical_execution_verified`, which remains false. Native provenance checks
remain available as diagnostics without making perfect sampled stationarity a
prerequisite for the model-based method.

The [current background Gazebo evidence](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-2/result.json)
captures all 12 resources without an unresolved modeled checkpoint. It has both
selected specifications active and bypass disabled. Native preparation then rejected
the first trajectory timestamp, 1 ns rather than 0 ns. The raw rejected points are
retained in the owner preparation; no command was authorized or sent. The next
gate is complete native preparation and common-composition evaluation before any
execution. These results do not establish successful recovery.


**2026-10-08 native preparation and evidence update:** owner preparation now
retains the raw native trajectory and explicitly inserts its observed initial hold
at time zero for the specific 1 ns, identical-position, zero-derivative initial
waypoint. Every native waypoint and the final duration remain unchanged; CCA and
execution use the same resulting prepared trajectory. Other invalid start times
remain unsupported. The resource-preparation suite passed 104 tests. Fresh
admission validation now binds controller identity and command revisions as well
as the declared motion model; its 70 admission tests passed before the subsequent
stationary-observation refinement.

The next [Gazebo attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-3.log)
reached report writing but could not serialize a native unsigned numeric value.
Its CCA decision is unavailable and cannot count as a completed acceptance run.
[Read-only reconciliation](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-3/post-failure-observations.json)
found unchanged controller instances and command revisions, with no active or
pending goals. The harness now preserves native integral values as JSON integers;
11 focused harness and Boolean-mode tests passed. Stronger physical guarantees
remain excluded. The next gate is a fully retained common-composition decision,
exact grant, dispatched trajectory and fresh observed completion under the
explicit models.


**2026-10-08 modeled admission and validation update:** fresh observed geometry
is re-grounded through every selected AP and the complete composition calculation;
changed measurements can proceed only when the AP trace, temporal boundaries and
winning immediate start agree. Controller incarnation, command revision, custody
and model configuration remain bound. Sampled `observed_stationary` is retained
as evidence but is not a physical-guarantee prerequisite in explicit model mode.
Source `resource_jid` remains optional as in the existing primitive schema; when
provided it must agree, and prepared owner bindings are checked independently.
Post-execution capture now refreshes Gazebo observations and uses the retained
per-joint model allowance when checking the final joint state.

The [fifth live attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-5/result.json)
passed owner model preparation with `model_execution_verified=true` and
`physical_execution_verified=false`. It exposed missing configured assembly-target
metadata in measured part geometry before the common graph could run; no command
was sent. The geometry adapter now preserves the exact `assembly_target_map`
symbol alongside measured envelopes, without asserting an assembly completion.
The next gate remains a fully recorded permitted live motion and observed outcome.

Current checks include 81 admission, 60 live-safety/nominal-adapter, 45 continuous
motion, 140 KMR and 23 nominal-admission tests passing. All 90 preparation cases
passed across the core run and the separately run UI case; the sandbox UI stall
was resolved by running that case outside the restricted sandbox. The combined
resource/simulation-timing run had 367 passes (including all 115 resource tests)
and six failures in existing collision-policy expectations and a missing archived
recording fixture. The broader coordinator run exceeded its 120-second limit;
its partial progress is not a passing suite. Poetry, compilation and UI CLI help
passed. Repository-wide diff checking reports an unrelated trailing blank line in
`test/test_environment_capabilities.py`; touched-file checks are clean. Logs remain
in the [run checks directory](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/checks).


**2026-10-08 first live-grounded common composition:** the
[sixth background Gazebo attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-6/result.json)
returned an allowed common composition for the supplied first `move_cartesian`.
The graph includes all 12 configured resources, both selected specifications as
67 instantiated rules, an allowed start and a completion witness. This is a
prediction using live observations and native preparation, not observed execution.
Fresh admission then rejected a missing modeled-start binding in the installed
nominal adapter, which overrides the controller preparation method. No grant or
command was emitted. The next gate is preserving the same model binding through
that existing shared adapter and observing the authorized command's outcome.

All 93 current preparation tests passed after preserving assembly-target metadata.
Final retained replay tests passed 22 cases, and source-provenance tests passed two.
The implementation uses supplied recovery motion to test CCA integration; no new
LLM recovery generation has occurred. The single-motion harness assumes other
resources remain in commanded holds. Incorporation of moving nominal work and
modeled failure branches is still a remaining implementation milestone.


**2026-10-08 granted native motion and retained failure evidence:** fixed-link
validation now uses literal native containing links and exact URDF fixed ancestry.
Captured, fresh and modeled root-fixed groups must have identical definite
occupancy for every bound region under the declared stationary model. Moving-body
enclosures remain checked separately, so arm occupancy cannot hide a changed base
AP. This adds no distance tolerance or physical guarantee. Observed initial joints
may differ from the planned initial joints only within named, declared model
bounds; their values and the prepared trajectory are not rewritten. The three
owner/adapter/preparation suites passed 239 tests; continuous motion passed 54,
continuous replay passed 17, and admission passed 81 after these changes.

The [ninth live attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-9/result.json)
received a committed CCA grant with both specifications active and bypass disabled.
Its native goal identity is retained. The observation executor then failed because
installed ROS Humble calls subscription callbacks with one argument, while the
reader required message metadata as a second argument. This caused a read-only
service timeout and prevented observed completion from being committed.
[Direct read-only reconciliation](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-9/post-run-direct-observations.json)
found the same controller incarnation, no active/pending goal, and final joints
within 2.93e-5 rad of the exact prepared endpoint. That observation is not a
retroactive CCA completion. The failed command and reservation remain recorded.

The callback now accepts the installed one-argument interface. Missing publisher
metadata remains unauthenticated for the legacy topic-only idle path; configured
native controller services remain the owner evidence source. Thirteen focused
callback/owner-query tests passed. A fresh identical Gazebo world will test the
complete corrected path without migrating the old grant or erasing its evidence.
Successful exactly-once completion and live occupied-region rejection remain the
next acceptance gates; moving nominal work, modeled failure branches and the four
full recovery scenarios remain unfinished.


**2026-10-08 first supplied live motion completed:**
[acceptance-11](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-11/result.json)
passed the first empty-custody `move_cartesian` gate in a fresh background Gazebo
world on school WSL. Both `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm` were active, with
`diagnostic_cca_bypass=false`. The common composition allowed the supplied event;
CCA committed a grant; the prepared, authorized and executed trajectories are
identical; authenticated observed completion was committed once, advancing
physical history to revision 1. The
[verification summary](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-11/verification-summary.json)
retains the exact command and native goal identities. This is a manually supplied
engineering candidate, not an LLM-generated recovery or completed `part_slippage`
scenario. `physical_execution_verified=false` remains explicit.

AP occupancy comes from the resource geometry along the prepared trajectory,
not the `move_cartesian` symbol or only its TCP target. In this run the configured
`assembly_board-v1` region has world bounds x/y [-0.25, 0.25] m and z [1.05, 1.4] m.
The final observed TCP is approximately (-0.100, -0.080, 1.500) m, while an observed
`ur5e-4` gripper finger envelope has x [-0.079, -0.048], y [-0.144, -0.113], and
z [1.299, 1.357] m, inside that region. Intermediate AP possibilities were evaluated
from continuous modeled geometry before dispatch; final geometry is separately
retained as observation evidence. These envelopes do not assert universal contact
or tracking guarantees.

Final checks include 239 owner/nominal-adapter/preparation tests, the updated full
95-case preparation suite after the callback correction, 81 admission tests,
54 continuous-motion tests, 17 continuous-checkpoint tests, 22 replay tests,
60 live-safety/nominal-adapter tests, 140 KMR tests and 23 nominal-admission tests.
Groups overlap and must not be summed. Poetry, compilation, ROS-free imports,
UI/harness CLI help, `make bootstrap-gazebo` (16 packages), and scoped diff checking
passed. The six unrelated simulation-timing failures and timed-out broader
coordinator run described above remain limitations; the global diff check still
reports the unrelated trailing blank line. No new Python modules were added.
The [aggregate verification record](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/verification.json)
indexes checks, all attempts and cleanup. Both owned Gazebo launches and their
descendants stopped; the earlier failed grant/history was not migrated or rewritten.

The next acceptance gate is a candidate rejected before dispatch because observed
geometry places another resource in the same region. Subsequent work connects
moving nominal operations, remaining recovery events, supported modeled failure
outcomes and authenticated continuous history across successive live programs.
Grasp/release and required KMR/machine/Conveyor/Buffer effects remain unfinished.
Actual recovery-event generation for the four scenarios remains downstream of
those model-based CCA gates. Automatic nominal resumption and universal physical
bounds remain outside this implementation scope.
