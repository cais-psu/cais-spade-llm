# Offline primitive observations for reusable safety specifications

`primitive_observation_safety.json` defines `receiving_region_entry` and
`shared_area_mutex`. Their AP meanings are independent of generated event/state
names. Concrete bindings supply exact resource, part, and region identifiers.
The existing action-based safety specifications and runtime admission remain
unchanged.

The checker is
`cais_spade_llm.agents.central_controller.primitive_program_safety.validate_primitive_program_safety`.
It calls the resource-owned
`cais_spade_llm.resources.primitive_observations.model_primitive_observations`
and checks the joint trace using `BaseSafetyChecker.transition_evidence`.

## Bindings

Each binding has a unique `rule_id`, a `specification`, and a `region` naming
an entry in `geometry.regions`.

| Specification | Additional binding fields |
| --- | --- |
| `receiving_region_entry` | `resource` (incoming resource), `receiving_resource`, `part` |
| `shared_area_mutex` | `resources` (two distinct, mutually exclusive resources) |

Supply every applicable binding. An omitted requirement is not checked; this
offline entrypoint does not discover plant-wide applicability. AP valuations and
DFA histories are scoped by `rule_id`, including when different instances use
the same AP labels. Identifiers are matched exactly, including case and spaces.

## Resource-model input

The inputs are dictionaries suitable for JSON configuration and saved evidence.
The caller must obtain `step_results` and `model_evidence` from the owning
resource's model. This entrypoint checks their consistency; it does not
authenticate their producer or obtain observations from a controller.

- `snapshot.resources`: spatial participants have `current_pose` and an explicit
  `held_part` (a part identifier or `null`). Receiving resources have a complete
  `contained_parts` list. A resource serving both roles supplies both.
- `snapshot.parts`: carried or incoming parts have `current_pose` and explicit
  `contained_by`. Every part in a receiving inventory has matching `contained_by`.
- `geometry.frame`: the exact shared coordinate frame. `geometry.axis` selects
  one xyz coordinate by index. All poses use the existing seven-value xyz/xyzw
  shape. Distances are metres.
- `geometry.regions`: each bound region has `frame` and `interval` (lower and
  upper bounds). These are supplied geometry, never inferred from region names.
- `geometry.resources`: each spatial participant has `frame` and `footprint`
  (lower and upper offsets from its pose on the selected axis). A participant
  carrying a part also has an explicit `attachment_offset` on that axis.
- `geometry.parts`: carried and bound incoming parts have `frame` and
  `footprint`, with offsets relative to the part pose.
- `horizon`: the common start/end times, in seconds on one shared time axis.
- `stationary`: per-resource lists of start/end intervals covering time outside
  supplied primitive steps. Evidence covers unchanged pose, custody, and
  containment, except for the modeled releases. Receiving resources need this
  coverage too. Overlapping stationary evidence cannot contradict motion.
- `programs`: one ordered program per moving resource. Each has `resource_id`,
  bound `primitive_steps`, and matching `step_results`. Other event/state
  identifiers are preserved as metadata.

Every `step_result` has `step_index`, the exact `primitive`, `resolved_params`
matching the authored bound `params`, `start_time`, `end_time`, and
`model_evidence` with the shared `frame`. Existing direct `start_snapshot` and
`projected_snapshot` pose/custody fields, when supplied, must agree with the
physical model. Arbitrary state names do not establish those facts.

| Primitive | Required `model_evidence` |
| --- | --- |
| `compute_place_targets` | Resolved `outputs` containing KMR `target` or robot `target_pose`. It produces data without moving anything. |
| `move_cartesian` | `trajectory`: ordered `{time, pose}` points covering the full step. The target and supplied waypoints must match the bound parameters. Both existing `target` and scalar xyz/xyzw interfaces are supported. |
| `release_part` | `released_part`: `part_name`, `frame`, `pose`, `contained_by`, and `stationary_until` covering the horizon. Position must agree with the held-part attachment at release. |

Release clears custody at the step's end. It updates the part's containment and
the receiving inventory using supplied evidence. The released part stays at its
modeled position while subsequent retreat moves only the robot/tool.
`assembly_slot` placement correction is outside this no-motion release model;
a non-null value returns `NEEDS_CONTEXT`. Other primitives, changing orientation,
translation outside the selected axis, or missing evidence also remain
unresolved. Existing primitive preconditions and execution code are unchanged.

## Observation and checking semantics

The model assumes fixed projected footprints. The resulting checks cover the
declared 1D projection, including the assumption that the other dimensions
overlap. They do not establish full-body 3D clearance, motion feasibility,
placement support, uncertainty bounds, or actual execution behavior.

The evaluator calculates every boundary crossing and includes each crossing
instant and a representative point between crossings. It uses exact rational
arithmetic derived from supplied decimal numbers to preserve simultaneous
contact. `time_exact` retains the exact rational time in the diagnostic trace.

Closed intervals count contact as occupancy. Initially being inside a receiving
region does not invent an entry event. Initially overlapping mutually exclusive
resources violate mutex immediately. Touching during simultaneous exit and
entry also violates mutex.

All participants are evaluated on one joint timeline. At release, the closed
resource envelope includes the just-released part for that instant and the
receiving inventory includes its supplied containment. Subsequent observations
use the robot/tool alone. This checks simultaneous release and entry without
choosing an arbitrary order between resources.

## Results and continuation

`is_safe` reports compliance with the two bound requirements under the supplied
model. `safety_ctx.status` is `safe`, `violated`, or `unavailable`. Missing evidence
returns `NEEDS_CONTEXT` and `safety_validation_unavailable`; a counterexample
returns `safety_rule_violation`, including its AP values, time, position, active
primitive steps, and DFA transition. The compatibility field
`feasibility_status` describes this offline result, not controller feasibility.

Input dictionaries are never changed. Accepted results provide
`safety_dfa_states_after` and `projected_snapshot` for an explicitly subsequent
offline program, preserving occupancy and deposited-part containment. Rejected
and unavailable results preserve the supplied DFA history. Their projected
snapshots, if present, still describe the unaccepted proposal.

Passing never authorizes execution. A future runtime adapter must supply fresh
evidence for all relevant resources and admitted motions, maintain joint
admission, and enforce that execution follows the checked model.

Executable configured examples and regression coverage are in
`test/test_primitive_program_safety.py`. All example coordinates are confined to
those fixtures. Run them with:

```bash
poetry run python -m pytest -q test/test_primitive_program_safety.py
```

## Reviewed offline checking with opt-in 3D observations

The separate entrypoint is
`validate_reviewed_primitive_program_safety`, exported from
[primitive_program_safety.py](../../agents/central_controller/primitive_program_safety.py)
and implemented in
[reviewed_primitive_program_safety.py](../../agents/central_controller/reviewed_primitive_program_safety.py).
The preceding sections describe the legacy entrypoint and its 1D contract. Its
DFA identifiers, meanings, and outputs are preserved. In particular, its
`receiving_region_entry` tool/part occupancy mismatch remains unresolved; the
new part-specific AP does not reinterpret that identifier.

The new path accepts `programs`, `snapshot`, `geometry`, `horizon`, and
`stationary`, plus `catalog`, `applicability`, `continuation`, `trace_complete`,
and optional `observation_slice`. It has no dispatch or admission authority.
`SystemBridge`, PA, RA, CCA, runtime configuration, and Safety-page approvals are
unchanged. A catalog supplied to this offline function is a reviewed input
assumption; this function does not authenticate a human review or an evidence
producer.

### Resource effects and geometry

`geometry.dimension: 3` selects the
[3D observation model](../../resources/primitive_observations_3d.py).
Region `bounds` and resource/part `footprint` each contain three closed intervals,
ordered X, Y, Z. Footprints are offsets in the declared fixed world axes. A
resource with `base_pose: [x, y, yaw]` also requires `base_footprint`; KMR base/body
occupancy is distinct from its arm/tool pose. These envelopes must conservatively
cover whatever physical body the study claims to model. The companion's small
synthetic TCP envelope is not a demonstrated whole-arm collision model.

The model evaluates every configured resource, part, and region independently
of requirement identifiers or formulas. It supports these evidence contracts:

| Primitive | Evidence and limits |
| --- | --- |
| `move_cartesian` | Complete piecewise-linear XYZ `trajectory`; exact target and ordered waypoints; fixed unit quaternion. |
| `move_to_named_pose` | `transport` only, with complete TCP trajectory and resolved `outputs.tcp_pose` and matching `outputs.pose_name`. `home` has unmodeled gripper effects and remains unresolved. |
| `move_base` | Empty gripper, complete fixed-yaw `base_trajectory`, and corresponding TCP trajectory with equal XY translation. |
| `compute_pick_targets` | Explicit `outputs` for `target`, `approach`, `lift`, `lift_waypoints`, and `seed`; no motion. `param_sources` links consuming parameters to exact earlier step outputs. |
| `grasp_part` | Exact part/initial pose, matching `grasped_part` pose and grasp transform, empty resource and consistent previous containment. |
| `release_part` | The held part's matching transform and `released_part` pose/containment; deposition is stationary until its next verified grasp or trace end. |

Custody changes occur jointly at the declared primitive end. A release cannot
move a part discontinuously. It updates complete receiving inventories; a later
grasp removes that containment before carrying the part. A just-released part
contributes to its releasing resource's occupancy at that boundary, then remains
at the deposition pose as the resource retreats. No supported primitive creates
`trim` or `assembly` completion. Unsupported motion, changing orientation,
missing trajectories, invalid transforms, or contradictory snapshots return
`unavailable` rather than being approximated as no motion.
Opaque `current_state` and `resource_location` labels remain annotations; APs use
the modeled pose, custody, containment, and process ledger fields. The checker
does not infer physical facts from those state labels.

### Fixed AP meanings and reviewed formulas

Catalog version 1 retains `id`, `requirement`, `formula`, and `aps`; each AP has
`label`, `full`, and `meaning`. Labels use the existing `ap001` form. Exact
identifiers and meaning strings must match the supported AP evaluators. Geometry
and event/state names cannot substitute for their evidence.

The existing `shared_area_first_resource` and `shared_area_second_resource`
meanings remain unchanged. The new APs are:

| Exact AP identifier | Meaning |
| --- | --- |
| `ap_event/physical_observation/part_region_entry` | The bound part begins touching or overlapping the bound region. Initial occupancy is not an entry; custody changes alone do not create entry. |
| `ap_state/processCompleted/process_result_completed` | The bound part's explicitly complete processCompleted ledger contains the exact bound process and result record. |

Part entry uses the part envelope even when the tool is already inside. A
complete ledger containing the exact `{"process": "trim", "result": "square"}`
record establishes the selected KET4 completion AP. A complete ledger without
that record establishes false. Missing/incomplete ledger evidence establishes
neither. `processCompleted_complete: true` and
`processCompleted_evidence: {complete: true, source_kind: ..., checkpoint: ...}`
are required and must agree. Ledgers remain fixed during this supported trace,
so completion observed at entry was already established at the checkpoint.
Initial occupancy creates no fabricated entry and provides no certification of
unprovided history before that checkpoint.

The strict compiler uses `LTLfParser` and MONA in a temporary directory. It
validates the generated automaton and AP labels without preview files, LLM calls,
or silently omitted rules. Supported formulas include `G`, `F`, `X`, and `U`;
a failed compilation or unsupported predicate returns `unavailable`. No mutex
DFA is substituted for a different formula.

### Deterministic applicability and declared coverage

Applicability version 1 supplies `scene_resources` with exact `resource_id` and
`resource_type`, `participant_resource_types`, each region's `resources` and
`excluded_resources`, and `rules`. A pair rule uses `bindings: "all_pairs"`.
The checker derives every unordered pair in deterministic order and requires
coverage of the declared population and catalog. Explicit per-rule bindings
supply the exact part, region, process, and result required by their APs.
Missing participants, applicable rules, geometry, or motion/stationary evidence
cannot be interpreted as safe inactivity.

The [storage_interruption companion](../../../test/fixtures/KMR_assembly_board-v1_recovery/storage_interruption/safety_evidence.json)
derives its population from the scene's four `ur5e` resources and `KMR`, giving
10 mutex pairs and one KET4 precedence instance. Seven fixed resources are
excluded under explicit reviewed assumptions. File hashes record the source
scene and original fixture records. This is deterministic coverage relative to
the supplied catalog, scene, and exclusions; it cannot discover an entirely
omitted hazard or authenticate an intentionally incomplete scene declaration.
Machine interlocks, capacity, arbitrary predicates, and a complete plant hazard
catalog remain subsequent work.

### Frozen trace clock and continuation

The clock version is `primitive_observations_joint_trace_v1`. Its one frozen
joint trace contains the initial observation, primitive/trajectory/custody
boundaries, every configured region contact boundary, and a representative
observation between consecutive boundaries. Motion, custody, containment, and
simultaneous contacts are evaluated together. Exact rational arithmetic over
supplied decimal coordinates determines boundary times. Interval samples
represent intervals only when supported physical AP values remain constant;
part-entry APs mark the first contact observation, not an extended interval.

`X` means the next observation in this declared trace, not the next primitive,
task, or elapsed second. Extra waypoints, scope changes, or geometry changes can
change that clock and therefore change `X`. There is no arbitrary sampling
period and no extra empty completion observation. The fingerprint includes the
clock version, full physical inputs, scope, AP meanings, compiled formulas, and
frozen observations. It makes no claim of equivalence under resampling.

`observation_slice: [start, end]` selects half-open indices. Every continuation
must begin at the previous `next_observation` within the same frozen trace.
Continuation records retain DFA `states`, `previous_observation`, trace identity,
clock version, and completion status. The checker replays accepted history to
reject forged states, gaps, duplicate observations, or changed inputs. This is
continuation within a fully supplied trace; accepting a new physical program
with a different horizon is a future integration problem.

| Result | Meaning |
| --- | --- |
| `prefix_checked` | The consumed prefix retains an accepting path in every formula DFA. Nonaccepting states are listed in `pending_rule_ids`. Even an accepting prefix is not declared complete. |
| `satisfied` | `trace_complete=True` at the actual trace end and every DFA accepts. |
| `violated` | A rule has no accepting continuation, or a nonaccepting rule remains at explicit completion. |
| `unavailable` | Required evidence, binding, compiler support, or trace identity is missing or inconsistent (`NEEDS_CONTEXT`). |

`is_safe` is true only for `satisfied`. The result returns checked observations,
bindings, coverage, rule transitions, counterexamples, and candidate continuation.
An accepted projection describes only the slice endpoint. Rejected/unavailable
checks preserve supplied continuation and DFA states; they do not provide an
accepted projected snapshot. Completion can finalize an already consumed trace
with an empty slice at its actual end, consuming no additional observation.
Per-rule accepting reachability does not establish a compatible physical joint
continuation. Local parallel composition remains responsible for that next gate.

### Conditional soundness and evidence limits

Under the assumptions that the declared population/exclusions are complete,
configured envelopes contain the modeled bodies, trajectories and custody
records are accurate and complete, and the reviewed AP definitions/formulas and
MONA compiler are trusted, the result checks those formulas on the declared
finite trace. For fixed envelopes undergoing linear translation, region
occupancy can change only at computed contacts or custody boundaries. Evaluating
all boundaries and constant intervals therefore detects supported mutex
conflicts and part-entry violations within this model. Complete checkpoint
ledgers make process-result membership deterministic; they do not verify that a
real machining operation occurred.

The argument is conditional and clock-specific. It is not a proof for arbitrary
physical predicates, continuous-time LTL semantics, changing orientations,
uncertain trajectories, unsupported geometry, or actual controller execution.
General `F`/`U` obligations and individual DFA continuation paths can still lack
a physically feasible joint continuation. Actual controllability, dependency
closure, runtime admission, Gazebo execution, and nominal resumption are later
acceptance gates. The fixture's pending M1 task records are preserved, not run.

### Verification on 2026-10-04

`poetry run python -m pytest -q test/test_primitive_program_safety.py` passed
**186 tests in 6.95 seconds**: 98 retained legacy tests and 88 new cases.
Sixteen formula tests compare 5,440 Boolean finite traces against an independent
semantic evaluator. The 22-primitive synthetic baseline gives 107 joint
observations and satisfies 10 mutex instances plus KET4 precedence. Negative
cases establish violations, unresolved evidence, preserved rejected history,
continuous slice semantics, exact primitive effects, and parameter consistency.
The original 22 primitive endpoints, part ledgers, Storage inventory, source
references, and pending M1 records are checked against the saved fixtures.

Poetry validation, compilation, focused Ruff checks, JSON/local-link checks, and
`git diff --check` passed. Poetry reported existing metadata deprecation warnings.
Protected original artifacts and unrelated tracked changes were preserved.
These results are offline software checks; no LLM calls, runtime integration,
Gazebo execution, or completed nominal resumption are asserted. The legacy
`receiving_region_entry` counterexample remains documented in the roadmap.

## Automatic offline grounding

`validate_grounded_primitive_program_safety` in
[`offline_safety_grounding.py`](../../agents/central_controller/offline_safety_grounding.py)
accepts `scene`, `catalog`, `requirement_scopes`, the existing physical inputs
(`programs`, `snapshot`, `geometry`, `horizon`, `stationary`), and optional
`task_evidence` / `state_evidence`. It also accepts `continuation`,
`observation_slice`, and `trace_complete`. It freezes the inputs, constructs
bindings, and delegates compilation and trace evaluation to the same reviewed
checker. Both existing public checking entrypoints remain available unchanged.

The population is exactly the keys of `build_environment_models(scene)`.
Every configured resource needs geometry, a snapshot, and complete motion or
stationary coverage; no caller-authored participant list, resource-type filter,
or exclusion pruning is accepted. Reviewed scopes supply requirement-specific
identifiers, for example:

```json
[
  {"specification": "shared_area_mutex", "region": "assembly_board-v1"},
  {
    "specification": "KET4_Square_4mm_trim_precedence",
    "region": "assembly_board-v1",
    "part": "KET4_Square_4mm",
    "process": "trim",
    "result": "square"
  }
]
```

Every unordered pair is constructed for a scoped mutex: 12 resources give 66
instances; adding a configured resource gives 78 instances without changing the
catalog or these scopes. Geometry does not decide which requirements apply.
All catalog requirements need scopes. Completeness is relative to the frozen
configuration and the supplied reviewed catalog, not to all possible hazards.

`geometry.resources[resource_id].stationary_only: true` declares a fixed
configured envelope, with an explicit `current_pose` and full stationary
coverage. It forbids resource programs, base motion, and invented gripper/custody
fields. `contained_parts` can still change when another resource deposits or
grasps a part. The grounder rejects this flag for resources whose model declares
`held_part`; it cannot bypass required robot custody evidence.

The [new synthetic companion](../../../test/fixtures/KMR_assembly_board-v1_recovery/storage_interruption/automatic_grounding_evidence.json)
embeds the frozen scene and the unchanged 22 primitives. Seven fixed resources
have explicitly synthetic envelopes and stationary evidence. The earlier
five-resource companion and original recovery records remain byte-for-byte
unchanged. The declaration builder accepts an additional configured robot with
no implicit handling assignment; the test adds `ur5e-5`, whose only nominal
event is `move_home`. This does not install or dispatch a new robot. Changing
the scene requires new model/trace identities; ordinal event references from
another scene are not transferable continuation evidence.

### Preserved task/state APs

The catalog retains each AP's exact `full`, `label`, and reviewed `meaning`.
Existing task descriptors keep task-execution semantics. For example, the saved
`ap002` is
`ap_event/assembly/any/recovery-resource-3/place_approach/destination=assembly_board-v1`;
the saved `ap008` is
`ap_state/assembly/any/recovery-resource-3/positioned/destination=assembly_board-v1`.
Their explicit scope associations are:

```json
{
  "ap002": {
    "source": "task_event", "resource_id": "ur5e-3",
    "resource_symbol": "recovery-resource-3", "process": "assembly",
    "product": "any", "function": "place_approach",
    "context": "destination=assembly_board-v1"
  },
  "ap008": {
    "source": "resource_state", "resource_id": "ur5e-3",
    "resource_symbol": "recovery-resource-3", "process": "assembly",
    "product": "any", "state_field": "resource_state", "state_value": "positioned",
    "context": "destination=assembly_board-v1"
  }
}
```

This object is the requirement scope's `ap_groundings`. Each structured AP
requires one association. Resource aliases are explicit reviewed bindings, not
inferred from spelling. Each resource symbol has one consistent association; an
existing configured identifier cannot be rebound to a different resource. `process`, `product`, and `context` support the existing
`any` selector; task `function` is always exact, including a literal `any`.
Unbound resource selectors and unsupported descriptor forms remain unavailable.
Strings retain their exact spelling and whitespace. Declared Boolean state
values use `true` / `false`; the existing null scalar spelling is `None`.
State values must satisfy the frozen resource model's field type/domain.

`task_evidence` contains `complete: true`, a nonempty `source_kind`, the exact
`horizon`, and `events`. Each event records `task_id`, `resource_id`, `process`,
`product`, `function`, `context`, `start_time`, and `end_time`. Intervals have
positive duration and are `[start_time, end_time)`. Duplicate IDs and overlapping
intervals for one resource are unavailable. A complete empty ledger can establish
that no matching task occurs; a missing ledger cannot.

`state_evidence` has the same completeness/source/horizon fields, `initial`, and
`updates`. Each initial row supplies `resource_id`, `process`, `product`,
`context`, and `values` keyed by declared state fields. Each update also supplies
`time` and `task_id`, matching that task's completion time and exact identity/context.
One joint update per resource/time applies all its fields together. Missing
fields, invalid domains, contradictory `held_part`, and ambiguous updates return
`NEEDS_CONTEXT`. These ledgers assert resource-owned observations; the checker
does not derive state facts from task names or certify their physical truth.

Physical occupancy can change inside one task interval. That does not fabricate
another `place_approach` occurrence. Newly authored recovery event names affect
provenance, while identical primitives and physical inputs produce identical
physical AP values. Physical predicates continue to use the existing evaluators;
no task descriptor is reinterpreted as an occupancy or process-completion predicate.

### Combined clock and evidence

`grounded_primitive_observations_joint_trace_v1` adds all supplied task starts,
task ends, and completed state updates to the physical boundary set **before**
constructing midpoint observations. Simultaneous effects share one observation.
At a task end its event AP is false and its completed state update is visible.
Supported AP values remain constant on open intervals between these boundaries,
apart from entry APs that mark their explicit outside-to-contact boundary.
`X` refers to the next observation of this combined trace. Adding a task boundary
can change an `X` formula; there is no clock-independent interpretation claimed.

Fingerprints include the frozen scene, geometry, primitive evidence, task/state
ledgers, catalog, scopes, concrete bindings, compiled DFAs, and observations.
Continuation requires contiguous slices of that exact trace and rejects the
previous physical-only clock. Explicit completion consumes no invented empty
observation. Rejected or unavailable checks preserve supplied monitor history.

`ap_evidence` exposes `descriptor`, `binding`, `evidence_source`,
`observation_index`, `time`, `phase`, and Boolean `value` for each rule/AP in the
requested slice. Physical `rule_checks` and counterexamples retain `active_steps`,
including the original recovery event and local primitive index under `source`.
Projected KMR custody and the pending M1 delivery records are preserved.

The conditional soundness argument above now assumes the complete population
returned by the frozen builder, with no unchecked exclusions. It additionally
requires complete and accurate task/state ledgers and reviewed identity
associations. Supported physical predicates change only at modeled boundaries;
task predicates change at interval ends/starts; state predicates change at
recorded completions. Their union therefore supplies one sufficient boundary
set for these meanings, and strict DFA evaluation checks the declared finite
word. This is automatic grounding over supported meanings and evidence, not
predicate discovery, physical feasibility, runtime enforcement, or nominal
resumption. The bounded offline composition extension below connects this evidence
to the existing nonblocking analysis; runtime integration remains a separate gate.

Automatic-grounding verification on **2026-10-04** passed **259 safety tests in
26.23 seconds**, including the prior 186 tests, and **53 relevant nominal
declaration/projection tests**. The synthetic baseline checks 66 mutex instances
plus precedence; the additional-resource case checks 78 mutex instances. Tests
preserve all 22 primitive references, custody, ledgers, and pending M1 tasks,
and reject incomplete or contradictory physical/task/state evidence. Boolean
states, exact string tokens, task interval boundaries, `X`, pending completion,
and continuation are covered. Poetry, compilation, focused source/safety-test
Ruff, JSON/reference checks, and `git diff --check` passed. The broader nominal
UI run was interrupted; nine pre-existing nominal-test Ruff findings remain.
No local composition, LLM, Gazebo, or runtime admission was invoked.

## Offline local parallel composition for generated recovery events

`analyze_grounded_recovery_composition(...)` in
[offline_recovery_composition.py](../../agents/central_controller/offline_recovery_composition.py)
has no execution authority. It uses the existing `Scope`, `Analysis`, `Budget`,
and `nonblocking_region` with an observation-driven path. The nominal backend's
task-transition and terminal-observation behavior is unchanged.

Its frozen `grounding_inputs` have the same scene, catalog, requirement scopes,
physical evidence, and optional task/state ledgers as automatic grounding.
`recovery_events` associate exact `outline_id`, `event_name`, `resource_id`,
`primitive_step_indices`, and `predecessors`. An optional explicit `des_event_id`
must agree with primitive provenance; it need not equal `outline_id`. Without
that optional field, one consistent nonempty source `des_event_id` is required.
`running_work` associates exact
`task_id`, resource, event name, remaining interval, and primitive indices.
Together these records must cover every supplied primitive exactly once.
`event_start_choices` supplies finite complete alternatives: an exact `id`, all
event `starts`, complete joint `programs`, and full `stationary` evidence, with
optional task/state ledgers. The implementation does not invent delays, new
trajectories, primitive stopping points, or motion uncertainty.

Each event commits to its supplied internal primitive behavior. Alternatives may
change its explicit start time, while already-running work retains its supplied
remaining behavior. Starts are controllable; all subsequent primitive progress
and completion are uncontrollable. Only explicitly supplied waits are available.
The graph retains unsafe forced edges, so a conflict halfway through an admitted
event cannot be avoided by pretending to pause at an observation boundary.
Event-relative durations, trajectories, primitive outputs, and source references
must agree across alternatives. Initial task/state evidence and external task
completions are fixed. Only a task whose exact `task_id` matches a recovery
`outline_id`, resource, function, and interval may shift with that event; its
completion values remain unchanged. Choosing when KMR starts cannot choose a
different completion state for already-running `ur5e-3`.

Trace preparation is separated internally from formula verdict evaluation.
Complete physical observations, AP evidence, and unsafe valuations remain
available to composition. All configured resources and applicable bindings are
included, without pruning. Every DFA consumes APs from its own `rule_id` mapping;
the reused labels `ap001` and `ap002` are neither renamed nor merged globally.

### Clock, history, and completion

The composition clock includes the supplied finite decision times along with
physical and task/state boundaries before interval representatives are created.
Simultaneous starts, completions, custody changes, and state updates produce one
joint observation. A start decision does not itself consume another observation.
Completion tests the actual final DFA states without adding an empty observation.

Parsed-formula inspection permits Boolean combinations of `G`, `F`, and `U`.
`X`, weak next, and other unsupported temporal operators make composition
unavailable. This restriction leaves standalone reviewed `X` support unchanged.
The clock and complete composition problem are versioned and fingerprinted.

An `accepted_prefix` contains a problem identity and a replayable edge path;
monitor states are reconstructed from the explicit checkpoint. History from
another frozen problem or the standalone trace checker is incompatible. Replay
preserves observations and pending obligations across event boundaries rather
than resetting DFAs. Future evidence such as `stationary_until` is not a present
physical fact on which a controllable choice can depend.

`completion` requires explicit resource and part fields. A completed witness
requires those physical conditions, all included running work finished, and
accepting selected DFAs. All initial process ledgers and pending nominal task
records are preserved. A safe prefix with no supplied behavior to discharge a
remaining eventual requirement is inconclusive. It cannot authorize a start
merely because the DFA has an abstract accepting path.
Supported resource goal fields are `frame`, `current_pose`, `base_pose`,
`held_part`, `gripper_state`, `grasp_transform`, and `contained_parts`; part
goals use `frame`, `current_pose`, `contained_by`, and `processCompleted` with
its completeness/evidence fields. Copied metadata such as `current_state` or
`resource_location` does not prove a goal and is unavailable as a completion
condition in this path.

The result exposes `allowed`, `held`, or `inconclusive`, scope, fingerprints,
explored states, permitted start/wait choices, replayable decision prefixes,
pending rules, and a completed witness or counterexample with original event and
primitive provenance. Exhaustion returns inconclusive. Defaults remain
**20,000 states / 2 seconds**; generous explicit budgets in software tests do not
change these defaults.

### Synthetic composition evidence and conditional soundness

The [composition companion](../../../test/fixtures/KMR_assembly_board-v1_recovery/storage_interruption/composition_evidence.json)
references the unchanged automatic-grounding companion by SHA256. It supplies
two explicit schedules over `[0, 26]`, with the same four recovery events and
22 KMR primitives plus the remaining motion of already-running `ur5e-3`.
Immediate placement conflicts during descent while `ur5e-3` still occupies
`assembly_board-v1`. Delaying that whole event until `12` permits its modeled
completion. Complete stationary evidence covers the wait and horizon tail.

The endpoint restores KMR holding `KET8_Square_8mm` at Storage, deposits
`KET4_Square_4mm`, and completes the included withdrawal. The original
`KMR_STORAGE_KET8_MOVE_TO_M1`, `KMR_STORAGE_KET8_PLACE_APPROACH`, and
`KMR_STORAGE_KET8_PLACE_RELEASE` remain pending. Placement preserves `trim`
evidence and does not create `assembly` completion.

The conditional argument assumes complete and accurate configured geometry,
fixed trajectories, custody, ledgers, event boundaries, and supplied alternatives.
The observation model partitions supported AP changes; strict per-rule DFA
transitions therefore check each supplied finite word. Shared prefixes expose
only the supplied event-start choices. The nonblocking fixed point retains
states with a path to an accepting physical endpoint and excludes states with
an unavoidable successor outside that region. Thus a permitted start preserves
a completion strategy within this finite evidence model. This claim does not
cover omitted behavior, timing uncertainty, arbitrary predicates, controller
controllability, physical feasibility, runtime enforcement, or nominal resumption.

Verification on **2026-10-04** passed **305 tests in 66.48 seconds**: 46 new
composition cases and all 259 existing primitive-safety cases. Four formula cases
use independent finite-word evaluation over every supplied schedule. Regression
coverage includes fixed event behavior, fixed external task/state effects,
pending `F`/`U`, incompatible prefixes, unavailable `X`, reused AP labels,
distinct event provenance, exact joint observation consumption, explicit waits,
incomplete evidence, unsupported completion fields, and budget exhaustion.
All 13 existing nominal local-composition cases also passed: 12 in the restricted
environment and the asynchronous arrival/admission case outside it after the
restricted run hung. Poetry, compilation, focused Ruff, JSON/source-hash checks,
local links, and `git diff --check` passed. Original JSONs, runtime configuration,
Safety-page selection/approvals, and unrelated changes remain preserved.
