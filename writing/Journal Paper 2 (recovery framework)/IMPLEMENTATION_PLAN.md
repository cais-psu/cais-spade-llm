# Recovery framework roadmap, including UI dry runs and trustworthy prompt inspection

## Current handoff — October 13, 2026 — read this first

This section is the starting point for the next chat. Older entries below are
historical evidence, not the current next-step list. Preserve them. The user
requested this handoff because the previous chat became too long and costly.

### Objective and current result

Focus on **Machining breakdown during part processing**:

**Capture → stop Gazebo → LLM proposal → RA → physical projection → CCA →
append/project → repeat until nominal continuation is established → UI review.**

The latest machine-breakdown attempt is **incomplete**, but it is no longer
zero accepted events. In `attempt_07`, the provider's corrected M2 `pause_job`
proposal passed RA, physical projection, and both CCA specifications. The
accepted prefix contains that one event. The later KMR `move_to_location`
proposal passed RA but could not establish physical projection. There is no
complete machine-breakdown outline, nominal-continuation proof, primitive
composition, recovery execution, or observed assembly completion to report.
Earlier Part slippage success is recorded separately below; it does not prove
this machine-breakdown case works.

Seven development API attempts are retained in the October 13 result bundle
(957,517 total API tokens). Do not repeat another API attempt against the same
deterministic KMR implementation blocker. Reproduce and repair it locally first.

### User constraints and authorization to carry forward

- Use one agent and the school Ubuntu checkout
  `/home/jongh/projects/cais-spade-llm`, branch `recovery-framework-journal`.
  Read `AGENTS.md` and inspect the dirty worktree before editing. Many changes
  predate this handoff; do not reset, discard, or rename them.
- Preserve `SystemBridge`, exact project identifiers, Offline Manual, expandable
  turns, original captures, historical provider responses, and the current UI run.
- Shared compact instructions and dynamic public `need_context` retrieval remain.
  Do not supply a preferred resource, recovery sequence, worked answer, or event
  count. Test fixtures and private validator programs must stay out of prompts
  and public retrieval. Every accepted recovery event must originate in an exact
  captured provider response.
- Preserve `primitive_support_semantics="successor_primitive_union"` and
  `validation_contract_version=2`. RA checks unordered primitive support against
  one starting snapshot. It does not establish an execution order or trajectory.
- Preserve `safety_M2_mutex.txt` and
  `safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt` unchanged.
  Keep unknown evidence unresolved; do not weaken CCA to obtain acceptance.
- The user already authorized OpenAI API use for the repair loop and an isolated
  Gazebo recapture when necessary. Use `gpt-5.6-sol`, reasoning effort `low`,
  service tier `default`, 80 turns, and no overall ten-minute deadline. Preserve
  the existing scenario conditions, durations, and seed. Do not reintroduce the
  rejected changed-condition evaluation or disable KMR, `side_access`, or M2.
- Ordinary offline preparation, retrieval, and validation must not call Gazebo,
  perception, controllers, or SPADE transport. Capture is a separate owned run.
- Primitive composition, recovery execution, tool changes, new KMR movement
  validators, and concurrent runtime CCA integration remain outside this repair.
  Reuse existing owner/projection components and repair demonstrated integration
  defects. Explain any unavailable capability before expanding the scope.

### Why another capture was discussed

KMR **was captured**. Its base was moving toward Storage in both saved frozen
checkpoints (`885361b9a7244e2fa4c6a8daa47ca278` and
`dfa4017c0119403088d1d4f353a407d5`). Freezing Gazebo time preserves that moving
state; it does not prove the base has stopped or will remain at that pose.
The saved records do not establish the stopped starting configuration required
by the current projection approach.

One new capture is therefore the planned way to obtain that starting evidence.
It is **not needed for each LLM proposal**, and it will not, by itself, repair
the missing KMR `move_base` projection connection. Code repair and local replay
can proceed without Gazebo. Do not promise that recapture alone completes the
outline.

The checkpoint repair now implemented in `recovery_observations.py` does this:

1. Block new nominal dispatch and save the original fault observations and
   pending tasks before stopping handling motion.
2. Cancel owned handling motions through existing owner interfaces. Wait for
   controller inactivity and native observations confirming stopped roots.
   A cancellation acknowledgement or frozen simulation alone is insufficient.
3. Leave machine job commands untouched: do not issue M2 `pause_job`, `stop_job`,
   or a recovery event. M2 must still be processing at the captured checkpoint.
4. Freeze simulation time, obtain owner job observations, physical observations,
   planning scene, and model evidence for that checkpoint; preserve fault time
   separately from checkpoint time. Record later shutdown separately.
5. Report an ended M1/M2 processing overlap or missing evidence without silently
   altering timing or automatically repeating the trial. Preserve the unfinished
   KMR Storage task; cancelling its motion does not complete that task.

Relevant code: `_cancel_checkpoint_motion`, `capture_machine_checkpoint`, and
`checkpoint_fact_fingerprint` in
`cais_spade_llm/recovery_framework/recovery_observations.py`.
The cancellation evidence is included in new checkpoint fingerprints without
changing the fingerprint format of historical records lacking that field.
**This repair has local tests; it has not had a new live capture trial.**

### Physical projection and local parallel composition

Physical projection answers which configurations, custody, geometry, and process
facts follow from the proposed event. CCA uses those facts for the selected APs.
An M2 mutex verdict requires supported robot/tool/carried-part occupancy through
the relevant interval; a station name or clear TCP is insufficient. Processing
precedence uses exact supported part/process/result completion evidence.

Local parallel composition already exists in nominal CCA admission through
`environment_admission.py`, `environment_composition.py`, and
`agents/central_controller/local_composition.py`. `EnvironmentAdmission` calls
`analyze` before granting nominal dispatch and includes interacting work and
the selected monitors. Do not rebuild this nominal functionality.

The current offline outline instead checks each proposed transition with RA,
physical projection, and `validate_outline_evidence_safety`; it then uses
`validate_nominal_continuation` to check the recorded unfinished tasks from the
accepted projected state. It does **not** establish concurrent recovery × recovery
or recovery × nominal execution. That later runtime integration must not be
described as already complete, or confused with the separate primitive
composition stage. In a combined CCA check, supported physical facts supply the
AP valuations; composition cannot manufacture missing geometry.

### Remaining demonstrated blocker and next steps

1. **Complete the existing KMR owner projection connection locally.**
   `kmr_outline_evidence.py::_motion_request` still explicitly raises
   `KMR base movement needs recorded continuous base-controller and carried-part motion evidence`
   for `move_base`. RA success cannot bypass it. Earlier released-part collision
   identity and destination-parameter preparation defects were repaired locally.
   Reuse `kmr_motion.py`, the owner declarations, and the existing motion/geometry
   calculations; keep supported parameters and provenance separate from proposals.
2. **Use the native preview work already present, without overclaiming it.**
   `PreviewRecoveryMotion.action`, `preview_recovery_motion.cpp`, and
   `recovery_motion_preview.py` contain `base_trajectory` support. This is not yet
   connected into a complete KMR owner/CCA proof. A sampled native collision
   check alone is not continuous-motion evidence. Do not invent a controller
   trajectory, use arbitrary interpolation as recorded Nav2 behavior, or present
   a moving capture as stationary. Separate endpoint and interval coverage.
3. **Carry the whole supported scene forward.** Retain peer configurations,
   carried-part geometry, attachments, deposited parts, retained jobs, processing
   time, and CCA history. Inspect `_finish_environment_projection` in
   `resources/environment_recovery.py` when connecting KMR: propagation of changed
   models/geometry into `successor_evidence` needs verification, not an assumed
   pass. Running peer jobs must not silently freeze during projected intervals.
4. **Take one isolated capture after local preparation is ready.** Use the fixed
   checkpoint ordering and unchanged saved setup. Verify KMR stopped evidence,
   M2 still processing, scene/model compatibility, complete relevant observations,
   and source hashes before stopping the owned simulation. Keep a new directory;
   do not overwrite or relaunch the existing UI run.
5. **Replay saved proposals before paying for another API attempt.** Use exact
   captured M2 and KMR proposals as local replay input only. Verify separate RA,
   projection, and CCA results and unchanged accepted state/history after rejection.
   If facts remain unavailable, report their exact sources and gaps.
6. **Then run a fresh offline API attempt.** Freeze implementation, models,
   specifications, and capture hashes during each attempt. Allow successive LLM
   events and dynamic retrieval until nominal continuation passes or a specific
   blocker is recorded. Diagnose each new blocker locally before another attempt.
   Never insert recovery events or relax evidence requirements to finish.
7. **Accept only a complete independently audited outline.** Every accepted event
   must match a provider proposal and pass RA, physical projection, and CCA; the
   unfinished nominal obligations must have supported continuation. Save exact
   requests/responses, retrievals, timings, tokens, and all failures for UI review.

### Evidence and commands for the next chat

All paths below are relative to `/home/jongh/projects/cais-spade-llm`.

- Current immutable source:
  `cais_spade_llm/monitor/environment_runs/recovery/derived_dfa4017c0119403088d1d4f353a407d5_20261013_public/observations.json`
  — SHA-256 `870b2ff445fbc600c8d73fbac5c4c94ad0c66e5107002d73d783c24634676dfd`.
- API/capture history:
  `cais_spade_llm/monitor/environment_runs/machine_synchronized_capture_20261013T040945Z/`.
  Read `RESULT.md`, `RESULT.json`, and `attempt_07/report.json` selectively.
  Exact attempt-07 provider records are in
  `cais_spade_llm/monitor/environment_runs/recovery/offline_b593cb4f3d1b4ad0b9a1a6026c41efcd/generation_0001/recovery_outline/requests/`.
  Use provider `raw_response.choices[0].message.content`, not an enriched session
  response, when checking exact LLM authorship.
- Latest completed local KMR replay:
  `cais_spade_llm/monitor/environment_runs/kmr_offline_goal_20261013T120213Z/local_replay_parameters/report.json`.
  This replay makes zero API/live-interface calls; M2 passes, KMR stops at the
  missing continuous `move_base` projection. Source observations remain unchanged.
- Current checkpoint repair evidence:
  `cais_spade_llm/monitor/environment_runs/kmr_owner_base_20261013T124342Z/`.
  Read `checkpoint_repair_report.json`, `tests_checkpoint_motion.log`,
  `poetry_check.log`, and `compileall.log`. `before.json` retains source and safety
  fingerprints. No capture or API attempt was started in this handoff work.
- Existing runners in `machine_synchronized_capture_20261013T040945Z/`:
  `replay_attempt_07.py` and `offline_runner.py` take source and new output paths;
  inspect their arguments before reuse. `capture_runner_03.py` is a template with
  fixed `capture_03` output: copy it into a new attempt directory and change only
  its output destination. It checks the prior setup and restores its recorded seed.
  The isolated harness used ROS domain 89 and Gazebo port 11379; check availability
  and process ownership before reuse. Source `/opt/ros/humble/setup.bash` and
  `/home/jongh/ros2_ws/install/setup.bash` before the live harness. The earlier first
  capture failed when ROS imports were unavailable.

Latest local verification: **168 passed in 40.24 seconds** across
`test/test_recovery_evaluation_capture.py`, `test/test_machine_jobs.py`, and
`test/test_conveyor_fault.py`. The checkpoint integration fixture was corrected
to include its recorded run/resource identities and participant list; production
compatibility checks were not weakened. Poetry check passed with existing metadata
deprecation warnings; `poetry run python -m compileall -q cais_spade_llm ros2`
passed. Both selected safety files and the source capture still match their hashes.
The earlier 432-test owner-evidence result is separate and overlaps other suites;
do not add test counts together.

Local pytest previously stalled inside the restricted sandbox; the completed
suite ran outside it with mocked live/API interfaces. All owned test commands
from this handoff finished. No new Gazebo or API process was launched. Verify
host processes before any future launch; do not terminate unrelated processes.

**Suggested next-chat instruction:** Read this current handoff first, inspect the
existing KMR projection blocker and saved local replay, and continue the scoped
repair toward a complete machine-breakdown offline outline. Keep one agent and
factual prompts. Do not start with another unchanged API call or read every large
historical capture. Finish the missing code/evidence connection locally, obtain
the one required stopped checkpoint, then resume offline validation and generation.

### October 9: restore RA primitive support and independent capability validation

New explicit version-2 UI/offline attempts pin
`primitive_support_semantics="successor_primitive_union"`. RA matches declared
public resource successor conditions, derives the complete unordered primitive
union, and requires an independent parameter/capability witness for every
primitive against the unchanged starting snapshot. Resource declarations need
not contain `resource_state`; `held_part` has no special matching rule. Declared
bindings, domains and supported conditions retain their exact identifiers.
Primitive preconditions/effects and product end conditions remain composition
obligations. Outline validation no longer searches for an ordered primitive
program. The shared parameter-assignment limit is 4,096; incomplete evidence
or search remains `NEEDS_CONTEXT`.

Scoped, sourced parameter evidence is passed to resource-owned evaluators.
Validator retrieval updates the next validation inputs and terminates when
recorded facts do not change. Detached offline resources perform no equipment
or SPADE calls. Private validator evidence remains outside PA's public prompt.
Historical unmarked attempts retain their validation path; saved-result loading
preserves the recorded marker and evidence. New marked requests reject stale
replies using different primitive support semantics.

RA `FEASIBLE` is recorded separately from missing geometry and CCA outcomes.
An unordered primitive union supplies no ordered trace, physical successor or
motion interval. The exact selected specifications and CCA history requirements
remain enforced. Unresolved events do not append, project, advance nominal
continuation or authorize execution. Supplying independent physical projection
evidence to CCA remains subsequent work; this RA repair does not implement it.

The saved buffer proposal replay now matches `place_insert` and derives
`move_cartesian`, `release_part`, `move_relative`. `part_state="ready"` and
`part_location` are deferred. Its remaining `NEEDS_CONTEXT` concerns scoped
parameter/capability evidence, including `_task_ctx.insert_pose`, `model_name`,
`assembly_slot` and the values needed for `dz`. The original observations hash
is unchanged. This replay is recorded-data verification, not new LLM generation.

**Automated verification:** 502 recovery/context/offline/UI/request/evidence
tests passed. The full feasibility suite recorded 158 passes and 12 failures
in retired AP descriptor fixtures and KMR prepared-execution provider setup;
the recorded stacks do not enter the changed union code, and a pre-change
baseline replay was not performed. Poetry checks, Python compilation and diff
checks passed, with existing Poetry metadata warnings.

**Requested model comparison:** three fresh attempts each for `gpt-4o` and
`gpt-5.6-sol low` completed in
[comparisons/20261009T173155Z/REPORT.md](../../cais_spade_llm/monitor/environment_runs/recovery/comparisons/20261009T173155Z/REPORT.md).
The user authorized all six after the explicit disclosure question; automatic
approval review accepted the retry. These are six new attempts, separate from
the preceding comparison. All six ended `NEEDS_CONTEXT`, with zero accepted
events. Twenty-one exact provider requests and responses were captured; no API
errors or timeouts occurred. Initial prompts, schemas, the two selected safety
specifications, original capture and 343 frozen files remained unchanged.

`gpt-5.6-sol low` requested saved resource context and proposed `place` for
`KET4_Square_4mm` at `Buffer For Machined parts` in all three attempts. RA now
derives `move_cartesian`, `release_part`, `move_relative` from `place_insert`;
the remaining blocker is scoped parameter/capability evidence. Retrieval of
`/resources/recovery-resource-3@localhost/parameter_witnesses` returned no facts.
The capture contains no resource `parameter_witnesses` fields. Recorded parameter
references still depend on `_task_ctx.insert_pose`, `model_name`, `assembly_slot`
and `dz`; the offline capability path does not calculate or establish this
evidence. `gpt-4o` also reached missing parameter/capability evidence, after two
structural errors and four unsupported resource successor proposals across its
three trials. Complete model quality remains unmeasured because validation stops
before any accepted event.

No CCA call, projection, nominal continuation, primitive generation or execution
occurred. Mean elapsed time to the blocker was 11.419 seconds for `gpt-4o` and
17.074 seconds for `gpt-5.6-sol low`; mean total tokens were 13,306 and 10,171.
The next work remains connecting declared parameter references and owner
capability calculations to saved facts, plus independent event physical projection
for CCA. Parameter applicability also needs correction: the saved `place_insert`
support includes `assembly_slot`, but the owning `release_part` accepts that
optional parameter only for `assembly_board-v1`; buffer placement must not demand
assembly-only slot values. This is a code-inspection finding, not a controller
test. This comparison does not establish those missing checks. The offline
guards recorded zero live calls, and the current UI run was not interrupted.
**User-observed Gazebo recovery and remaining assembly completion remain pending.**

### October 9: three actual offline attempts per model

`comparisons/20261009T164725Z/REPORT.md` and `combined_comparison.json` record
**three attempts each for `gpt-4o` and `gpt-5.6-sol low`**, including the earlier
pilot pair. The user requested two further attempts per model despite the shared
pilot evidence blocker. Only the separately saved comparison runner's trial range
and experiment stop condition changed. Application code, initial messages and
schemas, selected safety specifications, source observations and validation
limits remained unchanged. No recovery sequence or event-count target was added.

| Model | Attempts | Proposed / accepted events | Public retrieval turns | Mean generation time | Total tokens | Complete validated outlines |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-4o` | 3 | 3 / 0 | 0 | 6.975 s | 15,870 | 0 / 3 |
| `gpt-5.6-sol low` | 3 | 3 / 0 | 5 | 24.983 s | 39,935 | 0 / 3 |

All six attempts proposed one event on `recovery-resource-3@localhost` and
stopped at RA with `NEEDS_CONTEXT`. GPT-4o proposed
`recovery_pick_place_execution`, `align_gear_small` and
`transfer_misplaced_part_to_assembly` for `gear_small`. GPT-5.6 Sol proposed
`place` for `KET4_Square_4mm` at `Buffer For Machined parts` in attempts 1 and 3;
attempt 2 proposed `place_approach` toward `Exit` while retaining custody.
These are exact LLM proposals, not accepted recovery events or CCA verdicts.

The unresolved RA evidence set was identical: incomplete task programs and
primitive support, required owner evaluators, and incomplete parameter evidence.
No search exhausted the 4,096 limit and no primitive witness was derived.
Validator-driven retrieval returned saved model/snapshot declarations without
new evidence. Geometry projection, CCA, append/project and nominal continuation
were not reached. Outline safety is unresolved; execution safety is skipped.
Times and tokens describe progress to this blocker, with cached inputs in later
attempts. These results do not establish a winner for complete recovery.

All **17 provider requests and responses** were captured, without provider/parse
errors, timeouts or pending requests. The source hash and frozen application,
configuration and safety fingerprints remained unchanged. Guarded live-interface
and later-stage calls were zero; the current UI run was untouched.

**Automated verification:** continuation record checks passed for distinct runs,
input and source fidelity, exact proposals, provider parameters, complete request
outcomes and unresolved RA evidence. The prior 406 focused regression results
remain recorded separately; those suites were not rerun for unchanged application
code. The continuation runner compilation, Poetry checks, Python compilation and
diff checks passed; existing Poetry metadata warnings remain.
**User-observed Gazebo recovery and remaining assembly completion remain
pending.** The next evidence work remains exposing complete recorded resource-owned
task declarations to RA with provenance, followed by sufficient geometry,
attachment/motion and process-completion evidence for CCA and nominal continuation.

### October 9: actual gpt-4o / gpt-5.6-sol low comparison

`comparisons/20261009T162226Z/REPORT.md` records actual configured OpenAI API
attempts on the unchanged `1c3bec8faca6495cb1b0e22046277d5c` capture. GPT-4o omits
`reasoning_effort`; GPT-5.6 Sol sends `low`. Structured requests and summaries now
preserve that distinction, while saved historical settings remain unchanged.
Both models received identical initial messages and schemas with the two selected
safety specifications. Dynamic public retrieval remained enabled. No recovery
sequence or event-count target was supplied.

| Model | API calls | Public retrieval turns | Proposed / accepted events | Generation time | Total tokens |
|---|---:|---:|---:|---:|---:|
| `gpt-4o` | 2 | 0 | 1 / 0 | 6.317 s | 5,286 |
| `gpt-5.6-sol low` | 4 | 2 | 1 / 0 | 27.758 s | 15,015 |

GPT-4o proposed `recovery_pick_place_execution` for `gear_small` on
`recovery-resource-3@localhost`. GPT-5.6 Sol retrieved each robot's recorded
`static_capabilities` and proposed `place` for `KET4_Square_4mm` at
`Buffer For Machined parts`. Both stopped at RA abstract derivation with
`NEEDS_CONTEXT` and the same incomplete task/primitive declarations. The shared
missing complete task programs for `pick_approach`, `pick_grasp`, `place_approach`
and `place_insert` stopped further repetitions after the pilot pair. No supporting
primitive witness was derived. Geometry projection and CCA were not reached; no
event was appended and nominal continuation was not established. This is not a
complete validated recovery outline or a definitive model-quality ranking.

All six API requests and responses were captured, with zero provider/parse errors.
Source and code/configuration fingerprints remained unchanged; guarded live and
later-stage calls were zero. The actual initial grounding and outline messages
contained 4,570 and 11,426 characters respectively, excluding schemas. Outline
safety is unresolved, execution safety is skipped, and execution remains
unauthorized. The current UI run was untouched.

**Automated verification:** 165 model-parameter/request-record/offline tests and
241 context/validated-outline/geometry-evidence/workflow tests passed, totaling
406 focused tests. Comparison-summary checks, Poetry checks, compilation and
diff checks passed. Existing Poetry metadata warnings remain.

**Next evidence work:** expose complete recorded task declarations to the RA
validation path with preserved provenance, then obtain sufficient geometry,
attachment/motion and process-completion evidence for later CCA and nominal
continuation checks. Those missing facts were not fabricated. **User-observed
Gazebo acceptance and remaining assembly completion remain pending.**

### October 9: current offline comparison preparation and retrieval audit

The requested three-attempt `gpt-5.4 low` / `gpt-5.6-sol low` comparison is prepared
at `comparisons/20261009T153403Z/REPORT.md`, using the unchanged saved
`1c3bec8faca6495cb1b0e22046277d5c` capture and the current RA/CCA outline contract.
No new provider request has been sent: automatic approval review rejected the
external API disclosure, and explicit approval is pending. These preparation and
local replay records are not another completed model comparison.

PA still supplies the main facts, public capabilities and supported context
references; `need_context` retrieves recorded facts. The audit fixed projected
unknown part facts falling back to saved facts, duplicated/unsanitized retrieval
text, and omitted exact resource/custom part goals. The focused context,
validated-outline and request-record suites passed **231 tests**; Poetry,
compilation and diff checks passed. Original previews and observations remain
preserved.

Local replay of a previous exact LLM buffer-placement proposal, with its starting
state derived through the current runtime path, stops at RA with `NEEDS_CONTEXT`
for incomplete primitive/task declarations. CCA is not reached and no event is
appended. The capture also lacks the complete projected robot/tool/carried-part
and motion evidence required for later geometry and CCA checks. This is an
evidence limit, not a new model-comparison result. API generation, user-observed
Gazebo acceptance and remaining assembly completion remain unverified.

## 2026-10-09: RA-derived recovery outline evidence and CCA validation

New generation attempts use `validation_contract_version=2` and follow
**Capture facts → one LLM `next_transition` → ResourceAgent backward derivation
and forward validation → CCA validation → append and project → nominal
continuation validation**. Historical attempts retain their original contract
and status. The ProductAgent no longer supplies custody effects or ranks/inserts
recovery events on this path. The proposed event name remains an identifier;
the resource derives support from declared primitives and, when fully available,
complete task composition contracts. Search is bounded to 4,096 state/binding
combinations. Unsupported semantics, missing facts and incomplete searches remain
`NEEDS_CONTEXT`; a complete negative result or established contradiction is
`INFEASIBLE`.

The shared live/detached RA path records ordered prerequisites, bindings,
intermediate effects, model/source fingerprints and projected resource/part state.
Non-executing owner previews use an isolated recorded scene. Captures include
configured physical observations, joint configurations, tool/part geometry and
attachments when available. A held part is supported by actual attachment evidence.
Missing resources cannot disappear from the CCA population. FK configurations,
carried parts and deposited-part poses carry forward between events. A named-pose
endpoint does not prove an entire movement safe. Current native interfaces without
an isolated planning scene are not used as a substitute for future motion evidence.

CCA uses unchanged `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm` specifications. Each rule retains its AP
namespace and monitor history. Robot/tool/carried-part geometry determines
occupancy; `processCompleted` requires its complete recorded ledger or a supported
declared completion effect. Unknown occupancy is retained. Existing active CCA
history without a compatible recorded outline context blocks prospective reset.
Only a successful RA and CCA result commits the projected state and monitor state.
Nominal continuation must use the exact saved nominal task contract, respect
dependencies and discharge monitor obligations; a missing task index is not
completion.

Public `need_context` retrieval remains available to the LLM. Validator requests
are recorded separately and retry the same proposal only when new evidence is
available. Offline uses saved facts and local calculations, with no live interfaces.
Fresh initial observations cannot replace a predicted state after accepted events.
Per-turn records retain derivation, geometry coverage, retrieval exchanges, CCA
results and nominal-continuation evidence. The UI keeps the proposal separate from
these records and labels outline safety separately from execution validation.
Offline remains Manual. New outlines/proposals cannot enter the diagnostic
execution or saved-result Gazebo dispatch path.

**Automated results:** the combined outline integration, geometry, capture,
resource preparation, context, offline and workflow suites passed 579 tests. The
UI/request-record/setup suites passed 201 tests. RA focused checks passed 89 tests,
including 52 new derivation cases. The separate geometry/continuous-motion suites
passed 68 tests. `poetry check` and Python
compilation passed; Poetry reports its existing metadata deprecation warnings.
After the final public-feedback, retrieval-progress and stale-model fixes, the
five outline/context/offline/workflow suites passed 339 tests; the context/request
record suites passed 198 tests. These overlap the combined suites above and are
not an additional distinct-test total. Rejection prompts show exact violated rule
identifiers and AP values while omitting private models and primitive details.
Restricted-sandbox asyncio callbacks stalled even in a minimal standard-library
reproduction; mocked asynchronous suites ran successfully outside that sandbox.
The broader RA suite also reported 12 failures before/in legacy AP and KMR provider
fixture paths, separately from the new derivation tests: five slash-AP preparation
failures and seven unavailable KMR prepared-execution provider failures.

The original `1c3bec8faca6495cb1b0e22046277d5c/observations.json` SHA256 remains
`d92fe0228978431eaf2343d26f7c7fbe47322b9a81dc4cad3f713850d04110b2`.
That older capture is not automatically sufficient for this milestone: absent
joint/attachment/motion evidence and incomplete native completion contracts remain
explicit blockers. **User-observed acceptance:** pending through the UI. No live
LLM request, Gazebo launch, recovery execution or current-run restart was performed
for this implementation.

### Subsequent runtime work — not implemented by this milestone

- Validate complete primitive compositions against the accepted outline evidence.
- Check recovery × recovery and recovery × nominal before dispatch and against
  observed execution; retain shared CCA history and resource/region ownership.
- Bind execution permission to exact commands, models and observations through
  recovery and resumed nominal work.
- Handle stale evidence, deviations, failure and Stop with blocked admission,
  controller cancellation, and observed completion or clearance.
- Add concurrent execution only with adequate combined motion and timing evidence.

**Historical implementation, 2026-10-08:** the Recovery page follows **Example →
Safety → Recovery outline → Primitive composition →
Execution**. Details and its legacy controls are no longer rendered; existing
records remain. Example offers Gazebo / Offline, the four unchanged scenario
names, and Start failure with Part slippage initially selected. All four examples
use their existing failure checkpoints and retain observations; recovery
generation and execution currently support Part slippage. One Recovery mode
Auto/Manual control in Example applies to both Recovery outline and Primitive
composition. Gazebo defaults to Auto and stops before execution; Offline always
uses Manual, with the shared control disabled. Manual retains the Generate/review
steps in each stage. SystemBridge remains the public interface; background
workflow ownership survives page navigation. The Current observations card is
removed. Example retains the capture status, source run, capture time and
Download observations; Retry input preparation appears with the outline error.

Offline follows **Select saved observations → Generate → Review**, without a
Start offline test button. Captures appear newest first by capture time. Selecting
one prepares an input preview marked **not sent** without calling the LLM. Outline
Generate is available after preparation; disabled controls show the actual reason.
Offline pauses for review and Continue after the outline, then permits primitive
Generate. Each explicit Generate completes that stage's retrieval turns. Switching
back to Gazebo restores its previous mode selection. Stop followed by Generate
creates a separate attempt from the selected capture; obsolete preparations are
cancelled when the selection changes. Historical run status does not own work.

This first functional experiment explicitly uses `execution_mode="simulation"`,
`diagnostic_cca_bypass=true`, `runtime_recovery_validation_policy="no_validation"`
and no active safety requirement. Part slippage automatically selects
`safety_assembly_board-v1_mutex.txt` and
`safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt` in the Safety
panel. These selections remain editable metadata for later integration without
activating checks or adding them to prompts. Formal safety
and CCA admission are **skipped**. Offline validation retains schema, grounding,
declared transitions, capabilities and primitive state projections. Fresh
controller and trajectory checks are **deferred**, not passed. Test in Gazebo
checks source identity, configuration, unfinished tasks, custody, fresh poses,
controller readiness and primitive feasibility before enabling Execute recovery;
the explicit execution click rechecks imported proposals. Mismatches require
regeneration and are not silently rewritten. Simulation/controller guards reject
hardware execution. Confirmed primitive completion precedes nominal resumption.

Observations are saved before preparation or LLM calls. Preparation uses
registered `executables` and saved task definitions; errors leave the capture
available for download and retryable. Complete captures contain the prepared context, resource
models, primitive catalogs, configuration, geometry, task definitions and goals
needed by both stages. Offline attempts use detached saved resources, with no
Gazebo, controller, perception or SPADE transport calls; LLM generation still
uses the configured API and existing templates. Each attempt has separate
artifacts under `cais_spade_llm/monitor/environment_runs/recovery/<run_id>/`.
Exact request/tool/response captures, prepared input marked not sent and original
downloads appear in chronological turns. Each Input and Output expands independently
and loads its content on demand. Expansion choices survive polling, navigation
and reconnection, scoped to the run, generation, stage and turn. Retries and
provider tool rounds remain grouped; primitive turns from different resources
remain separate even when their local numbers repeat. Legacy summaries are
identified explicitly when exact captures are unavailable. Load saved result restores a
completed composition after application restart without another LLM call.

New outline sessions use `decision="need_context"` without committing candidate
events. The initial prompt selects observed failure, custody, unfinished goals
and interacting work from recorded relationships, and keeps alternative resources
discoverable through a compact capability index. Detailed public declarations,
poses, target geometry, pending tasks and primitive contracts are retrieved by
supported references. Saved facts are used first; offline retrieval never calls
live interfaces. Missing facts and repeated requests without new information stop
with `NEEDS_CONTEXT`. Full captures and validator context remain intact. Each turn
records context requests, returned facts, missing facts, inclusion reasons and
provenance separately from the authored proposal. Live revision changes invalidate
dependent generation; accepted transitions recompute projected prompt facts.

**Automated verification for dynamic retrieval and expandable turns:** 143 focused
tests passed, plus 11 existing prompt/schema/context/retrieval regressions. Poetry
and Python compilation passed. Focused Ruff checks passed for the new context
module and affected UI modules; broader checks of the existing test files retain
seven preexisting style findings. No live model call or equipment action was used.
On the same saved capture, outline prompt text decreased from 10,193 to 6,409
characters and grounding text from 6,358 to 3,367. These are offline rendering
measurements, not model-quality or live recovery results.

**Previous offline verification:** 118 focused workflow, offline, page, request-record
and diagnostic tests passed. Another 359 setup, saved-program, mocked Gazebo and
environment-admission tests passed. The separate feasibility suite completed
with 46 passed and 12 failed: five legacy AP fixture failures and seven KMR
prepared-execution fixture failures, outside the new offline path. Poetry,
compilation, UI CLI help, focused Ruff checks and diff checking passed.

**User-observed Gazebo acceptance: pending.** No live LLM generation, Gazebo
launch or recovery execution was initiated for this update. Earlier trials
observed both pickups and `gear_small` slippage and generated some outlines, but
did not execute generated recovery primitives or establish final assembly.
Older `environment_run.json` files remain viewable. In particular,
`aa8e27d245584008b593c84111daa47e` lacks a prepared recovery request, so it reports
`NEEDS_CONTEXT` for offline generation and requires a fresh complete capture.
The next gate belongs to the UI operator: capture failure → stop Gazebo →
generate both stages offline → expand compact inputs and retrieval outputs →
Test in Gazebo → explicitly Execute recovery →
observe recovery and remaining assembly completion, with safety checks deferred.

**Earlier status reviewed on 2026-10-08 (CCA integration):** nominal and recovery physical admission now
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

The earlier CCA implementation grounds the existing CCA path for nominal and
recovery execution under explicit motion and failure-model assumptions, preserving
the Gazebo plant and actuator behavior. CCA must still evaluate intermediate motion,
include relevant nominal work, bind grants to exact prepared commands, and retain
observed monitor history. Missing model inputs remain `NEEDS_CONTEXT`. Model-based
admission does not assert proven bounds on every physical deviation, contact or
stopping motion. Use background Gazebo on school WSL. Actual recovery-event
generation for the four failure scenarios follows validation of this CCA path;
automatic nominal resumption and full physical guarantees were later milestones
for that CCA work. The functional Part slippage workflow above is the current
implementation scope.

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
| Diagnostic UI | Recovery page has Gazebo / Offline, four failure scenarios, recorded Safety selections, visible observations, independent Auto/Manual generation stages and exact prompt inspection. Complete saved captures support generation without equipment or SPADE transport; each attempt has independent records and completed results can be loaded after restart. Test in Gazebo checks fresh compatibility before explicit execution. Details is removed; existing `scenario_runner` records and backend diagnostic tools remain. | Operator acceptance: capture failure, stop Gazebo, generate both stages offline, return to Gazebo, explicitly execute and observe remaining assembly completion. Generation/execution for the other three scenarios and active Safety integration remain later work. |
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


**2026-10-08 simple Part slippage UI and operator handoff:**

The Recovery page exposes Start part_slippage, Stop, Reset example, independent
Auto/Manual outline and primitive composition modes, Input/Output inspection,
feedback/Regenerate, and explicit Execute recovery. Existing diagnostic tools
and saved evidence are under Details. Runtime ownership survives page navigation;
Auto ends at operator execution review. Exact request/tool/response captures and
original downloads remain separated by run and generation. Manual mode exposes
the captured observations and prepared context before generation.

The selected example scopes formal-safety deferral to simulation, no selected
safety requirement, diagnostic_cca_bypass=true and no_validation. Both nominal
and recovery paths record safety as skipped. Controller feasibility, program
ownership, cancellation, custody and observed completion checks remain. The
original two specifications and their later Safety integration remain separate.

Live trials observed both pickups and gear_small slippage during ur5e-4 lowering.
They exposed missing resource dispatch, pose-planning and primitive-parameter
evidence, which were corrected with resource/controller-owned evidence. The
interrupted place_approach name, reconciled part observations and saved task
contracts now enter the prompt. Retained part assembly precedes cached resource
cleanup in the unfinished-task context. No recovery sequence or default operator
guidance is supplied to the LLM. Existing prompt templates remain in use.

At the user's request, agent-driven live trials stopped so the next test can be
performed visibly through the UI. No generated recovery primitives were executed.
The last run is 74f28bc26fb74c09a28c472b8c164b8e; earlier outline captures include
0a5711b46c3b4c1e85b19a08a7f2e4c9 and 04c97010cc70411aadf5635171d87171 under
cais_spade_llm/monitor/environment_runs/recovery/. Stopped and incomplete runs
remain retained. These runs establish the failure checkpoint and parts of outline
generation, not successful full recovery or remaining assembly completion.

Checks: 238 workflow/UI/fault/setup/saved-program tests passed before the final
context/UI changes. The updated workflow/UI group passed 63 tests; the subsequent
factual-context preparation test passed separately. Seven focused feasibility
regressions passed, including inbox survival after controller errors. Poetry,
Python compilation, UI CLI help and diff checking passed. Existing unrelated
legacy Safety fixture incompatibilities remain outside this deferred-safety task.

Next acceptance gate: the operator inspects captured observations in Manual mode,
generates/reviews the outline and primitive composition, executes once, and
observes both primitive completion and the remaining assembly. This is functional
Gazebo recovery with safety checks deferred. Existing offline diagnostics can use
compatible saved runtime_context bundles; export of the new live capture format
into that bundle is still required for a direct offline rerun.

**2026-10-08 visible observations, offline generation and later Gazebo execution:**

This update supersedes the preceding entry's Details layout and offline-export
limitation. The page order is Example, Safety, Current observations, Recovery
outline, Primitive composition, Execution. Gazebo and Offline are explicit
sources. The scenario names remain Part slippage, Conveyor breakdown, ur5e-1
breakdown and Machining breakdown during part processing. The Part slippage
preset retains KET4_Square_4mm upright in Buffer For Machined parts zone 4, both
observed pickups, and gear_small slippage during ur5e-4 placement lowering. The
other scenarios use their existing M1 checkpoints.

Reconciled failure facts are written to `observations.json` before prompt
preparation. Registered `executables` replace the invalid `function_names`
access. Errors preserve the observations, display their actual cause and allow
preparation retry. A successful capture adds the saved preparation context;
offline sessions never query live agents, controllers, perception or Gazebo.
Unknown required facts remain NEEDS_CONTEXT. Existing templates and request
writers are reused without a suggested recovery sequence or default guidance.
Prepared input is marked not sent; exact requests, tool exchanges, raw responses
and resulting outputs remain inspectable and downloadable. Recorded observations
retain their provenance and timestamps through grounding.

Every offline attempt has its own directory linked to immutable source facts.
Outline regeneration invalidates primitive composition. Primitive regeneration
preserves the outline checkpoint, and Stop/Reset preserve completed saved
results. Load saved result restores them after application restart. Test in
Gazebo can reuse the owned paused failure or rerun the example; it checks the
saved proposal against current identities, configuration, task state, custody,
fresh poses, controller readiness and primitive feasibility. A mismatch blocks
execution without rewriting the proposal. Execute recovery remains an explicit
action and repeats imported-proposal checks. Safety selections are metadata
only; formal safety is skipped, and offline physical checks are deferred.

Automated results are separate from live acceptance:

- 118 tests passed in `test_offline_recovery.py`,
  `test_part_slippage_recovery.py`, `test_project_pages.py`,
  `test_recovery_request_records.py` and `test_recovery_diagnostics.py`.
- 359 tests passed in the setup, saved-program, mocked Gazebo and
  environment-admission suites. These use test fixtures, not a live trial.
- `test_recovery_feasibility.py` completed with 46 passed and 12 failed. Five
  fail because legacy slash AP descriptors are rejected during fixture
  preparation. Seven KMR tests lack the current prepared execution provider.
  Those paths and their fixed symbols were not changed by this update.
- `poetry check`, `poetry run python -m compileall -q cais_spade_llm ros2`,
  `poetry run python -m cais_spade_llm.ui_main --help`, focused Ruff checks and
  `git diff --check` passed. Poetry reports existing metadata deprecations.
  Async regression tests ran outside the filesystem sandbox because its
  asyncio executor shutdown hung; LLM and equipment calls remained mocked.

Evidence remains under `cais_spade_llm/monitor/environment_runs/recovery/`.
The existing `aa8e27d245584008b593c84111daa47e/environment_run.json` retains
failure facts but lacks the prepared recovery request; it is viewable and is
explicitly incomplete for offline generation. A fresh capture is needed.

No live LLM request, Gazebo launch, generated recovery execution or assembly
completion was performed for this update. The user-observed acceptance gate
remains: Start failure → inspect/save complete observations → stop Gazebo →
Start offline test → inspect both stages → Test in Gazebo → Execute recovery
once → observe primitive completion and remaining assembly. Record that later
result as functional recovery with safety checks deferred.

**2026-10-08 dynamic context retrieval and expandable recovery turns:**

New sessions preserve complete saved observations and prepared recovery context
while selecting prompt facts separately. Selection follows failure, goal, custody,
shared pending-work destinations and explicit predecessor/successor relationships;
it does not use scenario-specific resource lists or insert a recovery sequence.
Alternative resource identities and declared capabilities remain visible even
when their detailed state is outside the initial context. Public field declarations
are available through `/resources/<jid>/state_variables`. The same allowlist
provides resource/part discovery, snapshots, recorded poses, target geometry,
pending tasks and declared primitive catalogs/contracts. Private models and
evaluator data are excluded.

Outline `need_context` stays in outline generation, commits no event and returns
recorded facts with source, capture time and revision. Missing live part poses use
the existing grounding path; missing resource poses use the owning
`get_recovery_snapshot` interface and require a feedback capture time. Offline
retrieval performs no live calls. Missing required context remains explicit;
repeated unchanged requests stop with `NEEDS_CONTEXT`. Candidate generation still
uses the existing validation and selection path with complete context and the
current diagnostic policy. Previously saved sessions and the current live process
were not migrated, restarted or interrupted.

The page removes Current observations and replaces stage tabs/artifact dropdowns
with independently expandable Input/Output for each turn. It preserves exact
messages and original-file downloads, keeps runtime retrieval/validation separate
from LLM output, and marks prepared previews as not sent. Indexing reads bounded
headers/tails; full payloads load only on expansion. Browser expansion choices are
scoped to run/generation/stage/turn. Provider retries/tool rounds are grouped with
their call; legacy primitive streams with repeated local turn numbers remain
distinct. Capture download and Retry input preparation retain evidence access.

Automated evidence for this update:

- 143 tests passed across `test_recovery_context_retrieval.py`,
  `test_recovery_request_records.py`, `test_project_pages.py`,
  `test_part_slippage_recovery.py`, `test_offline_recovery.py` and
  `test_recovery_diagnostics.py`. Coverage includes independent Auto/Manual modes,
  regeneration, Stop/Reset, offline reload, observation ownership, stale revisions,
  missing/private references, exact captures and independent lazy expansions.
- 11 existing tests passed in `test_case3_recovery_dryrun.py` selected by
  `prompt or schema or context or retrieval` (54 deselected). Tests used mocks;
  no provider requests, Gazebo launches or controller executions were initiated.
- `poetry check` and `poetry run python -m compileall -q cais_spade_llm ros2`
  passed. Poetry reports existing metadata deprecations. Focused Ruff checks
  passed for the new context module, affected UI modules and new test module.
  Broader lint inspection found four existing import-order findings in
  `test_project_pages.py` and three existing nested-with findings in
  `test_recovery_request_records.py`; these are unrelated to this feature.
  Previously recorded feasibility failures were not rerun or claimed fixed.
- Using the unchanged `e2392568bec34aabb32fb29552ada692/generation_0001/inputs.json`
  under `monitor/environment_runs/recovery/`, toggling context selection on the
  same initial session reduced grounding text **6,358 → 3,367** and outline text
  **10,193 → 6,409** characters. The source file was not modified. The legacy
  `recovery_outline/worked_previous/multi_turn_turn03_outline_request_20260802T145141.txt`
  has **4,682** characters of prompt text (trimmed message body) and **11,210**
  characters in its complete request summary including schema and metadata.
  This legacy example has two resources; the saved current capture keeps twelve
  resource identities discoverable. These comparisons exclude response schemas
  from prompt-text counts and do not establish recovery quality.

**User-observed Gazebo results for this change: pending.** The user retains control
of live generation and execution. After loading the updated UI for a new session,
inspect the compact input, observe any `need_context` turns, expand their outputs,
review the outline and primitive composition, then use the explicit Gazebo testing
and execution buttons. Record observed primitive completion and remaining assembly
separately from these automated results; formal safety remains skipped.

**2026-10-08 Part slippage safety selection defaults:** the existing mutex and
`gear_small` before `KET4_Square_4mm` specification files are preselected for a new
Part slippage example and added when switching back to Part slippage. Explicit
manual deselection is retained across polling and source changes. The UI reflects
the coordinator's selections without starting work. `selected_safety_file` remains
`__NONE__`; the diagnostic validation policy and skipped formal safety status are
unchanged. Saved records and the current live run were not modified.

Automated verification: 95 tests passed across `test_part_slippage_recovery.py`,
`test_project_pages.py` and `test_offline_recovery.py`, including scenario-switch
selection, manual override, UI synchronization and deferred-policy checks.
`poetry check`, Python compilation, focused Ruff checks and `git diff --check`
passed; Poetry retains its existing metadata deprecation warnings. No live
Gazebo or LLM action was initiated. User-observed verification remains pending:
load the updated UI for a new example and inspect the two Safety selections.

**2026-10-08 Safety panel selection visibility:** fixed the mismatch between
relative default selection paths and absolute paths returned by SystemBridge.
The dropdown keeps the recorded selection strings as option keys for matching
files and displays complete filenames as selected chips, including after
scenario changes. Adding defaults retains an already selected absolute path
without duplicating its relative counterpart. Manual deselection remains valid.

Automated verification: 74 tests passed in `test_part_slippage_recovery.py` and
`test_project_pages.py`; the UI suite was rerun after extending synchronization
to scenario changes. Poetry, compilation, focused Ruff and diff checks passed.
These are mocked checks; live UI verification remains pending. The app uses
`reload=False`, so changes require the next UI process restart. The ongoing run
was not restarted or modified, and formal safety remains skipped.

**2026-10-08 combined recovery mode:** replaced the separate outline and primitive
composition Auto/Manual toggles with one Recovery mode control in Example. Each
explicit change uses the existing SystemBridge operation to set both stage modes
to the selected value. Auto remains the default; Execute recovery remains an
explicit action. The per-stage runtime settings and saved records stay compatible.
Opening or reconnecting the page does not change modes or start work. Older runs
with different stage modes show their recorded values without rewriting them.

Automated verification: 99 tests passed across `test_project_pages.py`,
`test_part_slippage_recovery.py` and `test_offline_recovery.py`, including the
shared control for Gazebo/Offline, both mode selections, reconnection and existing
mixed-mode records. Poetry, compilation, focused Ruff and diff checks passed.
Live UI verification is pending the next UI restart; no live runtime was restarted
and no model or equipment action was initiated.

**2026-10-08 direct generation from saved observations:** SystemBridge now exposes
`prepare_offline_recovery`, retaining the existing start and saved-result entrypoints.
Preparation and generation remain owned by the coordinator/session. Capture
readiness is derived from recorded preparation inputs, with actionable missing
context, rather than trusting the persisted `offline_ready` flag. All recorded
resource identities and capabilities remain available for discovery. A recovery
model and primitive support are required when validation uses that resource;
missing evidence remains `NEEDS_CONTEXT`. Empty prepared pose fields no longer
hide measured poses in the same capture. Recorded poses seed the existing
grounding store with saved provenance, separately from projected states. The
requested drop pose is never substituted for a measured pose.

Automated verification: **147 tests passed** across `test_offline_recovery.py`,
`test_part_slippage_recovery.py`, `test_project_pages.py`,
`test_recovery_context_retrieval.py` and `test_recovery_request_records.py`.
Coverage includes direct Generate in both shared modes, all four retained stage
mode combinations, explicit execution, twelve resources including missing recovery
models, null prepared poses, chronological ordering, missing-context explanations,
no-call preparation, duplicate clicks, cancelled/late preparation, Stop/new attempt,
Reset, reconnection, saved-result reload and exact request captures. Offline tests
reject live interface calls and verify immutable sources and separate artifacts.
Poetry and Python compilation passed; Poetry retains its metadata deprecation
warnings. Focused Ruff checks passed for the affected runtime/UI/context modules
and the offline/context tests. Existing unrelated test-file lint findings remain
separate from those checks.

A preparation-only check used the actual capture
`1c3bec8faca6495cb1b0e22046277d5c/observations.json`, captured
`2026-10-08T20:46:46.864548+00:00` (October 8, 4:46:46 PM EDT).
It reached `outline_ready` with twelve resources, retrieved the exact saved
`gear_small` measured pose and produced a **4,030-character grounding preview**
(prompt text only, excluding schema/metadata). Artifacts were isolated in a
temporary directory; the source file hash remained unchanged. The LLM entrypoint
was configured to fail if called. No Gazebo, controller or SPADE operation was
requested. This verifies preparation and retrieval, not recovery quality or motion.

**User-observed results for direct offline generation: pending.** Load the updated
UI after the current run is finished; the app uses `reload=False`. Select Offline,
select the October 8 capture, inspect the not-sent input, and click Generate.
Inspect dynamic retrieval turns and review both generated stages before using
Test in Gazebo and the explicit Execute recovery button. Observe primitive
completion and remaining assembly completion separately. Formal safety remains
skipped. The current live run was not restarted or interrupted by this change.

**2026-10-08 Offline Manual policy, capture cleanup, and outline schema fix:**
Offline mode is enforced as Manual in the coordinator, offline runtime factory,
session progression and UI. Legacy entrypoints prepare an input without starting
generation; explicit generation, regeneration and saved-result loading retain
Manual review boundaries. Attempts to enable Auto through the mode operation
are rejected. Gazebo mode preferences survive source changes; existing request
artifacts retain their original settings.

The user-observed attempt `offline_4d5bd6209de040f5a5b594901a66c3c3` completed
grounding and then failed at Turn 2 with HTTP 400. The retrieval extension had
introduced a top-level `anyOf` in `multi_turn_outline_candidates_response`,
which the provider rejected. The provider schema now retains a root object,
the existing fields and `strict=false`. Conditional retrieval/candidate requirements
are checked locally before dispatch, retaining existing candidate validation.
Mixed responses, missing required candidate fields and malformed context requests
cannot commit events or advance to primitive composition. API failures mark the
affected stage failed and retain `stage_error.json`; explicit retries create a
separate generation directory.

The authorized one-time cleanup retained only
`1c3bec8faca6495cb1b0e22046277d5c/observations.json` and removed `observations.json`
from `e2392568bec34aabb32fb29552ada692`,
`offline_5fd94b310bba42dca8518a6003570e00` and
`offline_4d5bd6209de040f5a5b594901a66c3c3`. All 31 other files in those directories
were unchanged, including the failed attempt's provider captures. The retained
capture SHA-256 is `d92fe0228978431eaf2343d26f7c7fbe47322b9a81dc4cad3f713850d04110b2`.
The dropdown lists actual `observations.json` files; legacy `environment_run.json`
reading remains available through the existing reader. No automatic retention
policy was added; future records are retained.

Automated verification: **175 focused tests passed** across the offline recovery,
Part slippage runtime, project pages, dynamic context retrieval and provider
request-record suites. Added cases cover the provider schema boundary for four
outline modes and three action horizons, local conditional requirements, forced
Manual behavior, Gazebo mode restoration, saved-result compatibility and preserved
errors across outline/primitive retries. Poetry checks and Python compilation
passed. Focused Ruff checks passed for the recovery/UI/context helpers and
offline/context tests. Separately, Poetry retains metadata deprecation warnings,
and `test_recovery_request_records.py` retains three pre-existing `SIM117` findings.
The sandbox test run stalled in thread handling; the completed suites ran outside
the sandbox with mocked provider/equipment interfaces.

A separate replay used the retained full capture and the exact recorded grounding
response, with all outputs isolated in a temporary directory. A mocked provider
accepted one grounding and two outline request schemas; outline retrieval returned
the exact measured `gear_small` pose with saved provenance. The deliberately
repeated context request terminated at `NEEDS_CONTEXT`, confirming the no-progress
boundary. This verifies schema construction, request capture and saved-fact
retrieval, not a newly generated recovery proposal. The source hash was unchanged
and no live API, Gazebo, controller or SPADE operation ran.

**User-observed verification after the fix: pending.** At the next UI restart
(`reload=False`), select Offline and the retained capture, confirm Manual, then
Generate the outline. Review it, Continue, and explicitly Generate primitive
composition. Live API acceptance, proposal quality and any subsequent Gazebo
execution remain to be observed using the user's buttons. Formal safety remains
skipped; no live run was restarted for these changes.

### 2026-10-08: Saved observations selection and repeated entries

Reproduced a delayed page poll restoring the previous Gazebo/Offline selection
after the operator selected Offline and a capture. Poll completion now reads the
current bridge state and discards files from an obsolete run or generation.
Saved-observation options restore the current runtime selection, including a
previously selected offline copy, without starting preparation during refresh.
Unchanged options are no longer rebuilt by background list reads.

Offline attempts retained copies of the same input, so one captured failure
appeared three times. The list now groups equal observation content using the
existing observation fingerprint, prefers the original run's file, and retains
links to completed offline results. Different recorded facts remain separate.
No observation or diagnostic file was deleted or rewritten for this fix.

**Automated verification:** 123 tests passed across `test_project_pages.py`,
`test_offline_recovery.py`, and `test_part_slippage_recovery.py`. Coverage includes
an in-flight stale poll, selection restoration, unchanged-option refresh,
duplicate copies, changed facts, an unavailable original, and saved-result loading
through the original capture. `poetry check`, Python compilation for
`cais_spade_llm` and `ros2`, and focused Ruff checks for changed runtime/UI modules
and offline tests passed. Existing Poetry deprecation warnings and four existing
import-order findings elsewhere in `test_project_pages.py` remain separate.
A read-only check of the actual retained files returned one dropdown entry from
three copies of the October 8 capture; all three file hashes were unchanged.

**User-observed verification: pending.** Apply at the next UI restart
(`reload=False`), then select Offline and the capture once. No live generation,
Gazebo execution, or running-UI restart was performed for this fix.

### 2026-10-08: Shorter prompts and LLM proposed events

Each turn's Output now begins with the events from the captured LLM response,
in their original order, including rejected candidates. The table shows exact
event, resource and part identifiers. The full response follows with readable
JSON formatting. Runtime selections remain in a separate, initially collapsed
Validation records section; Returned context is independently expandable.
Retries, original downloads, missing captures and legacy parsed-response labels
are retained. Expansion choices survive refresh and reconnection and are scoped
to the run, generation and turn. Runtime-selected events never fill a missing
LLM proposal table.

The saved re-pick rejection was traced to `held_part.domain=[null]`, which reflects
captured empty custody rather than all future part identities. Successor matching
now permits preserved `held_part` when its declared `equals_from_param="part_name"`
guard matches the grounded task `part_name`. Conflicting parameters, other guards,
closed location/field domains and missing primitives remain checked. Progress
rules are unchanged: a custody change alone still does not qualify.

The exact recorded Turn 3 from `offline_5a7481bb4f4c4724b9f8a4a4979f6f35` was
replayed with live calls disabled and temporary output. It now selects the
recorded `recover_pick_gear_small` proposal because the saved `place_approach`
continuation is supported. The other two proposals remain rejected. This is a
saved-candidate test, not a new LLM proposal or executed recovery. Physical checks
remain deferred, safety skipped and execution unauthorized.

Prompts retain essential facts, all resource alternatives and previously retrieved
context while removing repeated display content. They distinguish symbolic
locations, measured poses and projected states. Feedback reports recorded reasons
instead of encouraging a larger claimed `expected_end_state`. Brief `thought`
and `rationale` instructions do not truncate responses or alter response schemas.

Measured message-body lengths from the same saved run, excluding response schemas:

| Turn | Before | After |
|---|---:|---:|
| 1, grounding | 4,030 | 4,079 |
| 2, outline | 6,645 | 5,628 |
| 3, outline | 9,822 | 7,474 |
| 4, outline | 11,478 | 9,332 |

Turn 1 adds 49 characters for concise-response guidance. All before sizes match
the original captured messages. All 32 archived generation files remained
hash-identical during the comparison, and the original observations retain
SHA-256 `d92fe0228978431eaf2343d26f7c7fbe47322b9a81dc4cad3f713850d04110b2`.

**Automated verification:** the combined project-page, offline recovery,
Part slippage runtime, context retrieval, request capture and feasibility suites
finished with **253 passed and 12 existing failures**. The same 12 failures were
reproduced with the pre-change successor function: five involve legacy safety AP
test data and seven involve the unavailable KMR prepared-execution test provider.
They are not reported as passing. Additional focused legacy prompt/replay checks
passed; these counts overlap the context suite and are not added to the total.
Poetry checks and Python compilation passed. Focused Ruff checks passed for the
UI, context and feasibility helpers and offline/context tests; existing prompt
complexity/import-style findings and Poetry deprecation warnings remain separate.

**User-observed verification: pending.** At the next UI restart (`reload=False`),
start a new Offline generation. Confirm the proposed-events table, shorter inputs,
collapsed validation/context records and Manual review boundaries. No live LLM
call, Gazebo execution, capture rewrite or running-UI restart was performed.


### 2026-10-08: Offline model comparison with selected safety requirements

Offline factories accept per-attempt `model` and `reasoning_effort` overrides.
Each attempt records the effective settings and their sources without changing
the captured settings, original observations, or UI defaults. Offline remains
Manual. Selected safety files are captured once for the attempt, with exact
requirement identifiers, text, source paths and hashes; both grounding and outline
prompts include those requirements. This supplies planning constraints only:
formal safety remains skipped, physical validation deferred, and execution is
not authorized by prompt compliance.

An initial diagnostic comparison exposed a retrieval interface gap: the models
requested `place_approach` and `place_insert` through primitive contract references,
while their saved declarations were task contracts. Public task discovery and
`/task_contracts/<jid>/<event_name>` now return recorded guards, effects, parameter
bindings and primitive support. Unsupported/private portions are omitted and
incomplete results explicitly marked `partial`; the complete saved context remains
available to the validators. Missing declarations do not trigger live observations.

The first factual prompt revision states the supported references, the 1-to-8 request
limit, and the existing rule that `need_context` excludes candidate fields even
when empty. It distinguishes task contracts from primitive contracts. No event
sequence or target event count was added. Two redundant whole-request copies were
removed from validation; detached output facts and all checks remain intact.
The second factual prompt revision explains the existing successor-matching rule:
a generated event need not execute the registered task next or require its nominal
predecessor now. Guard, domain, binding and primitive-support checks are unchanged.
Repeated retrieval now compares facts and errors by reference and revision, so
reordering already-answered requests cannot create a no-information loop. New facts
and refreshes after accepted transitions remain allowed.
A replay of the saved input produced identical projections, with ten projection
calls taking 0.3352 seconds before and 0.0031 seconds afterward. This local
microbenchmark is separate from API or complete-stage timing.

**Automated verification:** 253 focused recovery, offline, UI, context-retrieval
and provider-capture tests passed. The suite includes per-attempt override
precedence, safety snapshots, exact identifiers, private-field exclusion, explicit
missing declarations, offline calls blocked, and validation output independence.
`poetry check`, Python compilation of `cais_spade_llm` and `ros2`, and
`git diff --check` passed. Existing Poetry metadata deprecation warnings remain.

**Completed API comparison (prompt revision 2):** three attempts each used
`gpt-5.4 low` and `gpt-5.6-sol low`. The configured account did not expose the
`gpt-5.6` alias, so its documented target `gpt-5.6-sol` was recorded explicitly.
Results are under `comparisons/20261009T031500Z/REPORT.md`. `gpt-5.4` produced one
runtime-accepted outline in three attempts; `gpt-5.6-sol` produced zero in three.
All six initial inputs were identical. Original observations remained hash-identical,
and guarded live-interface calls remained zero.

The accepted outline has two LLM events, `move_home` and `recover_place_insert`.
The latter claims the combined recovery and assembly effect. Its recorded support
covers the `place_insert` successor; pickup, prerequisite establishment and the
complete part effects still require primitive composition. Clearance of the shared
assembly area before the other robot's remaining work is also unestablished.
This is recorded as a runtime-accepted outline, not demonstrated recovery.

The blocked attempts exposed misunderstanding of the recorded
`held_part.domain=[null]` despite permitted grounded parameter bindings, repeated
attempts to bind the nominal pickup's null symbolic source, and malformed mixed
candidate/context responses. The prior 12 documented unrelated feasibility-suite
failures were not covered or claimed fixed by the focused test result above.

**User-requested legacy inspection:** the saved August run used `gpt-5.4 low`;
its first outline message was 4,683 characters, versus 6,702 in revision 2.
Its five-event outline contains three LLM events and two `robot_task_program`
continuations. RA/CCA interfaces are marked mocked; actual model-based workspace
rejection is recorded. `primitive_program_complete=false` prevents treating that
result as physical recovery. The recorded scenario, field declarations and safety
texts differ from the October capture. Detailed evidence is in
`comparisons/20261009T031500Z/legacy_comparison.json`.

After the user explicitly clarified that further factual prompt changes are
allowed, revision 3 clarifies recorded custody versus permitted later bindings,
provides compact affected-resource field declarations initially, and distinguishes
candidate responses with null/empty retrieval markers. No preferred event,
resource choice, recovery sequence or event-count target is supplied.
Revision 3 automated verification: **284 focused tests passed**, including
exact neutral candidate markers, continued rejection of mixed retrieval/candidate
responses, source immutability, privacy, and agreement between initial declarations
and requested declarations. Poetry checks, Python compilation and diff checks
passed. The first outline message is 8,364 characters: essential declarations and
binding semantics add 1,662 characters; this is not claimed as a prompt-size
reduction. The matched API run is recorded separately under
`comparisons/20261009T033800Z`.

**Completed API comparison (prompt revision 3):** all six attempts used identical
initial messages and response schemas, the two selected safety requirements, and
unchanged validators and generation limits. Results and exact provider artifacts
are linked from `comparisons/20261009T033800Z/REPORT.md`.

**October 9 correction:** the two runtime-accepted outlines below conflict with
recorded workspace evidence. `gear_small` is 0.799839 m from the recorded
`ur5e-4` reach origin, beyond its 0.71 m bound including tolerance. Both outlines
assigned acquisition to that resource. Retrieval omitted `gripper_reach`, and
offline validation deferred a check that could be performed from saved facts.
These historical acceptance counts do not establish supported recovery outlines.
The independent replay is saved as
`comparisons/20261009T033800Z/recorded_workspace_review.json`; original comparison
and provider records are preserved. The square-peg proposals rejected in other
attempts retained custody during `place_approach`; no complete put-down was
proposed and rejected by those records.

| Model, effort | Runtime-accepted outlines | Mean generation time | Mean total tokens |
|---|---:|---:|---:|
| `gpt-5.4 low` | 0/3 | 123.3 s | 72,203 |
| `gpt-5.6-sol low` | 2/3 | 91.3 s | 65,237 |

The two accepted outlines contain three LLM events: `move_home`, `recover_pick`,
then `recover_place_insert` or `recover_place`. The prompt supplies neither those
events nor an event-count target. Projected custody changes from empty to
`gear_small` to empty. These remain outline proposals: primitive composition,
physical feasibility, and shared-area clearance before remaining assembly are
unproved. All four blocked attempts accepted only `move_home`, then treated the
missing symbolic pickup location as requiring more context despite the measured
pose and generated-successor support. Response-format failures did not recur.
All failures remain recorded. Three attempts on one capture do not establish
general reliability or demonstrate recovery for other failure scenarios.

The user also requires applicability to resources beyond robots. Context selection
and public fact discovery depend on saved relationships/declarations, not failure
names or robot types. Thirty additional tests cover `Conveyor`, `M1`, and
`3D Printing Station`, custom public fields, parameter bindings, closed domains,
missing declarations, private fields, and offline retrieval without live calls.
They exposed two retrieval defects: the supported `not_equals_from_param` guard
was omitted, and snapshots could expose private fields from the standard field
list. Both are fixed. The exact `held_part` interpretation is included only when
its declared guards/updates apply; no general domain exception was introduced.

**Final automated verification:** 314 focused tests passed after those two fixes;
Poetry checks, Python compilation and diff checks passed. Replay verified identical
Part slippage prompts and 56 public facts before and after the final fixes, with
the original observation hash unchanged. The final fixes and replay are recorded
separately from the frozen API comparison. Its guarded live-interface call count
was zero. The existing UI/offline factory still enables generation only for
`Part slippage`; these shared prompt tests do not enable or verify non-robot
execution. Existing unrelated feasibility-suite failures remain separate.
The interrupted initial diagnostic batch is retained under
`cais_spade_llm/monitor/environment_runs/recovery/comparisons/20261009T024700Z`;
it is excluded from the complete matched comparison. Its two completed attempts
and interrupted third attempt remain available with exact requests/responses.
The second diagnostic batch at `comparisons/20261009T030800Z` retains one completed
blocked attempt and one interrupted attempt. It exposed the repeat-request loop
and the unclear successor explanation and is also excluded from model rankings.

**User-observed Gazebo verification: pending.** No Gazebo trial, primitive
composition, controller call, execution, or running-UI restart was performed for
this comparison. Saved outline results still require fresh Gazebo checks and an
explicit Execute recovery action before claiming functional recovery.

### October 9: public outline facts and recorded offline constraints

The outline prompt and its retrieval interface now exclude predefined task
contracts, primitive catalogs, private guards, implementation paths and evaluator
records. Actual interrupted/pending task names and LLM-authored event names remain
visible. Internal successor matching, primitive support, progress comparison and
runtime continuation are retained. Removing public catalogs does not authorize
arbitrary generated effects.

The compact resource index advertises available public capability fields. Requested
facts include the recorded `gripper_reach`, `staging_areas`, `reachability` and
`reachable_locations`, with their exact source fields and conflicting declarations.
Feedback preserves public rejection evidence without exposing a suggested action
or recovery sequence. Both selected safety requirements remain prompt inputs;
formal safety validation remains **skipped**.

Offline validation now checks recorded custody, destination declarations and reach
bounds before deferring fresh controller/trajectory evidence. Acquisition is derived
from declared effects and custody, including combined acquisition/release, rather
than event names. Robot checks do not run on nonrobot resources. A robot's named
`home` pose is kept separate from part-station reachability. Preserved part fields
do not falsely require already-held custody; actual changes and acquisition still
retain their custody checks.

Part and resource locations are separate, including explicit null values. Predicted
part movement retains historical measurements as `last_observed_pose`, invalidates
their use as a current source, and keeps proposed coordinates in `projected_pose`.
This also covers `resource_only` movement of a recorded carried part. Neither saved
fallbacks nor observation-store overlays may restore an obsolete measured pose.
Required missing source evidence remains `NEEDS_CONTEXT`.

**Final automated verification:** 546 tests passed in the combined offline, recovery,
context-retrieval, UI, request-record, setup and case3 suites, including all 105
offline tests. The accepted-turn regression now exercises the real candidate handler
through pickup and release, then verifies that reacquisition cannot use the old
measurement. Nine focused physical custody tests and 18 successor/custody/support
tests also passed. The broader feasibility suite has 66 passes and the same 12
previously recorded unrelated failures: five legacy AP descriptor fixtures and seven
KMR prepared-execution-provider cases. One NiceGUI `AwaitableResponse` warning remains
in the combined suite. `poetry check`, Python compilation and diff checks passed;
Poetry reports its existing metadata deprecation warnings.

**Completed API comparison:** `comparisons/20261009T123200Z/REPORT.md` records
three attempts each for `gpt-5.4 low` and `gpt-5.6-sol low`. Source hashes and the
observation hash stayed unchanged across all six trials; all initial requests and
schemas match. Guarded live-interface calls: **0**. Previous attempts and their
corrections remain preserved. No event-count target or example recovery sequence
was supplied.

| Model, effort | Reached outline review | Accepted event counts | Mean generation time | Mean total tokens |
|---|---:|---|---:|---:|
| `gpt-5.4 low` | 3/3 | 5, 3, 4 | 119.2 s | 28,769 |
| `gpt-5.6-sol low` | 3/3 | 3, 3, 3 | 55.0 s | 22,840 |

The first outline message is **7,123 characters**, down from revision 3's 8,364
(14.8%); the August legacy message was 4,683 characters and had different recorded
facts. Every accepted event matches an exact LLM proposal. All six outlines stage
`KET4_Square_4mm` at `Buffer For Machined parts` and project `gear_small` assembled.
Invalid `ur5e-4` acquisitions, including a combined acquisition/place proposal,
are rejected by the recorded reach check. These are six diagnostic planning
outcomes on one capture, not six executed recoveries or proof of general reliability.

**Remaining evidence limits:** formal safety is skipped and shared-area clearance
is unresolved. The saved `ur5e-4` symbolic location is `3D Printing Station`, while
its measured TCP and failure checkpoint record stopped placement lowering; neither
establishes a validated assembly-area occupancy or clearance result. Proposed home
events do not establish observed clearance. `outline_review` also does not prove
exact nominal resumption or completed remaining assembly. See the separate
`outcome_review.md` and `prompt_review.md` for source pointers and captured evidence.

**Corrections after the fixed API batch:** final artifact review found that the
accepted-turn path uses `_apply_task_effects_to_symbolic_state`, while the earlier
pose fix covered `_apply_outline_task_effects`. The actual commit path is now fixed
and covered by the accepted-turn regression above. The batch never reacquired a
relocated part, so its first-acquisition reach evidence remains applicable. A separate
metadata correction labels unchanged saved `pending_tasks` as `saved_observations`
after projected transitions. The original API records and frozen code hashes are
retained; final-code saved-response replay is recorded separately in
`comparisons/20261009T123200Z/post_review_replay.md`, without new model calls or
new model performance claims. All six replayed response streams reached
`outline_review` with matching accepted events and candidate decisions. All 40
recorded responses were consumed unchanged; initial messages remained identical,
and changed later messages were recorded separately. API/equipment calls: **0**.
The original API manifest and source capture hashes are unchanged. The first replay
harness attempt failed to create its output directory; its records and explanation
are retained separately from the successful replay and do not count as model trials.

**User-observed Gazebo verification: pending.** This implementation and comparison
do not generate primitives, execute recovery, restart the running UI, or establish
remaining assembly completion. The source capture remains unchanged. Offline stays
Manual; later execution requires fresh checks and an explicit operator action.

### October 9: RA evidence preparation and isolated CCA preview repair

New outline attempts retain `validation_contract_version=2` and
`primitive_support_semantics="successor_primitive_union"`. Resource-owned parameter
preparation now connects recorded facts, declared bindings, and supported local
calculations to independent primitive capability checks. Every check uses the
same starting snapshot. This does not order primitives or establish composition.
The recorded buffer proposal can validate `release_part` without an inapplicable
`assembly_slot`; product `part_state="ready"` remains a separate obligation.
Missing movement inputs remain explicit rather than being replaced with prior
motion parameters, placeholder TCP offsets, or inferred coordinates.

New captures retain owner configuration, actual attachment evidence, full poses,
declared pick geometry and policy, finite gripper joint bounds, and TCP observation
provenance. Projected facts remain separate from recorded facts. A captured TCP
height offset cannot be reused after a different projected orientation. Private
validator retrieval uses precise source references and retries the same proposal.
Saved supplements are restored only against their source capture and checkpoint;
new attempts do not silently inherit old supplements.

`Prepare Gazebo evidence` is available for unresolved outline validation through
`SystemBridge.part_slippage_action(...)`. It verifies the owned paused checkpoint,
including configuration, custody, resource/task state, revision, acknowledgements,
transitions, and monitor history. A mismatch produces a separate capture and
`Use new observations`; it never modifies the original capture. Stop drains the
owned preview before another action is allowed. Evidence retries call validation
without another LLM request and retain separate records under the owning turn.

`PreviewRecoveryMotion` plans against a complete private MoveIt scene with exact
resource, frame, link, model, scene, capture, and proposal bindings. It supports
Cartesian waypoints and declared named-pose joint targets. Its server has no
execution, gripper, attachment-mutation, live-scene-update, or command-ledger
clients. Full paths, partial paths, cancellation, and stale evidence are recorded
separately. Typed ROS field hashing avoids CDR padding differences between Python
and C++. A complete preview reports `command_sent=false`,
`execution_authorized=false`, and `continuous_collision_verified=false`; the
existing continuous geometry and CCA checks remain required.

RA, physical projection, and CCA results are separate in the expandable UI records.
Missing physical evidence does not overwrite RA `FEASIBLE`. Only supported event
projections and CCA results can advance the accepted prefix and monitor state.
Peer configurations and carried parts remain in the projected scene. Nominal
continuation requires exact unfinished-task evidence; primitive support alone
does not prove it. Both selected safety specifications retain their AP meanings.

**Remaining evidence limits:** the October 8 capture lacks some joint, frame, and
TCP/attachment information. New captures do not automatically establish every
future motion. A loose dynamic part appearing at rest is not stationary interval
evidence. Capture records static geometry or the existing owner's supported idle
containment contract where available; unresolved parts remain explicit. Stable
release placement remains unresolved when `release_settling_modeled=false`.
Product `ready` and completion effects cannot be fabricated from a release.

**Automated verification:** the final combined recovery/offline/UI/request/evidence
suites passed **935 tests**, with the same **12 existing failures** recorded
separately: five retired AP descriptor fixtures and seven KMR prepared-execution-
provider fixtures. One existing NiceGUI `AwaitableResponse` warning remains.
`poetry check`, Python compilation, and diff checks passed; Poetry retains its
metadata deprecation warnings. `make bootstrap-gazebo` built 16 packages, and all
**8 installed ROS preview integration tests** passed in an isolated localhost
domain with synthetic scenes and no Gazebo or execution interfaces.

**Model comparison:** the fresh runner and verification records are in
`cais_spade_llm/monitor/environment_runs/recovery/comparisons/20261010T013058Z/`.
After explicit disclosure approval, all **six new API attempts completed** with
identical initial prompts/schemas, safety specifications, and the unchanged source
capture. Automatic approval review had rejected the earlier command before process
creation; that rejection and the subsequent approval are retained separately.
Implementation and safety fingerprints stayed unchanged throughout all six attempts;
guarded live-interface calls were **0**. See `REPORT.md`, `report.json`, and
`outcome_explanation.md` in that comparison directory.

| Model | Attempts | Complete outlines | Accepted events | Mean seconds | Mean total tokens |
|---|---:|---:|---|---:|---:|
| `gpt-4o` (reasoning parameter omitted) | 3 | 0 | 0, 0, 0 | 5.688 | 6,723.667 |
| `gpt-5.6-sol low` | 3 | 0 | 0, 0, 0 | 21.124 | 11,575.667 |

`gpt-4o` attempt 1 stopped during grounding because the saved observed pose of
`KET4_Square_4mm` is unavailable. Attempt 2 passed RA for `move_to_named_pose`,
then reached unresolved physical projection and CCA because the saved physical
scene is missing. Attempt 3 proposed unsupported successor conditions and then
lacked required capability evidence. `gpt-5.6-sol` attempt 1 proposed returning
`KET4_Square_4mm` to `Buffer For Machined parts`: `release_part` passed, while
movement inputs remained missing. Its attempts 2 and 3 proposed `gear_small`
pickup and lacked grasp/movement evidence. PA public context retrieval was used
in its attempts 1 and 3; private validator retrieval remained separate.

Five attempts ended `NEEDS_CONTEXT`; the grounding failure retains its original
runner classification `incomplete` with the explicit missing-observation error.
None established nominal continuation. Outline safety remained unresolved and
execution safety skipped. A separate exact saved-proposal replay confirms the
corrected release capability and remaining movement inputs; it is not an extra
model attempt. No missing evidence was bypassed and no prompt solution hints,
preferred event sequence, or event-count target were supplied.

**User-observed Gazebo verification: pending.** No live capture, Gazebo preview,
recovery execution, current-run interruption, or UI restart was performed. Fresh
capture and preview acceptance remain available through the UI controls. Complete
primitive composition validation, recovery × recovery and recovery × nominal
monitoring, shared CCA history/ownership, dispatch authorization, deviation and
Stop handling, and concurrent execution remain the subsequent runtime phase.

### Authorized Gazebo capture and `gpt-5.6-sol low` repair trial — October 9–10, 2026

The user subsequently authorized agent-operated Gazebo capture and model trials.
These results are separate from user-observed UI acceptance and from the six
fixed-capture comparison attempts above. Recovery execution remains disabled.

The trials identified and repaired omitted native mesh scale, attachment and
controller evidence, ROS octet decoding, stale MoveIt part geometry, and native
preview handling of captured passive mimic joints. Private previews now retain
those measured joints instead of replacing them with ideal URDF mimic values.
Eight native preview tests passed twice after that change. New attempts replay
recorded unfinished nominal tasks; historical attempts retain their prior
semantics. Nominal completion requires supported effects and CCA evidence.

The fresh `gpt-5.6-sol low` attempts under `experiments/20261010T0234Z` and
`experiments/20261010T0250Z` made real API requests and reached RA `FEASIBLE` for
their proposed events. Both remained incomplete at physical projection; neither
is counted as a passed outline. Exact requests, responses, and preview failures
are retained in their linked generation directories.

Isolated Gazebo contact tests then reproduced persistent gear motion from dense
mesh contact sampling. The selected correction preserves every original binary
triangle record, normal, and attribute in deterministic order. Separate collision
assets leave the visual originals unchanged. Global, gear, and table contact
limits are 64; the table is resolved explicitly to avoid Gazebo's cached model.
The isolated test observed all three gears naturally disabled with zero velocity
through its remaining samples; no body was forcibly disabled and no validation
threshold was relaxed. Proofs and raw samples are retained under
`experiments/20261010T0250Z/physics_diagnostics`. Eight focused asset checks passed.
The corresponding ROS builds each passed all 16 packages.

`experiments/20261010T0343Z` captured the intended drop but exposed the cached
table override. Its later capture had complete resource observations and all
unheld parts supported as stationary except `gear_small`. No LLM request was
made from that capture. The corrected full run is recorded separately under
`experiments/20261010T0348Z`. It retained three fresh captures, including
`capture_047a4eff436844198dfc21f05629b14f`, and stopped Gazebo before the
subsequent offline generation. All unheld parts except `gear_small` had
supported stationary evidence; the remaining motion is recorded, not hidden.

Two real `gpt-5.6-sol low` offline attempts then used that unchanged capture.
`experiments/20261010T040213Z` made four API calls and proposed returning
`KET4_Square_4mm` to `Buffer For Machined parts`, after two dynamic retrieval
turns. RA exposed a loader bug: legacy quaternion restoration compared a
complete owner snapshot with a separately sampled `robot_poses` entry. The
repair restores missing quaternion fields only, retaining the explicitly
recorded owner snapshot and all supplied-evidence integrity checks. The exact
proposal replay reached RA `FEASIBLE`; all 156 offline tests passed.

`experiments/20261010T040634Z` independently made four new API calls, again
retrieved context and proposed the buffer event. RA passed. Physical projection
and CCA remain unresolved because no matching owner event preview is recorded.
The attempt took 45.620 seconds and 16,090 tokens, with zero live-interface calls,
unchanged source and implementation hashes, and zero accepted events. Neither
attempt is counted as a complete outline. An isolated native preview of the
saved planning scene is being recorded separately, without Gazebo, controller
commands, additional LLM calls, or recovery execution.

**User-observed UI acceptance remains pending.** Agent-operated simulation,
model-based outline validation, and recovery execution are distinct results.

The subsequent live run `experiments/20261010T041055Z` captured the intended
drop and all required resource observations. Its separate capture
`capture_4862656369764af1b78a86063afe6602` records supported stationary evidence
for every unheld part, including `gear_small`, under the unchanged velocity
limits. Gazebo was stopped before generation. The observations hash remains
`a6c4c34f4f3953b11067684c5a1b4c903277f665204da2f3949083ee327ce787`.

`experiments/20261010T041521Z` then made three real `gpt-5.6-sol low` API calls:
grounding, dynamic retrieval, and a proposal to place `KET4_Square_4mm` in
`Buffer For Machined parts`. RA returned `FEASIBLE`. Generation took 32.885
seconds and 10,756 tokens. The original capture and implementation hashes were
unchanged, no live interfaces were called, and no event was accepted without
the required physical and CCA evidence.

The native preview now also accepts archived Goal/Result files. This uses the
same scene, model, joint, collision, path, timing, and provenance checks as the
ROS action, without an action server or controller connection. Eight native
file-mode regression tests passed, and the corresponding ROS build passed all
16 packages. The actual saved proposal reached the native calculation and
reported three starting contacts: `KMR_base_link` with `iiwa_link_0`, and each
of `ur5e_1_base_link_inertia` and `ur5e_2_base_link_inertia` with its recorded
pedestal `mounting_plate`. Those mounting declarations are being verified
separately; the original proposal, capture, and rejected preview remain intact.
This is still an unresolved outline, not a passed CCA or execution result.

Subsequent detached replays retained that same LLM proposal. The three mounting
contacts now have exact fixed-joint or recorded pedestal evidence; only those
verified contact pairs are permitted in the private scene. Projected support
records also retain JSON-compatible captured numeric poses instead of leaking
internal rational numbers into artifacts. The resource-owned placement
calculation derives a horizontal supporting face and stable contact from the
recorded rigid box, destination surface, gravity, and attachment transform. It
does not alter observations or claim a settling outcome. RA capability checks
and the independent preview now receive the same recorded physical inputs.

`experiments/20261010T045750Z` passed native Cartesian endpoint refinement but
remained unresolved in physical projection. Investigation found that the native
MoveIt model used the configured fixed robot base transform while Python used
the measured Gazebo base transform. The resulting micrometre-scale endpoint
difference prevented supported contact. This is a model-input inconsistency,
not a new LLM failure; that replay made no provider call and accepted no event.
The repair must preserve measured roots in the private model and retain both
original and calculated model fingerprints.

Automated results for this repair: 467 focused outline/evidence/offline/context
tests passed; Poetry, compilation, and whitespace checks passed. The native
file-mode suite passed 11 tests and ROS build 15 completed all 16 packages.
The broader feasibility suite recorded 426 passes and the same 12 previously
recorded failures in retired AP-descriptor and KMR prepared-provider paths;
those paths remain separate from the offline outline repair. Source observation
hashes for both the original October 8 capture and the fresh capture are
unchanged. User-observed UI acceptance and recovery execution remain pending.

The subsequent private-model repair preserves the measured Gazebo root
transforms while retaining the original URDF and source/model fingerprints.
It also retains the source joint limits and solver configuration. Native
Cartesian calculation precision is recorded separately from physical tracking
tolerance; Python independently checks the recorded endpoint against that
numerical contract. No measured pose, contact threshold, safety region, or
safety specification was changed to obtain acceptance.

The pending-proposal handoff now uses the retained task with its observed
starting conditions and checks that every LLM-authored field is unchanged.
CCA reuses certified interval refinement when a bound for the entire movement
is inconclusive. For this proposal, two bounded intervals establish clearance
while retaining the original seven semantic observations and all peer geometry.
Unknown occupancy remains unresolved if refinement cannot establish it.

**First accepted LLM event, full outline still incomplete:**
`experiments/20261010T052904Z/report.json` records a detached validation retry
of the exact `gpt-5.6-sol low` proposal from `20261010T041521Z`. The event is
`place`, owned by `recovery-resource-3@localhost`, placing `KET4_Square_4mm`
in `Buffer For Machined parts`. RA, independent physical projection, and both
`SAFE_shared_area_mutex` and `SAFE_gear_small_before_KET4_Square_4mm` passed;
the event was appended. The lineage audit links the accepted event to the
original provider response and the successful validation artifact. The retry
took 44.295 seconds, made zero API or forbidden live-interface calls, and
preserved the source, Python, and native implementation hashes. It did not
execute recovery or grant execution permission.

Nominal continuation is still `not_ready`. `environment_8` requires
`resource_state="picked"` for `recovery-resource-4@localhost`, whose supported
state remains `failed`. `ed9a5f9c8c4b463496402821945cfe72_pending_3` requires
`resource_state="picked"` for `recovery-resource-3@localhost`, whose supported
state is now `placed`. Further LLM-proposed events are required. This result
is one accepted event, not a complete successful outline or completed assembly.

Final automated checks for the current repair: 640 focused recovery evidence,
preview, validated-outline, continuous-motion, offline, context-retrieval, and
request-record tests passed in 45.93 seconds. Poetry, Python compilation, and
whitespace checks passed. The native file-mode suite passed 14 tests, and the
latest `make bootstrap-gazebo` build completed all 16 packages. The previously
recorded 12 broader-suite failures remain recorded separately above.

Further real generation is waiting for explicit disclosure approval. Automatic
approval review rejected a new API request because it would send saved internal
observations and recovery context to the configured external OpenAI API. That
request was not run or bypassed; subsequent retries above used the saved
provider response locally. User-observed UI acceptance, a complete accepted
outline, primitive composition, and recovery execution remain pending.

### October 10, 2026: six fresh offline comparison attempts

The user explicitly approved disclosure of the saved observations, internal
resource/task context, both specifications, and dynamically retrieved facts to
the configured external OpenAI API. All six requested attempts then ran under
`comparisons/20261010T102919Z`; `RESULTS.txt` and `results_summary.json` retain
their proposals, acceptance evidence, retrieval requests, timing, token usage,
and links to the complete original provider and validation records.

Each attempt started from scratch using `capture_4862656369764af1b78a86063afe6602`.
No earlier proposal or accepted prefix was loaded. The prepared grounding and
outline prompts were identical across all attempts. `gpt-4o` omitted
`reasoning_effort`; `gpt-5.6-sol` used `low`. Neither a preferred recovery sequence
nor a 3–5 event target was added to the prompts. Runtime generation limits and
the 600-second attempt deadline were retained; all three attempts per model ran
despite repeated blockers. Native physical previews used recorded models and
private scenes with live interfaces and execution blocked.

| Model | Attempt | LLM-proposed events | Accepted events | Elapsed seconds | Total tokens | Complete outline |
|---|---:|---:|---:|---:|---:|---|
| `gpt-4o` | 1 | 1 | 0 | 24.9 | 5,549 | No |
| `gpt-4o` | 2 | 1 | 0 | 22.2 | 5,562 | No |
| `gpt-4o` | 3 | 1 | 0 | 23.2 | 5,518 | No |
| `gpt-5.6-sol low` | 1 | 2 | 2 | 192.8 | 15,723 | No |
| `gpt-5.6-sol low` | 2 | 1 | 0 | 48.2 | 10,926 | No |
| `gpt-5.6-sol low` | 3 | 1 | 1 | 136.9 | 16,066 | No |

`gpt-4o` proposed `recover_gear_small`, `Fix_and_Reposition_gear_small`, and
`reallocate_failure_task`, respectively. Each assigned `gear_small` to
`recovery-resource-3@localhost`, whose recorded `held_part` and
`_task_ctx.part_name` are `KET4_Square_4mm`. RA returned `NEEDS_CONTEXT` with
`Recorded task context belongs to another part`; no CCA acceptance occurred.
The independent owner event preview also lacked a supporting declaration for
the requested combined end conditions. This classification stops for missing
evidence rather than allowing an LLM revision from infeasibility feedback, so
these attempts do not test the model's ability to revise that proposal.

`gpt-5.6-sol low` attempt 1 proposed `place` for `KET4_Square_4mm` into
`Buffer For Machined parts`, followed by `pick` for `gear_small` using
`recovery-resource-4@localhost`. Both events passed RA, physical projection,
and both CCA checks. Attempt 3 proposed and passed the same `gear_small` pickup
as its first event. Both attempts then stopped while validating the recorded
unfinished nominal task `environment_8` / `place_approach`: the private preview
reported `gear_small/link/collision <> table_xarm6/link/surface` in the starting
configuration. The supported pickup preview ends at grasp without a post-grasp
lift; the next preview treats that same part as attached and rejects its
remaining support contact. The runtime pauses at this nominal validation and
does not ask the LLM for another proposal. This is an integration gap between
pickup and loaded motion; it does not establish that the LLM failed to propose
a needed subsequent event. A repair needs supported contact separation or
loaded retreat evidence, with collision checking, rather than an assumed lift
or a general collision exemption.

`gpt-5.6-sol low` attempt 2 proposed `move_home` for
`recovery-resource-4@localhost` with explicit `part_name: null`. RA passed.
The runtime task constructor omitted that field, and `pending_outline_event`
rejected the changed record before physical preview. This is an implementation
error: preserve the authored explicit null through task preparation and retain
the exact-proposal check. No physical infeasibility or CCA rejection was
established for that event.

None of the six attempts reached 3–5 proposed events or a complete accepted
outline. The blocking implementation remained frozen throughout the batch;
these results must not be presented as a conclusive model-quality comparison.
There were 17 provider requests and responses, 59,344 tokens, zero provider
errors, and zero timeouts. Source, implementation, native preview, and selected
safety fingerprints remained unchanged; no forbidden live-interface calls or
recovery execution occurred. The focused request-record suite passed all 52
tests in the host environment after sandboxed collection stalled and was
cancelled. Poetry and compilation checks passed. The earlier 640-test repair
result remains separately recorded above. User-observed UI acceptance and
complete outline/assembly acceptance remain pending.

### October 10, 2026: declared departure and validation feedback repair

New attempts retain `validation_contract_version=2` and
`primitive_support_semantics="successor_primitive_union"`, and pin
`physical_preview_contract_version=3`. Explicit `part_name: null` survives task
construction and exact-proposal comparison. Recorded parameters remain tied to
their originating part and task; another part's `_task_ctx` cannot supply them.
Established contradictions return factual rejection feedback, and implementation
exceptions retain their original error details.

The independent owner preview now includes the departure declared after grasp
and the movement declared after release. These are resolved from owner
declarations and supported parameters, including `travel_z`; display examples
are not motion evidence. The private scene advances through each verified
interval and custody change, retaining peer configurations and carried-part
geometry. A validated pickup calculation remains tied to its owner, part,
contract, and binding for a subsequent placement. New attempts cannot reuse a
historical preview that omitted these movements. Historical artifacts are
preserved with their original contract versions.

Supporting-contact departure retains the exact recorded collision pair and
penetration bound. Native contact checks and independent continuous bounds
reject deeper penetration, unrelated contacts, incomplete contact inventory,
missing separation, or re-entry. `contact_surface_layer=0.001` is not used as
an additional penetration allowance. Neither preview nor outline validation
grants execution permission.

The sequential square-placement/gear-pickup replay exposed a separate incomplete
Cartesian search for `ur5e-3`. Its native IK query timed out under the recorded
0.05-second solver budget. No callback collision or joint-bound rejection was
reported; the last accepted wrist joint value was near its upper bound, which
does not establish the cause or exhaustive infeasibility. Finer sampling also
failed. The exact native goal, solver diagnostics, and refinement results are
retained in
`experiments/20261010T113634Z/diagnostics/cartesian_pickup/report.json`.
The declared motion, coordinates, and limits were not changed to force success.

For a complete-input bounded IK or joint-jump failure, physical validation stays
`NEEDS_CONTEXT`, CCA remains `not_checked`, and the accepted state and monitor
history remain unchanged. The PA can receive concise solver facts and propose
a revision. Repeating the same public conditions with unchanged scene, models,
owner evidence, and motion policy stops; renaming an event does not reset this
guard. Missing input facts, unavailable solvers, incomplete contact evidence,
global deadlines, and cancellation remain blocked. The same distinction applies
to pending nominal validation. No preferred event, sequence, or event-count
target was added to generation prompts.

Automated verification before the next fresh API attempt is recorded in
`experiments/20261010T113634Z/automated_verification.json`. The strict evidence
suite passed 190 tests; offline/workflow/context/outline checks passed 484 tests,
with a final nearest-suite run of 78 tests after the eager-copy regression.
Native-client checks passed 84 tests, native file-mode checks passed 21, and
native action checks passed 11. These are separate runs, not additive totals.
`make bootstrap-gazebo` completed all 16 packages; Poetry, compilation, and
whitespace checks passed. The broader earlier repair run passed 1,145 tests and
reported the same 12 previously recorded older-path failures. The original
capture hash remains
`a6c4c34f4f3953b11067684c5a1b4c903277f665204da2f3949083ee327ce787`.

The saved `move_home`, square-placement, and gear-pickup proposals each completed
their version-3 physical preview in `experiments/20261010T113634Z/replay_v3`.
These independent local replays are not a generated complete outline. They made
no API or live-equipment calls and executed no recovery. Fresh generation
results and user-observed Gazebo acceptance are recorded separately below.

### October 10, 2026: owner reach declarations and artifact memory use

New attempts pin `physical_preview_contract_version=4`. The independent physical
preview now checks its calculated Cartesian targets against the captured owner's
`static_capabilities`, including `gripper_reach`. This closes a gap in which a
native kinematic preview could pass while violating the owner's public declared
reach. The unordered RA union checks each supporting primitive against the same
starting snapshot; it does not prove the approach to a new pickup target. For
the saved `gear_small` pose, the calculated radius for `ur5e-4` is approximately
0.8000 m, exceeding its declared 0.7000 m radius and 0.0100 m tolerance. The
version-3 pickup replay above remains a historical native geometry result and
cannot satisfy the new physical contract. The original capture, motion limits,
selected safety specifications, and generation prompts were not changed to
force success. Exact owner declarations are bound before preview calculation;
incompatible historical cached previews cannot be reused.

The version-4 local preparation replay in
`experiments/20261010T120821Z/replay_v4_prepare/summary.json` prepares the recorded
`move_home` and square-placement proposals and rejects the recorded `ur5e-4`
gear pickup using that owner declaration. It preserves explicit `part_name:
null`, makes no API or live-interface calls, and leaves the source hash unchanged.

Artifact preparation also copied the same large physical evidence graph several
times before writing each turn, and built an unused transcript when transcript
writing was disabled. These copies were removed while preserving exact provider
captures and independent mutable audit records. A recorded turn benchmark
changed from 10.087 seconds / 1,230,432 KiB peak memory to 0.00198 seconds /
476,032 KiB, with equal request, context, and retrieval hashes. Loading that
fixture alone used 476,032 KiB. A complete local workflow replay with one saved
response used 1,405,036 KiB peak memory and preserved the accepted prefix,
projected resources, monitor history, and context. Evidence is in
`experiments/20261010T1215Z-memory-repair/report.json`. This is an implementation
repair, not an alteration of the LLM's proposal or validation requirements.

The immediately preceding third API attempt ended with `SIGKILL`; its parent
record confirms cancellation and no hard timeout. The saved process PID is not
the PID in the available kernel memory-kill entry, so the termination cause is
not asserted as proven. Its preserved provider records contain five responses,
22,546 tokens, two proposed events, and one accepted square-placement event.
Its gear pickup has no completed retained validation and nominal continuation
was not established. Original partial records were kept; reconstruction is in
`experiments/20261010T113634Z/attempt_3_gpt-5.6-sol/recovered_report.json`.

Pre-attempt verification is recorded in
`experiments/20261010T120821Z/automated_verification.json`: 223 owner/evidence
tests, 84 preview tests, 252 validated-outline/offline tests, 496
workflow/validated/offline/context tests, and 20 focused artifact/version tests
passed in their respective runs. These overlapping suites are not additive.
Two experiment-report tests also passed. Poetry, compilation, and whitespace
checks passed. Native code is unchanged since the successful 16-package
bootstrap, 21 native file-mode tests, and 11 native action tests. The previously
recorded 12 older-path failures remain separate from these results.

The fourth fresh `gpt-5.6-sol low` attempt is retained in
`experiments/20261010T120821Z/attempt_4_gpt-5.6-sol/report.json`. It made eight
provider calls (39,890 tokens), independently proposed five events, and accepted
two: square placement in `Buffer For Machined parts`, then `ur5e-3` movement to
`home` with explicit `part_name: null`. In between, the `ur5e-4` pickup was
rejected by its declared reach and the `ur5e-3` pickup remained unresolved after
an incomplete Cartesian search. The LLM requested additional resource
capabilities and proposed the home movement itself. Its final pickup proposal
was not fully validated before the 600-second generation deadline. The parent
confirmed process exit and complete cancellation; report finalization finished
after the generation deadline. This is a timed-out partial outline, not complete
success. Source, implementation, native, and safety fingerprints remained
unchanged, with zero forbidden-interface calls. Peak child memory was
5,352,720 KiB. Further local replay and memory profiling precede another API
attempt; no event count or sequence was supplied to the model.

The remaining turn-memory defect was traced to `prompt_input.session_state`:
the saved Turn 8 embedded 541,018,203 formatted bytes of internal preparation
state, including earlier turn inputs. Those bytes were not the provider prompt.
The version-2 builder now excludes unused internal history and validation
records from its detached prompt presentation; full validation state, captures,
and prepared inputs remain available separately. Rebuilding the actual Turn 8
against its complete public resource/part view yields an identical 17,357-character
provider prompt (SHA256
`1fcf38db55811d67ac9c6b8ca5293c60fbc8c31deef38178aaccd5b11ec6c182`),
while the preparation artifact becomes 2,019,487 bytes. The atomic diagnostic
writer also streams JSON with the same formatting instead of allocating another
complete encoded string. Evidence is in
`experiments/20261010T120821Z/memory_diagnostics/turn8_prompt_comparison.json`.

The rejection-feedback path now retains the owner's supported calculated target,
recorded `gripper_reach`, source/revision, and exact numeric rejection instead of
replacing it with generic wording. Private validation models remain excluded.
Recovery and nominal obstruction paths preserve the same verdicts and solver
evidence. The validated-outline/context suite passed 249 tests after this repair;
Poetry and compilation passed.

Local replay of the last proposal after the accepted `home` event still returned
`NEEDS_CONTEXT`, with native Cartesian fraction 0.2894736842. The search timed
out near the recorded wrist-3 bound without a collision rejection. Home retains
the same joint branch; this does not establish exhaustive physical infeasibility.
The exact replay is in `experiments/20261010T120821Z/replay_turn8_after_home`.
A separate diagnostic tried the existing declared
`ignore_current_height_for_travel_z=True` option with a recorded approach height;
it also remained unresolved (fraction 0.2857142857). It did not change production
policy, source observations, orientation, joint bounds, or solver settings and
does not count as accepted outline evidence. See
`experiments/20261010T120821Z/declared_pick_parameter_diagnostic_v4/report.json`.

The fifth fresh `gpt-5.6-sol low` attempt is retained in
`experiments/20261010T123123Z/attempt_5_gpt-5.6-sol/report.json`. It completed six
provider calls (28,326 tokens), including two context-request turns, proposed
three events, and accepted one square-placement event. The model independently
recognized that `gear_small` was outside `ur5e-4`'s declared reach and retrieved
`ur5e-3`'s capabilities. Its subsequent `ur5e-3` pickup remained `NEEDS_CONTEXT`
after incomplete native IK search. The exact next provider input included path
fraction 0.2894736842, query 22/error -6/timeout 0.05 seconds, and the last accepted
`ur5e_3_wrist_3_joint` value and declared bounds. The model repeated the same
resource, part, end conditions and absent parameters; the unchanged-evidence
guard stopped generation without appending it. The model was not given a
preferred recovery sequence. The parent recorded completed cancellation after
242.91 seconds, with 2,125,740 KiB peak child memory and no deadline expiry.

The independent `prefix_audit.json` and `independent_replay_report.json` beside
that report confirm the accepted event's exact provider provenance, saved RA
evidence, projected state chain, and replayed CCA history. Both selected rules
remain unchanged. Nominal continuation remains `not_ready`; no complete outline,
recovery execution or assembly completion is claimed. The audit made no API,
native-preview or live-equipment calls. All five repair attempts are consolidated
in `experiments/20261010T123123Z/repair_attempts_1_to_5.json` and the accompanying
text report. Implementation changed between those attempts, so these results
are not a controlled model comparison. Attempt 3's missing total duration and
unconfirmed termination cause are retained as unknown.

The latest automated checks are recorded in
`experiments/20261010T123123Z/automated_verification.json`: 249 validated-outline
and context tests, 22 diagnostic tests, and four targeted explicit-null/prompt/
artifact regressions passed. The broader artifact/request/UI suite passed 184
tests and retained one older `ra_admissible_event_ids` audit assertion failure.
These overlapping suites are separate runs, not additive totals. Poetry,
compilation and whitespace checks passed. Fresh route-calculation work, if
implemented, must supply separately supported owner evidence; incomplete IK
search is not exhaustive infeasibility. User-observed Gazebo recovery acceptance
remains unrecorded for these offline attempts.


The subsequent nominal-continuation repair recognizes supported completion of
an original product obligation by another declared owner. It compares exact
product effects and parameters, checks the projected completion record against
an accepted recovery event and its owner declaration, and preserves original
resource-state goals and cleanup tasks. The saved `ur5e-4`/event65 and
`ur5e-3`/event59 `place_insert` declarations exercise this distinction without
changing their identifiers. An LLM-authored `part_state="assembled"`, a missing
task entry, an unsupported effect, or an unresolved external dependency cannot
retire the original task. Historical same-owner behavior is retained. Separate
pending product groups are not combined to infer an external dependency;
uncertain cases remain pending.

The offline/workflow/context/validated suites passed 515 tests after this repair,
and the final nominal/completion selection passed 42 tests with the exact saved
declaration structure. The actual-declaration fulfillment check is explicitly
synthetic: it tests bookkeeping with independently supplied completion records,
not a newly validated physical assembly. The source observation hash is unchanged.
Records are in
`experiments/20261010T123123Z/nominal_owner_completion_repair.json` and
`nominal_owner_fulfillment_after_repair.json`. Poetry, compilation, and whitespace
checks passed. No API, native preview, or execution call was made for that check.

The older artifact assertion failure is now repaired: compact audit records
retain explicitly present `ra_admissible_event_ids`, including empty lists and
`null`, while absent fields stay absent. The case3/request-record/project-page
suites passed 186 tests in 20.47 seconds; the preceding failure record is
preserved. This changes audit fidelity, not recovery planning.

A sixth fresh attempt is staged in `experiments/20261010T131151Z`, with the same
source observations, selected safety specifications, initial prompt hashes,
`gpt-5.6-sol low`, Manual stages, 80-turn limit, and 600-second generation deadline.
It requires preview contract version 5 and records native route calculation
separately from strict trajectory replay. Route candidates alone are not
accepted event evidence. The staged runner also fingerprints the installed
action interfaces and native libraries, blocks equipment and execution calls,
and requires a separate saved-prefix audit before reporting complete success.
The staged runner has not called the API; generation waits for integration,
focused verification, and a frozen implementation.


The version 5 local replay of the exact attempt 5 pickup now establishes RA
`FEASIBLE` and complete physical projection, but CCA rejects
`SAFE_shared_area_mutex`. The derived continuous cells establish both `ur5e-3`
and `ur5e-4` occupying `assembly_board-v1`: observation 1 cells 9 and 10, and
observation 6 cell 0, each records both occupancy values as `[true]`. This is a
modeled safety violation, not an execution collision. The original first
counterexample displayed a coarser uncertain cell; the repaired diagnostic now
shows definite violating cell 9 with both AP values `[true]`. Exact saved CCA
replay took 8.39 seconds, retained monitor history, and separately found
`SAFE_gear_small_before_KET4_Square_4mm` feasible for this proposed event. The pickup was not appended,
and the accepted square-placement prefix, projected scene, and monitor history
were retained unchanged.

This replay reused four exact native results only after comparing every typed
Goal field; it made no API, new native, or equipment calls. Source observations,
original provider responses, and native records are unchanged. It took 170.84
seconds: 86.74 seconds for owner evidence preparation/finalization and 69.32
seconds for the validation retry. Records are in
`experiments/20261010T131151Z/local_attempt5_pickup_v5_retry4`; the complete exact
CCA replay input is retained separately as `cca_replay_input.json.gz`, and the
cell-level occupancy facts are in `cca_mutex_cells.json`.

The preceding preparation repairs are also verified: scene-object provenance
now constructs JSON lists instead of tuples, preserving exact equality after
saving without weakening conflicting-mapping checks (4 focused tests passed).
The continuous support-departure bound repair passed 39 focused tests and the
exact saved trajectory over 119 certified cells. Two synthetic audit tests
verify accepted-prefix order and no append after rejection. Native verification
passed 24 file-mode and 11 action tests, with 26 subtests; bootstrap completed
16 packages. Version 5 factory/historical restore tests passed 252 tests. The
full recovery outline and nominal continuation are still unproven; attempt 6
remains staged pending the CCA diagnostic check and implementation freeze.

The CCA factual feedback formatter also retains the recorded `cell_index`, exact
`interval`, and `definite_violation` in the PA prompt, alongside the rule-local
AP values. It adds no missing values or suggested recovery events. Eight focused
prompt and nominal-feedback tests passed; Poetry, compilation, and whitespace
checks passed. Initial prompt instructions are unchanged.


Attempt 6 ran the unchanged initial prompts with `gpt-5.6-sol low` and produced
six actual provider responses: three proposed events and two context-retrieval
turns, totaling 28,029 tokens. The owner rejected the first `gear_small` pickup
by `recovery-resource-4@localhost` because the calculated target is outside its
declared `gripper_reach`. The square-buffer placement by
`recovery-resource-3@localhost` passed RA, physical projection, and CCA and was
appended. The later pickup by `recovery-resource-3@localhost` reached physical
preparation, where the Python process unexpectedly exited with SIGSEGV (-11).
The run lasted 222.46 seconds, peaked at 1,888,304 KiB child RSS, and did not reach
the generation deadline. Kernel records confirm SIGSEGV; neither OOM nor a
specific Python function is established as the cause by the periodic stack.

The original partial report, provider captures, and native records are retained.
`experiments/20261010T131151Z/attempt_6_recovered` separately reconstructs the last
durable validated state and independently verifies its one accepted event;
nominal continuation and a complete outline remain false. Final in-memory guard
counters were unavailable after the crash and are recorded as unavailable.
Original observations and frozen implementation/native fingerprints match.

The exact pending pickup was then replayed locally from that saved prefix with
fatal tracing and phase instrumentation. It completed without crashing in
173.54 seconds, with four non-executing native calculations and zero API or
equipment calls. RA and physical projection passed; CCA rejected the established
`SAFE_shared_area_mutex` violation, leaving the accepted prefix unchanged.
Projected scene dictionaries contained plain Python values, no cycles or aliases,
and preserved exact JSON roundtrip equality. The installed generated action and
native library hashes matched the frozen run. This local replay does not explain
the intermittent crash or establish a complete recovery outline. Its records are
in `local_attempt6_pickup_fatal2`; the earlier wrapper import failure is retained
separately in `local_attempt6_pickup_fatal`. A full six-response local replay is
staged but has not run. No production workaround or validation relaxation was
introduced for the unexplained crash.

The combined CCA, outline, context, and offline regression run finished with
650 passing tests. A new actual attempt is staged separately in
`experiments/20261010T140822Z`, with fatal tracing enabled and the same generation
limits; its generated outcome and independent audit are recorded separately.


Attempt 7 produced seven actual provider responses, four proposed events, two
context-retrieval turns, and 35,723 tokens. The exact square-buffer placement
(`place_part`, `recovery-resource-3@localhost`) and subsequent `move_home` by
`recovery-resource-4@localhost` with `part_name=null` passed RA, physical
projection, and CCA. The first `gear_small` pickup was rejected by the definite
`SAFE_shared_area_mutex` counterexample; its exact cell, interval, and AP values
were included in the following prompt. The model then independently proposed
`move_home`. During the next pickup's route-evidence preparation, the interpreter
exited with SIGSEGV (-11), after 463.02 seconds and 2,049,708 KiB peak child RSS.
This was not the generation deadline. No complete outline or nominal continuation
was established.

`experiments/20261010T140822Z/attempt_7_recovered` separately preserves the last
durable state. Its independent audit verifies both accepted events and their
state/history chain, with no audit errors or equipment calls. Original
observations and frozen source/native fingerprints match. The crashed run's
final in-memory guard counters remain unavailable and are not reported as zero.

A full fresh-state replay then consumed six exact saved responses under gdb and
Python's debug allocator, preserving the original structured request adapter,
provider worker, parser, and cancellation path while blocking HTTP. It reproduced
SIGSEGV after `move_home` physical preparation, during the following validation.
The debugger caught a dereference of `0xdddddddddddddddd`, the debug allocator's
freed-memory fill, in Python function/frame handling. This narrows the failure
to memory lifetime corruption but does not yet identify its cause. The replay
is explicitly labelled as replayed, not a new provider capture; gdb's exit code
0 does not imply success. Records are in `local_attempt7_full_debug`. Periodic
stack sampling is being investigated separately from fatal tracing; no
production workaround or evidence requirement has been changed.

A controlled standard-library-only reproduction identifies the periodic stack
sampler as a concrete corruption source in the installed Python binary. With
`faulthandler.dump_traceback_later` enabled, two runs failed after 0.11 and 0.67
seconds (the second with SIGSEGV); matching fatal-only controls both completed
20 seconds. Disassembly shows the background sampler changing code-object
reference counts without holding the GIL, consistent with the freed
`co_zombieframe` pointer caught in the full replay. No ROS, project, API, or
third-party code was imported by the minimal probe. Detailed evidence is in
`attempt_7_gpt-5.6-sol/crash_diagnostics/faulthandler_reproduction`. A separate
full saved-response replay retains fatal tracing and gdb while removing both
periodic timer entrypoints. Its result remains pending; validation requirements
and source observations are unchanged.

The full fatal-only saved-response replay completed normally in 617.02 seconds
under gdb and the debug allocator. It replayed exactly the seven retained
responses with zero API/equipment calls and unchanged source/code/native hashes.
Square placement, `move_home` by `recovery-resource-4@localhost`, and the final
`gear_small` pickup by `recovery-resource-3@localhost` passed the runtime RA,
physical projection, and CCA checks. The earlier pickup still received the same
modeled mutual-exclusion rejection. No validation requirement was weakened.
The full reproduction with periodic sampling had crashed; the same lifecycle
without periodic sampling reached this result.

Nominal continuation remains correctly `not_ready`: `environment_8` requires
`recovery-resource-4@localhost.resource_state= picked`, but the projected value is
`idle`; `ed9a5f9c8c4b463496402821945cfe72_pending_3` requires
`held_part=KET4_Square_4mm` on `recovery-resource-3@localhost`, which currently
holds `gear_small`. These are explicit conditions for further proposal feedback.
The replay stopped before requesting any eighth response. It is not a new model
attempt, complete recovery outline, or execution. Records are in
`experiments/20261010T140822Z/local_attempt7_fatal_only`, including a separate
verification summary and the unchanged-source report. The real API generation
deadline remains 600 seconds; the 1,200-second cap was solely for this instrumented
local diagnostic.

The geometry calculation now reuses repeated FK results only within a fixed-input
motion checkpoint. On the exact saved real `gear_small` pickup fixture, complete
projection time decreased from 28.21 to 18.41 seconds (34.7%). The complete
projection, CCA result, counterexample, and monitor-history hashes are identical;
the same mutual-exclusion rejection remains. This measured result is separate
from the earlier synthetic approximately threefold improvement. No evidence
bound or validation requirement changed, and no cache crosses validation calls.
The comparison made no API, native, or equipment calls. Its source hashes, exact
outputs, fixture hash, and timing are retained in
`experiments/20261010T140822Z/geometry_performance/report.json`. Focused regression
results are recorded there separately when complete.

The geometry optimization regression run completed with 310 passing tests in
43.02 seconds; Poetry, Python compilation, and whitespace checks passed. The
implementation is frozen before the next actual model attempt.

Attempt 8 used actual `gpt-5.6-sol low` responses with fatal-only tracing and the
verified geometry optimization. It did not crash. Nine provider responses
contained five proposals and three context-retrieval turns (45,956 tokens). The
accepted square placement, `move_home` by `recovery-resource-4@localhost`, and
`gear_small` pickup by `recovery-resource-3@localhost` independently passed the
saved RA/state/CCA-history audit. The source capture and implementation/native
hashes were unchanged during the attempt; equipment/execution guard calls were
zero. Nominal continuation and the complete outline remain unproven.

The first pickup received the established mutual-exclusion rejection. The model
then retrieved the failed resource's snapshot and measured pose and proposed
`move_home`. Its later gear-placement proposal reached a checked native collision
between `Gear_Plate/Gear_Shaft_1/collision` and `gear_small/link/collision`; this
placement was rejected and did not alter the accepted prefix. The fixed
600-second generation deadline was exceeded; process shutdown and cancellation
completed at 639.90 seconds, with 2,148,912 KiB peak child RSS. The runner failed
to catch Python 3.10's distinct `asyncio.TimeoutError`, so its original report's
`error=null` is preserved and corrected only in a separate
`attempt_8_gpt-5.6-sol/outcome_annotation.json`. Exact provider captures, native
requests/results, and `prefix_audit.json` are retained under
`experiments/20261010T144129Z`. This is a partial validated outline, not a complete
recovery result; the gear-placement geometry is being investigated locally.


A separate saved-proposal diagnostic now validates Attempt 8's exact `gear_small`
placement after its three previously accepted events. It uses the unchanged
capture and historical version-5 prefix with version-6 calculations for this
pending event only. The recorded attachment link and the motion control link are
accepted as equivalent only through their verified fixed URDF connection; a
movable connection remains rejected. The three isolated native previews,
continuous mating evidence, projected fixture attachment, RA, and both CCA rules
passed. The event was appended, leaving four accepted events. Nominal continuation
remains `not_ready`: the recorded Square `place_approach` requires `resource_state`
`picked`, while its current projected value is `placed`. This is a local diagnostic,
not a fresh successful model attempt or executed recovery.

The diagnostic completed in 604.57 seconds: physical preparation took 528.65
seconds and the runtime validation retry took 44.65 seconds. A private one-entry
owner projection cache reused only its own successful calculation with identical
complete inputs and projector identity; the real projection took 177.94 seconds
initially and 2.28 seconds on the matching retry. CCA was checked independently.
Fifteen focused cache tests cover mutation isolation, changed inputs, missing
versus null, external route-file changes, untrusted supplied results, and the
single-entry limit. Source observations, historical records, provider response,
and code/native fingerprints stayed unchanged during this diagnostic; API and
forbidden equipment calls were zero. Original failed fixed-link diagnostics remain
separate. Records are under `experiments/20261010T151845Z`, including
`local_attempt8_placement_v6_retry/report.json` and
`owner_projection_cache_verification.json`. Repeated continuous mating proof
calculations remain a measured performance bottleneck before the next fresh
600-second model attempt.


The exact placement diagnostic was repeated after reusing fixed FK origins and
one locally produced successful continuous mating proof. It again accepted the
same fourth event and retained the same nominal obstruction. Total time decreased
from 604.57 to 179.47 seconds; physical preparation decreased from 528.65 to 98.97
seconds. All six mating checks returned the same complete 2,617-cell proof as the
baseline. The first calculation took 43.67 seconds; subsequent exact-input checks
took 0.139–0.169 seconds. Accepted events, derived and symbolic state, and CCA
monitor history match exactly. Independent native reruns have different measured
execution durations and artifact paths/hashes; only their three corresponding
response fingerprints differ in the projected evidence. All other native request,
trajectory, collision, and geometry fields match. Full differences and the exact
comparison are retained in `mating_optimization_comparison_final.json`; no API or
equipment call occurred. The mating/preview/outline regression run passed 490
tests, and the broader validation/UI/retrieval integration run passed 875 tests.

A separate audit of the recorded Square task found a configuration contradiction:
its declared origin was at z=1.0239916, above the actual supporting table at
z=1.015. Its 4 mm collision box fits inside the recorded 4.11 mm through-hole;
the complete plate mesh has no positive overlap with that footprint. There is
no declared retained fixture attachment for this part. The geometry declaration
was corrected to the measured-model table support height, with the original
capture unchanged. `square_recorded_release_support_audit.json` and
`square_geometry_repair.json` retain the exact source geometry and calculation;
311 surrounding geometry tests passed. A new capture is required before a fresh
model attempt because the configuration fingerprint changed. Attempt 9 remains
staged with explicit source/hash arguments, fatal-only tracing, and the unchanged
600-second/80-turn limits. Eleven mocked runner tests pass, including source
immutability and cancellation. Neither this configuration repair nor the saved
placement replay establishes a complete recovery outline or executed assembly.


A new isolated Gazebo failure capture was recorded on October 10 at 16:18:26 UTC
(`capture_ab5fbd8696e24a5284470ec127fa5334`, SHA256
`f6cb7a09f7a49e05fb1ae4987d4261d19ec962389914b1fe5b5bdcb799d1fd31`).
Both pickups and the intentional lowering failure were observed, and the Square
geometry uses the corrected z=1.015 declaration. No LLM request or recovery
execution occurred during capture. The previous source and UI records remain
unchanged. Detached Manual input preparation succeeds, but this capture is not
sufficient for whole-scene physical validation: `RGOCG4-50_Round_4mm` has recorded
angular speed 0.0028648084 rad/s, above the unchanged 0.001 rad/s rest criterion.
The top-level `physical_evidence.unresolved` list omitted this child
`stationary_part_evidence` status. The first capture audit checked only that
coarse list; its original reports are preserved and corrected separately in
`experiments/20261010T160847Z/fresh_capture_audit_correction.json`. The next audit
requires supported evidence for every unheld part.

New attempts now use physical preview contract 7 while RA remains contract 2.
Exact nominal task coverage accounts for each recorded step, guard, parameter,
calculated output, custody change, and simulation `snap_part_to_slot`. The snap
has separate scene-endpoint evidence; it is not called a continuous movement.
Owner task context is carried from independently replayed Cartesian endpoints
using the existing runtime bookkeeping. Changed declared product geometry,
unsupported steps, untrusted owner facts, and conflicting declared state effects
remain unresolved. Historical contracts retain their recorded behavior. Focused
owner/witness/workflow tests passed 559 cases; owner-authority checks passed 476;
effect-consistency checks passed 354. Endpoint verification separately passed
735 Python, 33 native file, and 11 native action cases, with bootstrap, Poetry
checks and compilation. These overlapping suite totals are separate runs.

A no-API replay of retained Attempt 8 responses against the new capture first
stopped at an exact native model-fingerprint mismatch (28.34 seconds). Its file
adapter used the previous capture's SRDF bytes. A separate parameter archive now
uses the exact new captured URDF/SRDF and archived local MoveIt declarations;
the native fingerprint matches `2c8664d56332c8406155a1121b3610e6d93054300f260f2079d8b5f5c2636602`
exactly, and the same saved native Goal completes. URDF, kinematics and planning
parameters are equal to the old archive; SRDF bytes differ. The matching proof
is retained in `model_preview/provenance.json`, without changing either capture.
The fresh capture does not itself archive every typed native startup parameter;
that future capture gap is recorded separately and is not claimed fixed.
A second no-API replay correctly stopped at the unresolved Round part's rest
evidence (31.19 seconds). Neither replay appended an event; both preserve source,
implementation and native hashes and report zero API/equipment/execution calls.
Attempt 9 remains unstarted pending a new capture with sufficient recorded rest
evidence; no thresholds or observations are changed to obtain acceptance.


A replacement isolated capture was saved at 17:03:11 UTC on October 10 as
`capture_f923020bf9ce496eb443826291f8f2c2` (SHA256
`e771f6b9d0f43e0f9efb2d51d779cdb1c92835313f14b28bd7dfde0897cb6e5e`).
The capture waited for recorded stationary evidence for every unheld part: the
first two samples retained unresolved moving parts, and the third supported all
of them under the unchanged criteria. All owned Gazebo processes were then
stopped. The exact native startup parameter files were archived while alive;
the file-mode native fingerprint matches the saved model
`cf377822f8e5148dc56343121aa67edc5f58064fd1894b683be1fb662b56c2a5`.
Independent detached preparation passed the stricter child-evidence checks,
actual pickups/drop, corrected geometry, exact selected safety specifications,
initial AP source fingerprints, retained history, and Offline Manual status.
It made zero API/live/execution calls and preserved both previous captures.
Evidence is in `experiments/20261010T160847Z/fresh_capture_f923_audit/report.json`
and `experiments/20261010T160611Z/fresh_capture_stationary_2/`. This confirms
initial saved evidence and preparation, not acceptance of future motions.
A fresh-source replay of the retained Attempt 8 responses is staged separately
in `experiments/20261010T170458Z`; actual Attempt 9 remains unsent until its
local validation outcome is known.


Fresh-source local validation then replayed all nine retained Attempt 8 responses
without API calls. It accepted four events after RA, physical, and CCA checks;
the earlier pickup remained rejected for a definite mutual-exclusion violation.
The remaining nominal resource3 task required `resource_state="picked"` while
the accepted projected state was `placed`. The replay stopped before a new
response, preserving the capture and implementation; elapsed time was 624.60s,
including 474.08s of physical preparation and validation. This is a local replay,
not a fresh complete model result. Exact records remain in
`experiments/20261010T170458Z/local_attempt8_fresh_v7/`.

Actual GPT-5.6-sol low Attempt 9 ran afterward with the unchanged 600-second and
80-turn limits. It independently proposed `recover_resource_4_home` first;
that event passed RA, physical projection, and CCA and was independently audited
against its exact provider capture. The attempt stopped cleanly after 179.34s
(183.50s process time), three provider responses and 13,375 tokens, with one
accepted event. It did not complete or time out. The next unfinished nominal
`place_approach` had a successful physical preview but RA could not bind the
calculated `place_targets.approach_pose` / `target_pose` x/y/z values to its
independent `move_cartesian` capability checks. No nominal state was committed.
This concrete parameter-preparation gap is under repair; missing evidence was
not bypassed. Peak child RSS was 2,339,428KiB; no forbidden live/execution calls
occurred, and source/Python/native fingerprints stayed unchanged. Report,
independent prefix audit, and exact local RA replay inputs are retained under
`experiments/20261010T170458Z/attempt_9_gpt-5.6-sol/`.


The exact Attempt 9 nominal task was replayed locally after repairing RA's
calculated `place_targets` parameter witnesses; RA became FEASIBLE. Focused RA
and outline verification passed 483 cases, with the same 12 unrelated legacy
AP/KMR fixture failures recorded separately. The next observed blocker was CCA:
it stopped at an early uncertain occupancy boundary, hiding a later supported
Square region entry. CCA now retains possible safe monitor states through later
observations while preserving the earliest uncertainty unless a definite
violation is established. The unchanged saved trajectory proves precedence
failure at observation 5 (`ap001=[true]`, `ap002=false`); mutual exclusion remains
FEASIBLE. No occupancy threshold, AP meaning, path, or geometry was changed.
The CCA repair passed 468 focused tests, Poetry checks and compilation.
A full retained-evidence retry then returned that exact factual rejection to PA
and restored `outline_ready` for another proposal; it did not mark the outline
complete or commit nominal state/history. This check took 85.02 seconds with
zero API/native/live/execution calls, and preserved source/implementation
hashes. Its feedback is recorded in
`experiments/20261010T170458Z/local_attempt9_nominal_cca_retry/feedback_summary.json`.
Fresh actual Attempt 10 is kept separately in `experiments/20261010T173709Z`,
with the same capture, 600-second/80-turn limits, and fresh provider responses.
The production capture provider still omits a complete typed native startup
parameter archive; the experiment capture explicitly archived those actual
startup files and verified its exact native model fingerprint separately.


Actual Attempt 10 reached the unchanged 600-second generation deadline while
preparing its fourth proposed event. GPT-5.6-sol low returned seven real responses,
four proposed events and two retrieval turns (four exact context references),
using 41,682 tokens. Three events were accepted in the model's original order:
`move_home` for `recovery-resource-4@localhost` with `part_name=null`, `place` of
`KET4_Square_4mm` at `Buffer For Machined parts` by
`recovery-resource-3@localhost`, and `pick` of `gear_small` by that resource.
The fourth proposal was `place` of `gear_small` at `assembly_board-v1`; its
physical preparation was interrupted by the deadline, so it was not appended.
The nominal Square `place_approach` was rejected for its supported precedence
violation and returned factual feedback before the second recovery proposal.

Independent audit verified all three accepted events against exact provider
captures, RA candidates, physical bindings, CCA verdicts, projected states and
monitor history. Nominal continuation and the complete outline remain
unestablished. Runtime was 623.29 seconds and total process time 627.59 seconds,
including cancellation and artifact cleanup; cancellation completed. Peak child
RSS was 4,006,100KiB. Source, parent capture, Python and native fingerprints
stayed unchanged, and no forbidden live or execution call occurred. Original
report, precise timings, token usage and independent audit remain under
`experiments/20261010T173709Z/attempt_10_gpt-5.6-sol/`. This is a valid partial
outline, not a complete success. The next work is a measured repair of repeated
archive copying and JSON writing; generation limits and validation requirements
are unchanged, and another API attempt has not started.


Before Attempt 11, narrow read-only-copy repairs preserved exact saved nominal
evidence, projected state and CCA history while reducing one full retained retry
from 70.2226s to 57.6137s (17.96%). The complete saved session bytes were
identical. Verification passed 783 focused cases, 11 runner guards, Poetry checks
and compilation. The writer retained its streamed memory behavior.

Actual GPT-5.6-sol low Attempt 11 used exactly the same initial prompt hashes as
Attempt 10. It stopped after 348.04s (353.04s process time), not at the deadline.
Four real responses used 20,644 tokens: grounding, `move_home` for
`recovery-resource-4@localhost`, one context request for that resource's
capabilities and `gear_small`'s target, and `place_approach` for
`recovery-resource-3@localhost` carrying `KET4_Square_4mm` at
`Buffer For Machined parts`. The first event passed the independently audited
provider/RA/physical/CCA chain. The second proposed `resource_state="positioned"`
and `part_state="in_gripper"`; it was not appended.

The RA successor union had a matching `place_approach` contributor and a feasible
`compute_place_targets`; its `move_cartesian` checks lacked scoped witnesses for
`place_targets.approach_pose` and `target_pose`. Independently, the owner projection
reported a conflict between its declared effect and the proposed `part_state`.
These are retained as separate findings; the broad RA summary must not be read
as an empty union. The runtime left the accepted prefix and monitor history
unchanged. Nominal continuation remains incomplete. Peak child RSS was
3,858,100KiB; source/Python/native fingerprints stayed unchanged and forbidden
calls were zero. Exact report, findings and audit are retained in
`experiments/20261010T175921Z/attempt_11_gpt-5.6-sol/`. Another API attempt remains
held until the concrete parameter and feedback paths are repaired and checked
locally. The stronger audit is staged separately so historical audit files are
not rewritten.

The Attempt 11 buffer parameter repair made its exact RA proposal `FEASIBLE`.
The separate owner-effect check now returns factual revision feedback for the
established `part_state` conflict (`in_gripper` requested, `in_transit` declared),
without a native preview, append, projected-state change, or monitor commit.
The exact local retry took 25.2097s without profiling; its complete saved session
bytes matched the profiled replay. The profile exposed streamed artifact writing
and copied historical evidence as costs; the small isolated RA-copy prototype
was not adopted. Verification included 1,091 focused tests, 23 producer tests,
27 runner/audit guards, Poetry and compilation. The full feasibility suite
retained 12 unrelated legacy AP/KMR failures. Separate records are under
`experiments/20261010T181625Z/`.

Actual GPT-5.6-sol low Attempt 12 retained the same initial prompts and immutable
`capture_f923020bf9ce496eb443826291f8f2c2` observations. Seven real responses used
44,563 tokens. Two public retrieval turns requested five references. Four events
were proposed; three were appended after RA, physical projection and both CCA
checks: `recovery_move_r4_home_1`, `recovery_place_ket4_buffer_2`, and
`recovery_pick_gear_small_r3_3`. The independently replayed original Square
nominal approach remained rejected by its precedence rule. The final proposal,
`recovery_place_gear_small_r3_4`, was still in physical preparation when the
600s generation deadline fired. Cancellation completed; process time was
611.02s and peak child RSS was 3,931,156KiB. This attempt is incomplete, not a
model or execution success.

The separate prefix audit verified exact provider authorship, all three accepted
RA/physical/CCA chains, complete scene propagation and monitor history. Source,
parent capture, Python and native hashes were unchanged; forbidden calls were
zero. Reports and audit are preserved in
`experiments/20261010T181625Z/attempt_12_gpt-5.6-sol/`. The completed pickup's
physical preparation took 78.15s and validation retry 32.77s. The next step is a
local, exact-input positive preparation/validation profile with retained native
responses; no unchanged API repetition or increased generation limit is planned.

The positive Attempt 12 pickup was then replayed locally with its independently
reconstructed two-event prefix and exact provider proposal. Each of four saved
native responses was reused only after every typed Goal field matched. The
baseline unprofiled Python preparation took 69.624s and retry 32.951s. Selective
profiling identified repeated ROS message subtree copies, repeated static joint
transforms in the support-departure proof, and detached copies of historical
validation data. The streamed writer was retained: its actual unprofiled runtime
writes cost 6.49s, substantially less than their profiler-inflated timings.

Narrow copy and static-calculation repairs reduced the same preparation to
57.535s and retry to 31.241s: combined 102.575s to 88.776s (13.45%). The complete
prepared evidence and complete resumed runtime session retained identical
semantic fingerprints, including all projected states and CCA monitor history.
The replay's archive scope used the original turn 6 as its one full turn; this
is a controlled before/after local comparison, not another real model attempt.
No new API, native planning, or equipment call was made. Evidence and profiles
are retained under `experiments/20261010T184631Z/`. Verification passed 494 nearest
outline cases, 11 ROS conversion cases, 45 support-departure cases, and 30
conservative owner-effect expression cases. The latter guard leaves the current
capture's 48 supported resource updates unchanged. Attempt 13 is staged in
`experiments/20261010T190600Z/`, with the same source, initial prompt hashes,
model/effort and 600s/80-turn limits; it has not started at this update.

After the final entry-copy regression checks and 1,120 combined tests, actual
GPT-5.6-sol low Attempt 13 ran with the same capture and initial prompt hashes.
It stopped at `NEEDS_CONTEXT` after 314.85s of generation (329.43s process time),
before the deadline. Five actual responses used 28,628 tokens. Two retrieval
turns requested four public references. The newly named `recovery-step-1`
(`move_to_named_pose`, `recovery-resource-4@localhost`, explicit `part_name:null`)
passed and was independently audited. The original nominal Square approach was
again rejected by its precedence rule without a state or monitor commit.

The next actual proposal was `recovery-step-2`, a loaded `place_approach` by
`recovery-resource-3@localhost` to `Exit`, retaining `KET4_Square_4mm` and proposing
`part_state="in_transit"`. RA found the matching successor union; it did not fail
because the event name was new or the union empty. The owner calculation lacked
a declared support surface height for `Exit`, leaving the approach/target xyz
parameters unresolved. Physical preparation independently reported that the
recorded geometry could not establish the requested loaded destination. CCA was
not checked for this unresolved proposal; nothing was appended. This is neither
a geometric rejection nor complete outline success. All source, parent capture,
Python and native integrity gates passed; no forbidden call occurred. Exact
findings, usage and the independent one-event audit are in
`experiments/20261010T190600Z/attempt_13_gpt-5.6-sol/`. Another API call remains
held while the recorded `Exit` evidence boundary is assessed.


### 2026-10-10: measured retrieval-generation overhead and staged Attempt 14

- Replayed only the exact saved API13 retrieval responses for turns 3 and 4, through the original provider worker with HTTP, native planning and execution blocked. Both reconstructed prompts matched their saved provider requests exactly; accepted prefix, projected state and monitor history remained unchanged.
- The scoped profile measured 223.67 s in five full debug publications out of 239.17 s under cProfile. Prompt construction was 3.01 s and artifact writing 0.012 s; profiled durations are diagnostic overhead, not API latency.
- Removed only the redundant end publication for contract-2 successful `need_context` turns that immediately continue. The synchronous artifact writer receives the current session; the next loop still publishes independent snapshots before its first await. Terminal, failed, paused, legacy and exhausted paths retain their publications. Exact request/audit/resume bytes and mutation isolation are covered by regressions.
- Unprofiled exact retrieval replay improved from 87.738 s to 51.018 s (41.85%). Original prompts/responses and retrieval artifact facts match; only local replay provenance paths differ. Capture and original session hashes are unchanged. Evidence: `experiments/20261010T190600Z/retrieval_optimization.json`.
- Verification: 551 focused tests passed, 11 targeted publication/artifact tests passed, `poetry check` and full Python compilation passed. No archive pruning, validator weakening, changed generation limits or suggested recovery events were introduced.
- Attempt 14 is staged under `experiments/20261010T193410Z`, using the same `capture_f923020bf9ce496eb443826291f8f2c2`, model/effort, selected specifications and 600-second/80-turn controls. Its API is not started; the original API13 `Exit` proposal must first complete the separate local validation replay. The pure geometry calculation result is recorded separately from full RA/physical/CCA acceptance.


### 2026-10-10: Attempt 14 retained four accepted events and timed out

The separate API13 `Exit` replay completed before the fresh attempt. The repaired
owner calculation established the receiver height from recorded geometry; RA
was `FEASIBLE`. Its native direct path collided with `ur5e_3_upper_arm_link`, and
the alternative route lacked the required fixed starting orientation. The
workflow returned factual revision feedback without appending or changing its
accepted state/history. That local result is not another model attempt.

Actual GPT-5.6-sol low Attempt 14 used eight provider responses and 50,220 tokens,
with two retrieval turns and five proposed events. Four passed: the resource-4
home event, Square placement at `Buffer For Machined parts`, resource-3
`gear_small` pickup, and its placement at `assembly_board-v1`. The exact fifth
provider proposal was `pick_ket4_for_nominal_continuation`, a resource-3 pickup
of `KET4_Square_4mm`. Its physical preparation was interrupted by the unchanged
600-second deadline. Workflow elapsed time was 606.109 s, including delayed
cancellation; process elapsed time was 613.055 s. This is a timeout, never a
within-deadline completion. Completed preparation/retry stages totalled
410.770 s; nominal continuation remained `not_ready`.

The separate audit verified all four exact provider/state chains, saved
RA/physical evidence bindings, and recomputed CCA histories. It did not perform
new RA derivation or native preview. Capture, parent, Python and native hashes
were unchanged and forbidden calls were zero. Reports, the independent audit,
and a separate deadline attestation are retained under
`experiments/20261010T193410Z/attempt_14_gpt-5.6-sol/`. No recovery execution was
performed. Attempt 15 is only staged: the next local work validates the actual
fifth proposal and recorded nominal tasks, and measures archival of full nominal
proofs without pruning runtime validation inputs. New audit guards require an
explicit terminal timestamp within the deadline, a clean reaped process, and
manifest membership for every selected provider/validation proof artifact.


### 2026-10-10: complete nominal proof archival and local fifth-proposal check

New attempts save the full nominal validation record before replacing repeated
in-memory diagnostic copies with its exact relative path, SHA256 and accepted
state/source/model/safety binding. The complete proof remains downloadable and
expandable; missing, changed or incompatible files cannot establish completion.
Pending nominal tasks/scenes, accepted state, monitor history and validation
inputs remain intact. Historical embedded proofs retain their prior behavior.

Verification passed 1,335 cases across nine focused suites (53.35 s), 72 UI
cases, 111 targeted archive cases and 48 runner/audit guards. `poetry check`, full
Python compilation and diff checks passed. An earlier sandbox selector stall
was interrupted/reaped and is not recorded as a complete test pass. Evidence is
under `experiments/20261010T195300Z/automated_verification.json`.

The exact API14 fifth provider proposal is now undergoing a separate local
1200-second diagnostic from its four-event independently audited prefix, with
new private native file calculations and subsequent recorded nominal tasks.
There are no new provider, equipment or execution calls. A separate retrieval
benchmark will include archival cost and compare the complete hydrated proof,
public prompts and accepted state/history before any actual Attempt 15 launch.


The retained-prefix diagnostic completed in 356.326 s with no API/forbidden call
and unchanged source, session-source, Python and native hashes. The actual fifth
Square pickup passed and appended locally. The recorded nominal `place_approach`
also passed. The next recorded `place_insert` produced two complete native motion
results and a separate complete snap endpoint, but exact nominal program coverage
correctly remained unresolved. Its `lift` step resolves `dz=0.052500000000002656`
from the current nominal task context (target z=1.3384999752316755); the preview
instead used the prior pickup travel height (target z=1.500773097154814). Insert,
release and snap step coverage had passed. This is a parameter-source mismatch,
not floating-point noise or a missing native result. The five-event recovery
prefix remained retained; no execution or complete nominal result is claimed.
The exact witness and compact replay input are saved under
`experiments/20261010T195300Z/local_api14_pending_replay/`. Attempt 15 remains held
for this narrow repair and the separate archival performance comparison.

The nominal departure repair now retains the exact authoritative `_task_ctx` for
`place_insert`, including its explicit null fields, and binds cached coverage to
that recorded context. The corrected lift target is z=1.3384999752316755. A new
local native retry passed insertion, the separate snap endpoint, lift, RA and
CCA, then exposed a final-goal reader defect: supported `part_state=assembled`
and `part_location=assembly_board-v1` were being shadowed by the retained capture
fields `current_state=misplaced` and `current_location=null`. Version-2 completion
now reads its supported fields first, distinguishes absent from null, and keeps
historical behavior. The source capture was not modified.

The subsequent exact retained-evidence retry completed the nominal suffix in
258.731 s, including persistence, with no new API, native or live calls. The
independent audit matched all five accepted events to their original API14
provider responses, replayed both CCA histories and derived state transitions,
validated exact nominal task ownership/program coverage and replacement evidence,
and checked final goals. `accepted_chain_valid`, `nominal_chain_valid` and
`nominal_complete` are true, with no unfinished task or audit error. This is a
local saved-provider replay; `fresh_generation_valid` and fresh `outline_complete`
remain false. It establishes no recovery execution. Evidence is in
`experiments/20261010T195300Z/local_api14_nominal_goal_retry_2/independent_audit.json`.

Final focused verification passed 1,377 cases across nine suites, 75 runner/audit
guards and the earlier 72 UI cases. `poetry check` and full Python compilation
passed. The staged fresh-attempt gate additionally binds every provider request
and response to its current process, saved source, empty generation seed and
artifact root. The exact archival-cost benchmark precedes actual Attempt 15;
its 600-second deadline and 80-turn bound remain unchanged.


### 2026-10-10: archival benchmark and fresh Attempt 15 deadline result

The exact saved retrieval benchmark includes full archival creation, writing and
hashing. Generation time fell from 48.720 s to 16.878 s (65.36%); the latter
includes 6.280 s to archive the proof. Total benchmark time fell from 72.241 s to
36.977 s and peak RSS from 3,177,780 to 1,593,672 KiB. Both public prompts, the
complete hydrated nominal proof, accepted state/history and retrieved context
matched exactly. Original artifacts were unchanged; API, native and forbidden
calls were zero. Results are in
`experiments/20261010T195300Z/nominal_archive_comparison.json`.

Fresh Attempt 15 used the same immutable `capture_f923020bf9ce496eb443826291f8f2c2`,
`gpt-5.6-sol`, low reasoning, 80 turns and a 600-second deadline. Eight actual
provider responses contained two retrieval requests and five recovery proposals
(51,135 tokens). Four events were accepted before the deadline. The fifth
Square pickup also validated, but its synchronous retry ended at 640.147 s; the
explicit stage boundary raised `TimeoutError`. This is not within-deadline
completion. The process was fully reaped, with no recovery execution.

The independent audit validates all five exact provider/state chains, saved
RA/physical bindings and replayed CCA histories. Fresh-generation identity,
manifest membership, source, parent, Python and native integrity all pass;
forbidden calls and audit errors are zero. Square `place_approach` and
`place_insert` remain unfinished, so `nominal_complete`,
`completed_within_deadline` and `outline_complete` are false. This audit performs
no new RA derivation or native preview. Records are retained under
`experiments/20261010T195300Z/attempt_15_gpt-5.6-sol/`.

Exact stage boundaries attribute 5.463 s to input preparation, 113.516 s to
generation (37.723 s provider calls and 75.793 s other generation work),
304.170 s to physical preparation plus evidence persistence, and 216.998 s to
validation retry. The final physical stage used 44.394 s and its retry 50.921 s.
The next work is measured performance repair and exact saved-result parity;
an unchanged fresh Attempt 16 has not been launched.

### 2026-10-10: Attempt 16 and exact named-pose endpoint repair

The user removed the ten-minute comparison deadline. Fresh Attempt 16 used the
same immutable `capture_f923020bf9ce496eb443826291f8f2c2`, unchanged initial
prompts, `gpt-5.6-sol` with low reasoning, and the existing 80-turn bound. It
finished normally in 584.12 s; no timeout occurred. Nine actual provider
responses used 56,709 tokens and proposed five events. Four appended after RA,
physical projection and both CCA requirements. The independent audit confirmed
all four provider/state/evidence chains, fresh-generation identity and unchanged
source, code and native inputs, with no forbidden call or audit error. Nominal
continuation remained incomplete.

The fifth proposal, `clear_assembly_board_resource_3` / `move_home`, exposed a
native interpolation defect: `ur5e_3_wrist_3_joint` ended at 5.142049823384552
instead of its exact named target 5.142049823384553. The final waypoint now uses
the exact target before the existing bounds, contact and trajectory checks.
Post-timing endpoint equality is checked; no tolerance or contact rule changed.
`make bootstrap-gazebo` completed 16 packages and the native suite passed 46
tests plus 42 subtests, including collision rejection at interior and final
states. Replaying the exact saved Goal produced exact first/final configurations
and retained all peer joints.

A separate 69.63 s local replay then accepted this same fifth provider proposal
through RA, physical projection and CCA. It made one private native calculation,
zero API/live/execution calls, and preserved all recorded source/code hashes.
The nominal result correctly remains `not_ready`: the next recorded task requires
`resource_state=picked`, while the supported projected state is `idle`. This is
factual feedback for another LLM proposal, not an inserted recovery event.
Evidence is under `experiments/20261010T211008Z/`, including
`home3_endpoint_diagnosis/` and `local_api16_home3_replay/`.

The next fresh attempt will exercise the production Offline Generate operation.
The implementation connects saved-model native calculations and same-proposal
validation retries to that operation, retains immutable companion model files,
and allows completed outline review to reload separately from primitive
composition. Focused integration verification is in progress. No user-observed
Gazebo acceptance or recovery execution is claimed for this work.

### 2026-10-10: fresh Attempt 17 completes the offline outline

Fresh `gpt-5.6-sol` with reasoning effort `low` completed the production Offline
Generate operation from unchanged
`capture_f923020bf9ce496eb443826291f8f2c2/observations.json` and its separately
stored, verified `native_preview_model`. Generation took 1,029.50 s, including
33.27 s in eight actual provider requests. It used 51,759 tokens, two public
context retrieval turns and 21 private file-native calculations. Gazebo,
perception, controller, SPADE transport, primitive generation and execution
calls were blocked; no prohibited call occurred. No event count or recovery
sequence was added to the prompts. Initial prompts matched Attempt 16 exactly.

The five independently proposed and accepted events, in order, were:

| `outline_id` | `event_name` | `resource_jid` | `part_name` |
| --- | --- | --- | --- |
| `clear_failed_robot_from_assembly_board` | `move_home` | `recovery-resource-4@localhost` | `null` |
| `stage_ket4_at_buffer` | `place` | `recovery-resource-3@localhost` | `KET4_Square_4mm` |
| `pick_misplaced_gear_small` | `pick` | `recovery-resource-3@localhost` | `gear_small` |
| `place_gear_small_on_assembly_board` | `place` | `recovery-resource-3@localhost` | `gear_small` |
| `reacquire_ket4_for_nominal_continuation` | `pick` | `recovery-resource-3@localhost` | `KET4_Square_4mm` |

The independent saved-evidence audit passed all five provider/RA/physical/state
chains and replayed both exact CCA specifications, `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm`, with retained monitor history. It
verified nominal tasks `pending_3` and `pending_4` and their final goals:
`fresh_generation_valid`, `accepted_chain_valid`, `nominal_chain_valid`,
`nominal_complete` and `outline_complete` are true; `unfinished_task_ids` and
`errors` are empty. Source, parent capture, frozen implementation, native
inputs, provider records and manifest membership verified unchanged. The audit
checked saved RA and physical evidence and replayed CCA; it did not rerun RA
derivation or native previews. The original attempt report remains unchanged;
the completed assessment is recorded separately in
`experiments/20261010T213423Z/audited_result.json` and `independent_audit.json`.

Production Load saved result also passed against this real result, restoring
all five events and the complete bound nominal proof into a separate Manual
review run. Primitive composition remains waiting and execution is disabled.
All 40 protected source files and 62 copied provider/validation records retained
their hashes; the reload sent no provider, native or live call. See
`saved_result_restore_verification.json` in the same experiment directory.

After the audit, the progress display was corrected to show `generating_outline`
while automatic offline physical calculations are active instead of retaining
an intermediate `blocked` label. This changes no validation rule or frozen
Attempt 17 record. The original observation file remains unchanged.

Automated preparation verification passed 414 backend/offline cases, 347
workflow/offline cases, 543 outline/request fixture cases and 90 runner/audit
guards before Attempt 17; these suites overlap and are not summed. The earlier
broader run's seven fixture failures were corrected and both affected suites
passed; its existing NiceGUI warning is retained separately. Native verification
passed 46 tests plus 42 subtests, and `make bootstrap-gazebo` completed 16
packages. Previously recorded unrelated feasibility failures are not claimed
fixed by this work.

After the progress-label change, both focused workflow/offline suites passed
again: 347 tests in 35.28 s. `poetry check`,
`poetry run python -m compileall -q cais_spade_llm ros2` and `git diff --check`
passed. Poetry reported its existing metadata deprecation warnings.

This is a complete offline outline and model-based nominal-continuation result.
User-observed UI/Gazebo acceptance, separate primitive composition, execution
authorization, recovery execution and observed assembly completion remain
separate subsequent milestones. No new recovery execution was performed.

### 2026-10-12: four cases and reserved evaluation (in progress)

The same public prompt and one-event response path now supports `Conveyor
breakdown`, `ur5e-1 breakdown`, `Machining breakdown during part processing`,
and `Part slippage`. Context selection follows recorded relationships and
custody; all eligible resources remain discoverable. Public prompt and retrieval
tests exclude evaluator records, reference recoveries, route hints, private
primitive sequences, and operator metadata. Operator-guided runs remain outside
the unassisted evaluation. Large selected safety sets list every requirement
with exact identifiers and provide the full formula and AP bindings through
the existing public retrieval references.

Resource-owned calculations now prepare `KMR`, `M1`, `M2`, and `Conveyor`
capability evidence against the same starting snapshot. Unordered
`successor_primitive_union`, `validation_contract_version=2`, and the 4,096
assignment bound remain unchanged. Private physical projection, CCA validation,
and nominal continuation remain separate obligations. Missing continuous base
motion, clearance, or controller evidence remains unresolved rather than being
converted from a route or a station label into an approved motion.

The Recovery UI retains the four exact scenario names, scenario-matched saved
captures, Offline Manual, expandable evidence, and original downloads. New
cases stop at outline review. Source capture, configuration, selected
specifications, and implementation provenance are recorded separately from the
LLM proposal. A failure checkpoint no longer acts as operator Stop.

Automated verification completed so far: 465 owner/offline/declared-program/
validated-outline tests, 276 prompt/retrieval/request/evaluation tests, and a
later 213 KMR/retrieval tests passed. These overlapping suites are not summed.
Poetry checks, Python compilation, evaluation CLI help, and Gazebo bootstrap
passed. Poetry retains its metadata deprecation warnings. A broader earlier
safety run was interrupted to release simulation resources after 857 passes;
it is not recorded as a completed suite.

Development capture evidence is retained in
`monitor/environment_runs/recovery/evaluation_development/`. The fifth
Conveyor development run reached its real paused failure checkpoint and
prepared recovery context without an LLM or recovery-execution call. Earlier
runs exposed a missing system-Python dependency boundary, a disabled Nav2
launch, and lost lifecycle-transition replies. The repaired readiness path
requires an observed map and active managed nodes; the lifecycle proxy checks
observed state after a lost reply without repeating physical goals. The fifth
run then exposed an unset optional KMR parameter during offline model archival;
the original observations remain retained unchanged. The native companion was
subsequently archived offline after preserving unset ROS parameters as typed
records and omitting only their inapplicable YAML overrides. The companion's
native model fingerprint matches the captured model.

New-attempt preparation now preserves all twelve resources' declared fields,
explicit null values, and capability catalogs from that capture. A detached
replay reported `outline_ready` with zero API, controller, observation, or native
calls and an unchanged observation SHA256. The completed `ur5e-1` pickup at the
Conveyor failure checkpoint remains recorded as completed; it is not relabelled
as a failed task. The recorded `gear_small` assembly is complete, leaving eleven
unfinished tasks for `KET4_Square_4mm`.

Further verification: 385 recovery/context/capture tests passed, 312 shared
prompt/request/evaluation/KMR tests passed, and 13 UI/attempt-version tests
passed. These suites overlap. The installed native preview suite passed 46
tests and 42 subtests in an isolated synthetic planning scene without Gazebo or
motion execution. Poetry, full Python compilation, and UI CLI help passed.
New attempts use verified native motion durations for UR projection clocks;
historical ordinal records retain their original interpretation and status.

KMR base travel still requires independently supported controller-motion
evidence. Saved routes, docking targets, and map samples alone do not establish
the timed closed-loop Nav2 motion or carried-part clearance. The development
record `evaluation_development/KMR_base_motion_evidence_gap_20261012.json`
distinguishes missing capture data from the unimplemented complete base
projection. Such results remain `NEEDS_CONTEXT` and cannot be counted as LLM
planning failures or complete recovery outlines.

The 24-attempt evaluation has not started. It requires eight separately
reserved Gazebo captures, each produced by the same unchanged implementation,
with actual normal/changed-condition facts and verified native companions.
Development captures cannot be promoted into this batch. The runner uses
`gpt-5.6-sol`, `low`, 80 turns, and no attempt deadline; all 24 results must be
retained, including unresolved evidence and API errors. Complete success
requires provider authorship, RA, physical evidence, CCA, and validated nominal
continuation. No recovery execution or user-observed UI acceptance is claimed.

The first reserved capture batch, `evaluation_captures/20261012_frozen_01`,
was stopped before any model attempt. Its Conveyor capture exposed an old
Part slippage observation contract requiring Nav2 to be disabled. The new
capture records KMR's source-bound command gate and sole output publisher,
retaining direct Nav2 activity as observed or unknown. It does not equate a
closed actuator gate with all Nav2 goals being idle. KMR base hold additionally
binds the physical Gazebo description to the same links and joints in the
planning description; the MoveIt description alone omits the required Gazebo
declarations. Offline hold checks retain both evidence obligations.

The changed Part slippage condition now binds its reach bound to KMR's
captured map extent, controller, complete planning base chain, and physical
base evidence. Planning proxy joint limits alone cannot establish mobile-base
reach. The bound assumes the captured map and model configuration remain
unchanged and grants no trajectory approval. The capture harness enables
Nav2 for all four capture cases; existing UI launch defaults are unchanged.

Further automated verification: 977 combined capture/safety/outline/Gazebo
Python tests passed, followed by eight real-validator map/base integration
tests and 33 capture tests. These suites overlap. The evaluation runner also
passed 30 tests covering complete eight-condition manifest verification,
manifest changes before API calls, and SIGTERM cancellation with owned-child
cleanup. Poetry and Python compilation passed. A new reserved batch is still
required after the final capture checks; the stopped batch remains retained.

The second reserved batch, `evaluation_captures/20261012_frozen_02`, also
stopped before any model attempt. Its Conveyor capture exposed a different
observation gap: the already assembled `gear_small` was attached by a native
fixed joint to static `assembly_board_v1::link`, but the old attachment record
did not establish the live joint or its endpoint transforms. Zero velocity and
missing contact samples cannot supply that evidence. `GETRECOVERYSTATE` now
records the observed joint, endpoints, relative transform, and attachment
revision. A detached calculation checks those observations before recording
conditional stationary geometry. This does not grant execution permission or
infer any future placement outcome.

The corresponding combined regression run passed 796 tests and exposed three
stale error-message expectations from this milestone's preservation of original
exception types. Those expectations were corrected; all three focused retry
tests passed. They are recorded as corrected expectations, not unrelated
failures. The native observation plugin rebuilt successfully. A new development
Conveyor capture is checking the complete observation path before another
reserved batch. No reserved model attempts or recovery execution have started.

### October 12: M1 breakdown machine job control

New simulation machine programs use `select_program`, `start_job`, `pause_job`,
`resume_job`, and `stop_job`; `machine_part` remains the machining task. Retained
job identities, active processing time, paused time, configuration, and completion
history survive pauses and unloading. Paused jobs can resume only with the same
part, machine, program, tool and workholding. Stopped jobs cannot resume. Program
selection cannot change tools or workholding. This is simulated active-time
accounting, not measured material removal or physical machining resumability.
Historical `dwell` programs retain their replay path.

The new machine-breakdown order includes `RGOCG4-50_Round_4mm` alongside the
existing square and gear. M2/ur5e-2 are enabled for nominal preparation. The
`during_processing` checkpoint waits for observed overlapping jobs and selects
a seeded nonterminal time; `during_processing_halfway` remains available.
`processing_observations` records failure-time job facts before cancellation;
shutdown records are separate. No recovery events or preferred sequence were
added to prompts. Both exact machine safety specifications remain selected.

RA uses the same pure owner calculations offline and live, with unordered
`successor_primitive_union`, contract version 2 and the 4,096-assignment bound.
Job controls have separate physical projections. Missing evidence remains
`NEEDS_CONTEXT`; an active peer job cannot silently freeze during another
positive-duration event. Current access/workholding evidence is required after
projected custody changes. Primitive composition and recovery execution remain
outside this milestone.

Automated results: 225 machine/setup/legacy tests passed; 227 overlapping
machine/safety tests passed; final focused runs passed 24 machine/Gazebo checks,
28 outline checks and 19 retrieval checks. Poetry, Python compilation, UI help,
and the 16-package Gazebo bootstrap passed. One unrelated robot-control test
stalled and was excluded from the 225-test run; two broader runs timed out and
are recorded separately, not counted as passed.

Live acceptance remains blocked. One isolated Gazebo capture failed before
machining at the ROS readiness query timeout. It produced no new observations,
made no LLM calls, left the preserved files unchanged, and stopped all owned
launch groups. No offline GPT-5.6 attempt was started, because the existing
empty-M2 capture does not represent the requested case. No recovery execution,
validated nominal continuation, or user-observed acceptance is claimed.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_job_control_capture_20261012/IMPLEMENTATION_RESULT.md`
and its `implementation_verification.json`, capture report and retained logs.
Next: resolve the observed readiness blocker, capture overlapping M1/M2 jobs,
then perform the single authorized `gpt-5.6-sol`/`low` offline attempt with the
80-turn limit and no ten-minute generation deadline. Do not alter requirements,
insert recovery events, or repeat trials to obtain a favorable outcome.

### October 12 follow-up: readiness diagnostics and observed M1 failure

Readiness commands now retain startup phase, launch identity, ROS domain,
elapsed time, timeout, exit status, stdout and stderr. The actual failure reason
reaches SystemBridge. Owned probe processes are terminated and reaped on timeout
or cancellation. Required service checks, the 10-second probe timeout, serialized
discovery, and stale-launch rejection remain in place.

Automated verification: 112 startup/capture/machine-job tests passed, followed by
33 overlapping capture tests after adding the diagnostic-path assertion. Poetry,
Python compilation, UI/capture CLI help and diff checks passed. No ROS2 source
changed in this repair. The original readiness failure did not recur and its
original underlying cause remains unconfirmed.

The first instrumented capture passed readiness but exhausted its 600-second
capture allowance during circular-part transfer to M2. A retained retry used the
identical setup and failure seed with a 2400-second capture allowance. It observed
the requested M1 `during_processing` failure while both machine jobs were running:
M1 square processing was at 24% and M2 circular processing at 12% of their configured
active simulation durations; neither process was complete. The original capture is
`recovery/4a81d7fc7ccd46d1a18b79740d16add7/observations.json`. All owned Gazebo and
agent processes stopped, and existing preserved-file hashes remained unchanged.

Offline acceptance remains blocked by evidence reconciliation. A later physical
export reported `KMR.current_pose` conflicting with owner state. The original saved
capture's local offline preparation found `M1.active_job=environment_21` and
`M2.active_job=environment_23` in prepared snapshots/environment valuations while
both owner snapshots recorded explicit `null`; `job_state` also disagrees. The
check made zero forbidden/live/LLM calls and left the capture hash unchanged.
No values were overwritten and no validator was bypassed.

API disclosure approval for one machine-breakdown GPT-5.6-sol/low attempt is now
recorded, but generation was not started because preparation is blocked. No
recovery proposals, accepted recovery events, RA/CCA recovery verdicts, validated
nominal continuation, recovery execution, or user-observed UI acceptance are
claimed. The next repair is consistent capture revisions and owner snapshots for
machine job records, plus retention/reconciliation of the exact KMR pose evidence.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_startup_repair_20261012_02/IMPLEMENTATION_RESULT.md`,
`implementation_verification.json`, `local_preparation.json`, and
`recorded_state_comparison.json`; both capture attempts retain their diagnostics.

### October 12 follow-up: machine/KMR snapshot repair and one offline attempt

Capture now preserves validated machine-worker job fields through physical
snapshot refresh. KMR preparation uses frozen measured observations and retains
earlier primitive endpoints as history, without rereading owner state during
preparation. Conflicts, changed revisions, and frame mismatches remain explicit.
Raw observations and preparation errors persist even when export fails. Readiness
checks saved-state consistency before generation.

The original `4a81d7fc7ccd46d1a18b79740d16add7/observations.json` remains unchanged
(SHA-256 `a6070287e3a90ae30ea5493b45118c88b3460667a4a907c9ffff0d4dad4682be`).
A separately derived capture records corrections and provenance. Local replay
reached `outline_ready` with all LLM and live interfaces forbidden. Failure-time
processing and later physical observations retain their different timestamps;
their compatibility is not assumed for physical validation.

Automated checks: 489 capture/offline/machine/resource tests and 511
validation/retrieval/UI tests passed, with overlapping counts. Poetry, Python
compilation, UI/capture CLI help, and diff checks passed. Sandboxed asyncio thread
cleanup hung in a minimal reproduction; the checks completed outside that sandbox.
An obsolete machine-scenario test was updated to expect M2 and ur5e-2 enabled.
A final 307-test subset passed after ensuring missing machine-worker records
remain unknown in the detached loader. The attempt's recorded M1/M2 job fields
were rechecked unchanged; no additional API attempt was made.

One unchanged-settings live recapture passed readiness but stopped before the M1
failure checkpoint: `/KMR/move_base` aborted because arm/payload clearance timed
out after 0/1 states. Owned Gazebo and agents stopped, with preserved-file hashes
unchanged. This additional clearance-service blocker was not expanded into new
KMR motion work.

Exactly one approved GPT-5.6-sol / low offline attempt then used the repaired
original capture, both selected machine safety specifications, shared factual
prompts, and dynamic retrieval. It made 9 provider requests/responses, 4 retrieval
turns, and 4 proposals for the first event: M2 `pause_job`, M1 `interrupt_job`, M2
`pause_job`, and KMR `move_to`. Accepted events: 0. The pause proposals retained
an active job contrary to the recorded pause successor; M1 was unavailable; KMR
`move_base` lacked an accepted parameter witness. Physical preview preparation
then stopped at missing `ur5e-4` attachment-link evidence. CCA was not reached and
nominal continuation was not established. Outcome: `NEEDS_CONTEXT` with the
original stage exception retained; no evidence was bypassed.

The attempt took 136.89 seconds including preparation and used 101,964 tokens
(99,086 prompt, 2,878 completion), with no provider errors or forbidden live calls.
Implementation and source hashes remained unchanged. No recovery execution or
primitive composition occurred. User-observed UI acceptance remains pending.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_observation_repair_20261012/IMPLEMENTATION_RESULT.md`
and `RESULT.json`, `machine_observation_repair_20261012_capture/report.json`, and
`machine_observation_repair_20261012_offline_01/report.json`. Remaining work is
separate: obtain compatible complete physical observations, diagnose the live
KMR clearance-service timeout, and repair the demonstrated KMR parameter-contract
gaps before claiming a fully validated machine-breakdown outline.

### October 12 follow-up: local offline validation repair without recovery hints

Completed the optional/nested primitive schema correction, KMR empty-custody
parameter preparation, precise public successor-conflict feedback, and structured
missing-scene evidence handling. New catalogs and descriptors are rebuilt; original
captures and historical results remain unchanged. The prompt implementation is
byte-identical to its pre-repair version. No API calls or Gazebo launch occurred.

Local replay of the four saved proposals now gives RA `FEASIBLE` for KMR `move_to`.
Both M2 `pause_job` proposals still reject the conflicting `active_job` value with
specific feedback, and unavailable M1 still rejects `interrupt_job`. A supported
paused-state regression fixture passed separately; it was not provider-authored.

Physical evidence remains unresolved: the recorded
`KET4_Square_4mm/link/collision` identity is absent from the saved MoveIt object list,
which instead contains `KET4_Square_4mm`. Its scene acknowledgement is false, so no
identifier substitution or clearance evidence is invented. The existing stale
`ur5e-4` observation and processing/physical timing distinction remain recorded.
Unresolved preparation now stops with source references, without an unproductive
validation retry. No outline append, CCA approval, or nominal advancement occurred.

Automated checks: 1,071 focused tests and 438 outline physical-evidence tests
passed; the broader RA suite had 198 passes and 12 failures, all reproduced with
the pre-repair sources (legacy AP descriptors and KMR prepared-execution-provider
tests). Poetry, compilation, and UI command-line help passed.
Source capture, safety-file, and prompt hashes were preserved. User-observed UI
acceptance, a fresh capture, and another model attempt remain separate work. The
live KMR clearance-service timeout was not changed.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_validation_repair_20261012/IMPLEMENTATION_RESULT.md`,
`local_replay_after.json`, and `verification.json`.

### October 12 follow-up: one fresh offline GPT-5.6 retest

The requested `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture, both selected machine safety specifications, shared prompts, and dynamic
retrieval. All live equipment and later-stage interfaces remained blocked. There
were 17 successful provider calls: one grounding turn, three retrieval turns, and
13 proposals for the first recovery event. Accepted events: zero. Physical
projection, CCA, and nominal continuation were not reached in this attempt.

The test exposed a separate outline response-schema defect. M2's `active_job`
declaration permits `["string", "null"]`, but its incomplete observed domain
contains only `"environment_23"`. `_outline_state_field_schema` uses that domain
to generate `["string"]`, excluding the `null` value required by the declared
paused state. Local replay verified that this value passes RA while the provider
schema rejects it. The local fixture was not sent to the model or accepted as an
LLM event. No implementation or prompt changes were made during this retest.

The owned attempt was stopped after confirming the contradictory constraints,
with all 17 started provider calls completed. Duration: 230.54 seconds; token
usage: 187,128 prompt and 7,020 completion, totaling 194,148. Sources and
implementation fingerprints remained unchanged. The raw cancellation snapshot
retains `generating_outline`; the separate test result records the stopped
process and terminal outcome. No second attempt or recovery execution occurred.

Next repair: respect declared nullable types when constructing public outline
response schemas. Saved physical-scene evidence remains a separate unresolved
requirement; this schema repair alone cannot establish CCA acceptance.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_validation_retest_20261012220045Z/TEST_RESULT.md`,
`TEST_RESULT.json`, and `schema_reproduction.json`.

### October 12 follow-up: declared outline response types repaired locally

The existing outline field-schema helper now preserves an explicit declared
type before considering observed domain values. M2's nullable active_job remains
nullable in both resource-only and part-bound response schemas. Legacy fallback
is unchanged; malformed explicit types use the existing preparation-error path.
No RA, physical, or CCA validation logic or public interface was added.

Local replay regenerated the actual response schema from the saved machine
session. The original pause_job proposal remains rejected by RA. A separately
labelled supported paused-state fixture passes the corrected schema and RA; it
was not sent to a provider, included in prompt content, or appended as an LLM
event. Unsupported M1 operations remain rejected. Saved KMR move_to support
remains FEASIBLE, with its physical evidence unresolved.

All 900 focused context, request-record, validated-outline, workflow, and offline
tests passed. Poetry checks and Python compilation passed. Prompt instructions,
source observations, and historical request/response records were preserved.
The changed source is covered by the existing implementation fingerprint.
The earlier 12 unrelated broad RA-suite failures remain separately documented
and were not rerun in this scope.

No API calls, Gazebo launch, recovery execution, or live UI acceptance occurred.
The remaining saved physical-scene evidence gap is unchanged; complete CCA
acceptance and nominal continuation are not established.

Evidence: `cais_spade_llm/monitor/environment_runs/outline_schema_repair_20261012221658Z/RESULT.md`,
`verification.json`, `schema_replay.json`, and `response_schema_not_sent.json`.

### October 12 follow-up: real offline retest after the nullable-type repair

One fresh `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture and both machine safety specifications. The exact provider requests
now permit `active_job=null`. After two public retrieval turns, the model first
proposed M2 `pause_job` retaining `active_job="environment_23"`; RA rejected that
conflict. Its next proposal independently changed that field to null and passed
RA as `FEASIBLE`. No proposal was rewritten or inserted by the runtime.

Physical projection and CCA remained unresolved because failure-time processing
records at simulation times 84.0/84.2 lack recorded compatibility with physical
measurements at 85.65/85.8665. Zero events were appended and nominal continuation
was not established. This is missing evidence, not a proved safety violation.

The subsequent offline evidence calculation also exposed a preview problem.
The standalone test initially lacked the installed ROS message-library environment
and failed importing `moveit_msgs`. Sourcing ROS fixed that import locally without
starting Gazebo. Replaying the unchanged proposal with its saved prepared inputs,
with API and live calls forbidden, then failed an SRDF named-pose lookup for the
M2 job-control proposal. That fallback needs the appropriate machine-owner
projection path. It cannot substitute robot motion evidence for job control.

The single API attempt completed five successful provider calls in 69.28 seconds:
one grounding, two retrievals, and two proposals for the first event. Usage was
42,537 prompt and 1,186 completion tokens (43,723 total). Zero provider errors
and zero forbidden live calls were recorded. Original/derived captures, safety
files, and frozen implementation hashes remained unchanged. No additional API
attempt, primitive composition, recovery execution, or user-observed UI acceptance
occurred. The prior 900-test schema-repair result remains recorded separately;
this retest changed no runtime source.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_schema_retest_20261012T222911Z/TEST_RESULT.md`,
`TEST_RESULT.json`, `attempt_01/report.json`, and `local_prepared_evidence_replay.json`.

### October 12 follow-up: resource-owned outline evidence routing repaired

Offline evidence preparation now uses the existing resource-owned projection
for machine and other nonrobot proposals. Robot and KMR paths are selected from
the recorded resource type; unsupported owners return missing evidence. Machine
job control no longer requires a native robot archive or falls through to SRDF
named-pose lookup. Source-hash checks and unresolved checkpoint compatibility
remain enforced. The change adds no prompt instructions, primitives, safety
specifications, recovery events, or execution permission.

Exact local replay preserves the previous M2 proposals: Turn 4 remains RA
INFEASIBLE for its active_job conflict; Turn 5 remains RA FEASIBLE. Turn 5 now
returns the recorded processing/physical checkpoint-compatibility gap directly,
without a ROS import or robot named-pose exception. No event is appended, and
CCA approval and nominal continuation remain unresolved. Failure-time processing
and later physical observations cannot be combined without supporting evidence.

All 1,108 focused tests passed: 675 machine/environment/workflow/physical-evidence
tests and 433 offline/validated-outline/KMR tests. The tests include live and
detached M1/M2 owner calculations, Conveyor and Buffer For Machined parts,
unsupported owners, stale source hashes, retained peer geometry, and forbidden
robot preparation for nonrobot events. Poetry checks and Python compilation
passed. The initial sandboxed mocked stage-mode test stalled; the same test and
both complete suites passed outside the sandbox. That interrupted diagnostic run
is recorded separately. Earlier unrelated broad-suite failures were not rerun.

All 44 protected hashes remained unchanged, including captures, the prompt
builder, both safety files, and prior request/response records. No API call,
Gazebo launch, recovery execution, or user-observed UI acceptance occurred.

Evidence: `cais_spade_llm/monitor/environment_runs/owner_evidence_repair_20261012T224635Z/RESULT.md`,
`verification.json`, `saved_proposal_replay.json`, and `source_changes.patch`.

### October 12 follow-up: one offline API retest after owner-evidence routing repair

One fresh `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture, both machine safety specifications, shared prompts, dynamic retrieval,
Offline Manual, 80 turns, and no overall generation deadline. The runner was
reused unchanged; no runtime, model, prompt, or specification edits were made.

All five provider calls succeeded. The model grounded the capture, made two
public retrieval turns, and proposed two versions of the first M2 pause_job
event. Turn 4 retained active_job="environment_23" with resource_state="loaded"
and job_state="paused"; RA rejected the full end state. This attempt's feedback
identified resource_state="completed" with the other fields fixed. Turn 5 made
that change while retaining the active job. RA derived start_job support from
machine_part, but the owner rejected its capability checks because a machining
job was already active. RA returned NEEDS_CONTEXT, not FEASIBLE. Its private
parameter_witnesses retrieval returned no saved evidence.

Subsequent physical evidence preparation returned the recorded checkpoint
compatibility gap directly, before robot/native preparation. The earlier ROS
import/SRDF exceptions did not recur. Successful machine projection and CCA were
not reached; zero events were appended and nominal continuation was not
established. This result differs from the earlier local replay that passed RA
with active_job=null. Two proposal revisions are not two accepted events.

The attempt took 65.43 seconds and used 43,696 prompt plus 1,264 completion tokens
(44,960 total). All 523 frozen files and 44 protected historical files passed
hash verification. Zero forbidden live or execution calls occurred. Exact
requests/responses, retrievals, RA, and physical-preparation records are saved.
The preceding 1,108-test result remains separately recorded; no new regression
suite or user-observed Gazebo/UI acceptance is claimed for this retest.

No additional API attempt, capture repair, Gazebo launch, primitive composition,
or recovery execution occurred. Further repairs remain separate work.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_owner_evidence_retest_20261012T235206Z/TEST_RESULT.md`,
`TEST_RESULT.json`, `attempt_01/report.json`, and `attempt_01/session.json`.

### October 12 follow-up: repair derivation and shorten outline prompts (local replay)

**Implemented and locally verified.** Outline derivation now compares intended
conditions with the recorded or accepted projected start. Only declarations
that explicitly support a requested resource change contribute primitives;
preserving a requested value cannot add unrelated support. Owners pass their
existing snapshots into preparation. Contributors, bindings, preserved
conditions, and additional effects remain recorded as obligations. Unordered
`successor_primitive_union`, independent capability checks,
`validation_contract_version=2`, and separate physical/CCA validation remain.
The internal parameter-trial limit accepts a positive configured value; its
default remains 4096. No extra validation layer or movement validator was added.

Local replay of the saved M2 evidence verifies that the minimal
`job_state="paused"` fixture derives only `pause_job` and passes RA. The original
provider proposal with `active_job="environment_23"` still contradicts the
relevant declaration. Its feedback is independent of JSON key order and does
not suggest `resource_state="completed"`. Supported paused-state fixtures remain
explicitly separate from provider responses and prompt content. The nominal
checker still validates each exact recorded unfinished task; unchanged public
values alone do not establish task completion.

Initial outline message size fell from **20,202 to 11,127 characters (44.9%)**.
The provider `response_format`, including its unchanged schema, remains
**16,653 characters**. One resource index and retrieval section replace
duplicate listings. Initial processing facts and public field meanings are
compact; full declarations, job records, programs, and safety bindings remain
available through existing public references. Both exact machine-breakdown
requirements remain included. The saved four context requests were replayed
successfully without live calls; their prepared message is 17,079 characters.
All new previews are marked not sent. Full observations and audit files remain
available, and no action, resource preference, sequence, or event-count target
was added to the prompt.

Automated verification: **1,979 passed; 12 pre-existing failures**, reproduced
against saved pre-edit source. Those failures are five retired AP descriptor
cases and seven KMR execution-provider cases. `poetry check` and
`poetry run python -m compileall -q cais_spade_llm ros2` passed. The guarded local
replay made zero API/live-interface calls and verified all 53 protected hashes.
Original/derived observations and historical requests/results were preserved.

**Remaining evidence gap:** failure-time processing and later physical
observations still lack recorded checkpoint compatibility. RA success does not
establish a physical projection, CCA approval, or nominal continuation. No new
API attempt, Gazebo capture, primitive composition, or recovery execution was
performed. User-observed UI/Gazebo acceptance remains separate.

Evidence: `cais_spade_llm/monitor/environment_runs/derivation_prompt_repair_20261013T002221Z/RESULT.md`,
`local_replay.json`, `verification.json`, `initial_outline_after_not_sent.txt`,
`retrieval_replay.json`, `source_changes.patch`, and the baseline/final test logs.

### October 12 follow-up: one offline API retest after derivation and prompt repair

One fresh `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture, both exact machine safety specifications, Offline Manual, 80 turns,
and no overall generation deadline. The runner was reused unchanged with an
independent session; no runtime implementation or global settings were changed.

All four provider calls succeeded. Turn 1 grounded the facts; Turn 2 retrieved
M2 and M1 `static_capabilities`. Turn 3 proposed M2 `pause_job` while retaining
active_job="environment_23". RA returned the precise active_job contradiction,
without the former unrelated completed-state correction. Turn 4 proposed
active_job=null with job_state="paused" and resource_state="loaded"; **RA passed
and derived only pause_job support**. Part conditions remain deferred obligations.

Physical projection returned the known failure-time processing/later physical
observation compatibility gap. CCA was invoked and returned unresolved with no
completed safety checks; this is not a demonstrated safety violation. Owner
evidence preparation reported the same gap without robot named-pose fallback.
Zero events were appended, the symbolic state and monitor history were
unchanged, and nominal continuation was not established. The two proposals are
revisions of one event, not two accepted recovery events.

The actual initial outline message was 11,127 characters; its response-format
schema remained 16,653 characters. The complete provider payloads before the
first proposal contained neither pause_job nor start_job. No operator guidance,
preferred sequence, or event-count target was added. Dynamic retrieval served
both requests from saved facts without errors.

The attempt took 55.34 seconds and used 24,355 prompt plus 1,020 completion
tokens (25,375 total). All 523 frozen files and 53 protected historical files
passed hash verification; zero forbidden live or execution calls occurred.
The preceding repair's regression results remain separately recorded and were
not rerun here. No additional API attempt, fresh capture, primitive composition,
recovery execution, or user-observed UI/Gazebo acceptance occurred.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_derivation_retest_20261013T005942Z/TEST_RESULT.md`,
`TEST_RESULT.json`, `attempt_01/report.json`, and `attempt_01/session.json`.

### October 12 follow-up: frozen machine checkpoint implementation; live trial blocked

Implemented machine checkpoint collection before worker cancellation for
`during_processing`. New nominal dispatch stops first; the owned Gazebo world
is frozen without issuing `pause_job`. Read-only machine requests account for
processing through the observed checkpoint and preserve completion races.
GETRECOVERYSTATE can service paused-world observations under the physics lock.
Raw native observations, machine replies, physical evidence and the planning
scene are retained before preparation; shutdown remains separate.

`processing_capture_compatibility` now requires recorded checkpoint identity,
revision, source and fact bindings. Equal timestamps alone cannot certify a
capture. Preparation and the capture harness reuse the frozen checkpoint and
report incomplete evidence before generation. Stop retains the observation
connection until it can unfreeze the owned world. Historical records, both
machine safety specifications and recovery prompts remain unchanged.

Automated results: **1,813 passed; 34 skipped** in the focused suites;
**36 passed; 1 skipped** in the final cleanup checks; **384 passed** in the
final capture/offline checks after binding unfinished obligations. These suites
overlap. Poetry checks, Python compilation, UI/capture CLI help,
`make bootstrap-gazebo` and the native preview suite passed. Local fixtures
do not establish a complete accepted sequence for the live case. Existing
checks still require supported processing progress across event intervals;
unaccounted running jobs remain unresolved rather than silently frozen.

Observed by the isolated harness: the first startup failed before jobs ran
because its invocation lacked the sourced ROS environment. The corrected
invocation retained the same scenario setup, durations and seed `4000168541`.
It reached nominal work, but KMR `pick_approach` for `KET4_Square_4mm` failed
with `KMR fresh joint feedback timed out` at 21:34:18 EDT. Logs also retain
read-only ROS query retries and lost Nav2 map-server heartbeats. The underlying
timing/transport cause is not yet established. M1's failure and the new frozen
checkpoint were never reached. All owned simulation processes stopped.

**Milestone remains incomplete.** No consistent new capture was available,
so the one offline `gpt-5.6-sol` / `low` attempt was not started: zero API
calls/tokens and no new proposed or accepted events. RA, physical projection,
CCA and nominal continuation were not evaluated for a new generated outline.
The historical capture was not substituted and no evidence checks were
bypassed. KMR fresh-feedback diagnosis and a successful checkpoint remain the
prerequisites for the complete offline attempt. No primitive composition,
recovery execution or user-observed UI acceptance occurred. All 61 protected
file hashes and both safety specification hashes are unchanged.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_checkpoint_repair_20261013T011337Z/RESULT.md`,
`verification.json`, `preservation.json`, `capture_sourced/report.json`,
`capture_sourced.log`, `capture_sourced/gazebo.log` and the test/build logs.

### October 12 follow-up: requested offline test from the previous capture

The user requested reuse of the recorded machine-breakdown observations instead
of another Gazebo trial. Offline preparation now permits proposal generation
when physical checkpoint compatibility is missing, displays that limitation,
and retains the compatibility result for physical/CCA validation. Conflicting
resource facts still block preparation; live capture checks remain unchanged.
This supersedes the preceding requirement for a new capture before any offline
generation, without changing the evidence required to accept an event.

One real `gpt-5.6-sol` / `low` attempt used the unchanged
`derived_4a81d7fc7ccd46d1a18b79740d16add7_20261012` capture, both selected machine
safety specifications, Offline Manual, 80 turns and no overall deadline. Shared
prompts and public retrieval were unchanged. Seven requests succeeded: one
grounding turn, four retrieval turns, and two revisions of M2 `pause_job`.
The first retained a conflicting `active_job="environment_23"`; the second
requested only `active_job=null` and `job_state="paused"` and passed RA.

Physical projection remained `NEEDS_CONTEXT` for the saved processing/physical
checkpoint compatibility gap. CCA returned `unresolved`; no event was appended,
projected state and monitor history were unchanged, and nominal continuation
was not established. The attempt stopped on that evidence finding, not an
event-count limit. Elapsed time was 70.88 s; usage was 48,531 prompt and 1,464
completion tokens (49,995 total). No API errors or forbidden live-interface
calls occurred. No Gazebo launch, new capture, primitive composition, or
recovery execution occurred.

Automated checks: **958 passed**, Poetry checks passed with existing metadata
warnings, and Python compilation passed. Original and derived captures, safety
files and prompt sources passed hash verification; implementation/model/safety
fingerprints stayed fixed during generation. This is an automated offline API
result; user-observed UI acceptance and complete outline acceptance remain
separate and pending.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_saved_machine_retest_20261013T020100Z/RESULT.md`,
`attempt_01/report.json`, `attempt_01/session.json`, `preservation.json`, and
`verification.json`. Exact turns are saved under
`recovery/offline_77022cb8e9dd48b7bfa6bd208614baa0/generation_0001/`.

### October 12: required evidence checks and instantaneous machine controls

Local replay of the unchanged saved M2 `pause_job` now reaches its owner
calculation. Preparation recovers the omitted `ur5e-4` collision bounds from
the saved native `dual_robot` record, without inventing a tool transform or
motion coverage. Machine fields remain sourced from job records. Capture-wide
compatibility no longer vetoes unrelated proposals: checks use the event,
selected AP dependencies and projected duration. Resource discovery retains
declarations with missing geometry; CCA still requires relevant participant
evidence. Private retrieval preserves projected facts and terminates when no
new evidence is returned.

CCA owner ledgers now represent declared instantaneous simulation job controls
with separate, fingerprinted before/after checkpoints. Complete local fixtures
pass both unchanged machine safety specifications; missing peer geometry still
returns `NEEDS_CONTEXT`. No positive motion duration is invented.

**Saved-capture result:** RA is `FEASIBLE`, but physical/CCA validation remains
unresolved and zero events are accepted. M2's job record is at 84.0 simulation
seconds; physical measurements are at 85.65–85.8665. The owner model bounds
processing to at most 2.4665 of 5.0 seconds, excluding completion, but cannot
establish `active_job` and `job_state` alongside that physical checkpoint.
The later owner export copies the earlier record. Saved measurements and
shutdown records were not rewritten or presented as simultaneous. Compatible
job/control history or a synchronized capture is still required; nominal
continuation is not established. No new API request was made against this
known evidence gap, and no Gazebo or recovery execution was started.

Automated results: **1,620 focused tests passed**, followed by **102 final
targeted checks**; Poetry checks and Python compilation passed. Protected
capture/specification/prompt hashes remained unchanged. The complete saved
response replay retained zero accepted events and unchanged state/history.
User-observed acceptance and full-outline acceptance remain pending.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_nominal_repair_20261013T022658Z/RESULT.md`,
`local_replay/transaction_turn.json`, `verification.json`, `preservation.json`
and `implementation_changes.patch`.

### October 13: synchronized capture and successive offline repair attempts

The authorized isolated harness reached M1's fault while M2 was still
processing. Capture 03 retained native and machine observations at frozen
simulation time 84.8675, instance `11407416760602`. Export failed after a pure
local calculation exceeded the wall-age check; raw checkpoint evidence was
already saved. A separate derived capture restored omitted ur5e-2 fields from
those frozen measurements and matching recorded models. Original observations
and shutdown records remain separate. KMR was moving at the checkpoint;
freezing simulation does not establish a stationary controller. All owned
Gazebo processes stopped.

Repaired nominal expanded-task lookup, saved public facts and exact resource
identities, declared robot approach preparation, optional callable defaults,
recorded detection provenance, frame/TCP evidence recovery, persistent factual
rejection feedback, and public M2 mutex binding retrieval. The shared recovery
instructions contain no suggested sequence, preferred resource or event-count
target. RA retains unordered `successor_primitive_union`; physical projection
and CCA remain separate. Both machine safety files are unchanged.

Seven independent `gpt-5.6-sol` / `low` development attempts used Standard
service (`default`), Offline Manual, 80 turns and no overall generation
deadline. They used successive repaired implementations, so these are not a
frozen comparative evaluation. All seven accepted one M2 job-control event;
**none established nominal continuation**. Total reported API usage was
957,517 tokens. Exact proposals, retrievals, validation records, timings and
per-attempt usage are retained. Source and implementation fingerprints stayed
fixed within every attempt; no forbidden live or execution calls occurred.

In attempt 07, the exact turn-5 M2 `pause_job` proposal passed RA, physical
projection and CCA. Its next movement proposals did not pass all stages. The
last KMR proposal passed RA but scene preparation could not bind the recorded
`KET4_Square_4mm/link/collision` identity to the captured model-name collision
object. Local replay reproduced this result with API and live interfaces
forbidden and the source hash unchanged. Separately, the existing KMR owner
projection explicitly lacks continuous `move_base` evidence; repairing an
object binding alone cannot fill that gap. Recorded robot pickup targets also
exceeded their declared `gripper_reach` limits. Those limits and measurements
were not changed to obtain acceptance. No new KMR movement validator was added.

Automated checks (overlapping suites): 1,323 recovery tests passed; 921
capture/offline/outline tests passed after physical-field restoration; 648
RA/projection tests passed with 14 deselected; 278 context-retrieval tests
passed after mutex binding repair. Installed native preview checks: 46 passed.
Gazebo bootstrap built 16 packages. Poetry checks, Python compilation, UI CLI
help and the final scoped whitespace check passed. A broader run recorded
904 passed and 12 failures in legacy AP fixtures and KMR prepared-execution
mocks. These are reported separately; no restored baseline comparison was
performed. The older independent outline-audit runner also requires metadata
not emitted by this runner, so no audit pass is claimed.

**Milestone remains incomplete.** Remaining scene-binding and movement-evidence
gaps must be resolved before another unchanged API attempt is useful.
Automated capture and generation results do not establish user-observed UI
acceptance, a complete CCA-approved outline or assembly completion. Primitive
composition and recovery execution were not started.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_synchronized_capture_20261013T040945Z/RESULT.md`,
`RESULT.json`, `preservation_final.json`, `attempt_07/report.json`,
`local_replay_attempt_07/report.json`, and `verification/`.

### October 13: KMR saved-evidence preparation repairs

Local replay of the unchanged attempt-07 provider proposals now passes the
released-part scene binding and missing coordinate preparation. Captured KMR
`release_part` acknowledgements bind the installed model-name collision IDs to
their exact SDF/native declarations, retaining both identities. The requested
`move_to_location` destination supplies its declared docking coordinates without
selecting a destination, route, or recovery event. Captured empty/carrying program
variants are respected; new physical preview records use version 2.

For future captures, an unsuccessful KMR base-hold check retains the raw
controller observation and physical fields already collected, together with the
original error. Moving or incomplete controller evidence remains unresolved.
Historical observations and both machine-breakdown specifications are unchanged.

Automated verification: **432 focused tests passed**, Poetry check passed with
existing metadata deprecation warnings, and Python compilation passed. Local
replay made zero API or forbidden live-interface calls. M2 `pause_job` still
passes RA, physical projection and CCA. KMR movement passes RA but reaches the
explicit missing continuous `move_base` projection. The saved KMR was moving;
stationary geometry cannot be assumed. **Nominal continuation remains incomplete.**

No Gazebo launch, primitive composition or recovery execution occurred. No new
API attempt was spent on this remaining deterministic implementation gap.
User-observed UI acceptance remains separate.

Evidence: `cais_spade_llm/monitor/environment_runs/kmr_offline_goal_20261013T120213Z/RESULT.md`,
`RESULT.json`, `local_replay_parameters/report.json`, and `tests_owner_evidence.log`.
