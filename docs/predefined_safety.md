# Predefined safety specifications

The journal framework assumes that AP meanings, LTLf formulas, and requirement
scopes are supplied and reviewed. Recovery behavior may be proposed by an LLM;
the selected safety definitions are fixed inputs. Deterministic compilation and
complete grounding do not prove that the reviewed hazard catalog is complete.

The selectable [document](../cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt)
is JSON stored in a `.txt` file for existing Safety-page discovery. Select it and
use **Compile DFA** to prepare its own preview. Review and activation remain
explicit; the existing selection, approvals, and narrower historical previews
are preserved. No safety-generation API call is needed.

| Specification | Supplied LTLf | Concrete bindings |
| --- | --- | --- |
| At most one resource may occupy the assembly_board-v1 destination area at any time. | `G !(ap001 & ap002)` | `ap_state resource slot 1` and `ap_state resource slot 2`: every unordered pair from `build_environment_models(scene)`, with `region: assembly_board-v1`. Current 12-resource scene gives 66 instances. |
| KET4_Square_4mm may enter assembly_board-v1 only after assembly of gear_small is completed. | `(!ap001 U (ap002 & !ap001)) \| G !ap001` | `ap001`: `part: KET4_Square_4mm`, `region: assembly_board-v1`; `ap002`: `part: gear_small`, `process: assembly`, `target: Gear_Plate/Gear_Shaft_1`. |

AP definitions use the context-free Product-Process-Resource schema. For example:

~~~json
{"kind":"ap_state","product":"*","process":"*","resource":"ur5e-1","state":{"symbol":"any","arguments":{"region":"assembly_board-v1"}}}
~~~

This is ap_state(*, *, ur5e-1, any@assembly_board-v1). The region qualifier is
independent of the current discrete state and event name. Resource wildcard
expansion covers the full configured scene and every unordered distinct pair.
Source templates use two internal resource slots; grounded rules contain exact
resource identities. Every label remains local to its rule_id.

The other predicates use typed conditions part_region_entry(region) and
processCompleted(target) with their product/process bindings. Receiving-region,
inventory, process-result, task-event and resource-state requirements use the
same schema. A canonical JSON key is derived from each definition, never parsed
as a slash-delimited path. No AP context field is accepted. Catalog/document
version 2 and ap_schema_version: 2 invalidate old saved AP artifacts; recompile
from the selected source before admission.

The target-completion AP requires an explicitly complete checkpoint ledger and
the exact assembly record. Its part binding is separate from the entry AP's part.
The existing `process_result_completed` AP is unchanged. The precedence formula
permits a no-entry trace, remembers completion, and rejects first simultaneous
completion and entry. A part already inside at the initial checkpoint creates no
fabricated entry and certifies no missing earlier history.

`SafetyLogic` compiles the exact formulas with the strict LTLfParser/MONA path.
It skips LLM parsing, AP generation, interpretation generation, renumbering, and
formula rewriting. Artifacts retain the full document, source/semantics hashes,
and configured product-geometry hash. Corrupt or changed definitions, bindings,
or DFA artifacts fail closed. The bundle's native-only plan checker cannot certify
these physical requirements; complete physical grounding is still required.

CCA loads these definitions before the legacy generation fallback. It binds all
selected scopes from its own source into the observation/composition inputs.
Physical rules stay on the grounded clock. Existing native monitors retain their
task start/completion clock; both monitor sets must share an accepting continuation.
Recovery plan registration binds behavior to fixed rules instead of generating
replacement safety rules. Outline projections alone cannot certify them.

The CCA configures physical evidence on the existing `RecoveryCompositionAdmission`
in `recovery_admission_runtime.py`, using the existing resource owners and shared
admission lock. `cca.live_safety_runtime` and `runtime.live_safety_runtime` reference
that same coordinator in `cca.recovery_composition_admissions`; they do not install
another admission authority. `EnvironmentAdmission` retains nominal composition,
grants, goals and acknowledgement history. Nominal and recovery
requests both require an exact owner-prepared primitive composition. Its parameters,
controller/configuration revision, initial geometry and attachment state, and every
trajectory point are bound to the grant. A changed or consumed command cannot be
dispatched under that grant. KMR retains its prepared motion inside its worker;
robot execution reconstructs the admitted trajectory without replanning or retiming.

The shared command ledger records pending authorization, dispatch, actual controller
goal identity, and completion or failure. Selected mutex regions are reserved
atomically for the entire composition. Physically occupied regions and previously
authorized incoming motion block conflicting entrants. A sole occupant can receive
a safe exit grant. Completion alone does not release a reservation: fresh observations
must establish that motion ended and the region is clear. Timeouts retain ownership
and exclusion; they do not assert that cancellation physically stopped the robot.

`LivePhysicalMonitor`, kept with the existing CCA safety checker, predicts from
detached copies of retained monitor states.
Authenticated owner execution certificates advance its conservative state set once;
generating another recovery does not recreate history. Every selected physical
requirement participates, including precedence. Outline validation can be explicitly
deferred for primitive generation, with `is_safe: false` and
`execution_authorized: false`; this is never an execution grant.

The finite composition checker includes the complete modeled scene when independence
cannot be established. The current live provider also uses this conservative fallback
and serializes physical programs. It does not yet establish a smaller dependency
closure for concurrent independent live work. Offline controlled fixtures test the
local/full composition agreement and joint nominal/recovery outcomes.

## Live acceptance status

The installed adapters do not establish a missing physical model. The current scene's
owner contracts explicitly declare `future_execution_tracking: not_established`.
An idle reading or an ideal planned path cannot replace bounded future motion,
stationary hold, and failure/stopping evidence. Live admission therefore returns
unverified for these contracts. Do not change that flag alone to obtain permission:
a trusted owner must supply a validated containment certificate tied to the exact
controller, geometry, attachments, and stopping behavior.

`RecoveryJointTrajectoryController` now exposes `~/recovery_motion_contract`
using `SetRecoveryMotionContract`. `prepare` checks the exact trajectory and
controller snapshot without dispatch or reservation. `arm` binds a reservation,
physical binding fingerprint, trajectory and action goal identity; competing
action/topic commands are rejected while ownership is retained. `stop` retains a
fault hold, and `release` requires fresh observed stationary completion. Controller
positions, velocities, update periods and tracking errors are diagnostic evidence.
They are not physical containment bounds.

Deactivation changes the controller instance identity and clears its observations
while retaining an existing reservation. Ownership operations check both
simulation time and the steady-clock observation age; a paused clock cannot keep
old evidence available for `arm` or `release`. Read-only `recovery_state` uses the
native update identity and steady-clock age independently of asynchronous `/clock`
delivery. The two-second availability cutoff is not a certified observation,
reaction, or stopping bound.

CCA requests this evidence through `PreparedRobotEvidence` after native planning
and before grounding or granting the motion. The read-only preparation provider
remains `gazebo_idle_continuous_v1`. The installed position controller reports
`physical_execution_verified=false` because Gazebo physics evolves between its
updates. Neither successful command ownership nor a supplied verification flag
can promote this report to a physical admission certificate. No inferred error
margin is installed in `joint_position_error`.

`scripts/check_live_recovery_motion.py` records a supplied single `move_cartesian`
from `GAZEBO_MOTION_SAFE`, both predefined specifications, raw observations, CCA
decisions and owner evidence in a new output directory. It dispatches only an exact
committed CCA grant. Native stationary ownership diagnostics are separate from CCA
acceptance. A `NEEDS_CONTEXT` result remains incomplete live motion acceptance.

Nominal expansion must also be lossless. Configured symbolic functions are expanded
only through supported owner program operations. Custody changes, mating/placement
corrections, and other operations without complete physical owner models remain
unverified. The continuous observation layer supports certified custody effects;
that does not itself provide a live controller certificate.

KMR arm preparation does not certify mobile-base motion. KMR base movement remains
unverified. Recovery admission with additional native temporal rules also remains
unverified until their commit and acknowledgement handling can participate in the
same transaction as physical admission; a successful preview alone cannot commit
that conjunction. These limitations are enforced in the runtime, not merely warnings.

The requested Gazebo demonstration of permitted nominal/recovery execution is not
established by the unit tests or offline traces. It remains an acceptance requirement
after those owner contracts are supplied and verified. Historical run evidence is
retained and is not relabeled as version-2 live evidence.

Optional `product_effect_evidence` records exact task/resource identities, declared
product effects, and completion timestamps. Those times join the joint trace.
Supported acknowledged assembly effects are resolved from configured `place_insert`
declarations; motion, release, and an event name alone create no completion record.
Composition may project a declared effect at a branch completion. Replay requires
a corresponding validated acknowledgement and exact observed effect before actual
history advances. Predicted metadata is compared on a detached copy only; actual
history retains acknowledged provenance. Unchanged prior acknowledged checkpoint
records remain intact. Missing or mismatched acknowledgements block further grants. After this optional
product-effect clock has been used, subsequent horizons must retain a complete
`product_effect_evidence` ledger, even when its `updates` is empty; dropping it
changes the clock identity and invalidates continuation.

The soundness claim is conditional on complete, correct supplied scene geometry,
fixed-orientation piecewise-linear trajectories, durations, stationary coverage,
custody, declared task effects, and observed history. The finite branch model must
include all running work and permitted starts. Under those assumptions, every
consumed valuation is grounded in that trace and CCA grants only starts inside its
nonblocking region. This does not establish arbitrary predicate discovery, physical
feasibility, a complete hazard model, or live execution safety.

The composition restriction on `X`, **20,000 states / 2 seconds** defaults,
`SystemBridge`, PA/RA/CCA authority, original fixtures, pending M1 delivery tasks,
and the documented legacy `receiving_region_entry` mismatch are retained.
Live provider installation preserves the public `SystemBridge` interface; `ui/bridge.py` remains unchanged.
