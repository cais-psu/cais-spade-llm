# Journal Paper 2: Recovery Framework

## Compact M1/M2 and KMR delivery (2026-09-21)

The new order is
[`assembly_board-v1-kmr-storage-m1.json`](../../cais_spade_llm/specification/products/orders/assembly_board-v1-kmr-storage-m1.json).
It references the existing `assembly_board-v1` product and geometry, selects only
`KET4_Square_4mm`, and uses `quantity: 1`. Its `completion_conditions` are a
conjunction of exact descriptor fields: M1 loaded with that part, its Storage
inventory false, and KMR idle, empty, and at M1. Orders without these conditions
retain their assembly goals.

To use it, open **projects → recovery-framework → setup**, select this Product
Order, retain **Simulation**, all 12 permitted resources, and no failure, then
click **Save setup**. Open **run** and use the existing **Start System**. Startup
prepares the scene and controllers, snapshots the saved configuration, and
rejects changed configuration or inconsistent observed inventory. **Stop System**
cancels preparation and execution while retaining acknowledged custody and
leaving Gazebo available. Use **Reset Scope → Reset Gazebo → Reset** before
repeating a transfer. Restart the UI to load the updated Python code.

ProductAgent plans `pick_part`, `move_to_resource`, and `place_release` through
the nominal ProcessPlanner. KMR ResourceAgent dispatches through the existing
CCA protocol. Storage and M1 participate in the same atomic nominal handoffs.
`move_to_resource` retains the `DockKMR` controller and `/KMR/dock` endpoint.
Arm/gripper controller results, acknowledged attachment/release, measured part
poses, and robot withdrawal are required before task completion. Ordinary
placement records neither machining nor assembly completion.

M1 and M2 now use the compact envelope in
[MACHINING_STATION_LAYOUT.md](MACHINING_STATION_LAYOUT.md); M1 still handles
square pegs and M2 circular pegs. Loading openings, interiors, stands, and
fixed-joint grasp attachment are simulation assumptions. Gazebo evidence is
saved under
`cais_spade_llm/monitor/recovery_gazebo_runs/<run>/run.json`, and can be read in
**results → Gazebo delivery runs**. Saved inputs, descriptors, configuration
fingerprints, plans, observations, acknowledgements, histories, and outcomes
remain tied to that run. Rendering and report selection perform reads only.

The active `lg_slippage.json` file and its resource bindings have been removed.
Historical results and isolated legacy test fixtures are retained; `LG` is not
an available NIST component. NIST failure settings and resource restrictions
remain blocked for this delivery until their execution is integrated.

[Adaptive requirement/capability matching](ADAPTIVE_REQUIREMENT_CAPABILITY_MATCHING.md)
for this Gazebo delivery is **planned, not implemented**. Failure injection, recovery behavior, full
assembly execution, optional setup plan generation, and the outstanding
selector correction remain subsequent work. This delivery uses the existing
`SystemBridge` Start/Stop interface. The concurrent read-only capability accessor
is preserved; this delivery makes no CCA or Java repository changes.

### Recorded validation (2026-09-21)

**Observed Gazebo transfer:**
[`20260921T194635_5b778116/run.json`](../../cais_spade_llm/monitor/recovery_gazebo_runs/20260921T194635_5b778116/run.json)
records three acknowledged tasks through the real Start System agent path:
`pick_part`, `move_to_resource`, and `place_release`. M1 finished `loaded` with
`KET4_Square_4mm`; Storage inventory became false; KMR finished `idle`, empty,
and at M1. The released part was observed at approximately
`(-6.080000, 1.820000, 1.059990)` m, and the withdrawn TCP at
`(-6.845278, 1.927482, 2.271986)` m. `processCompleted` remained empty.
Reapplying the recorded Gazebo acknowledgements through the shared projector
reproduced the final resource and ProductState values.

The [control check](../../cais_spade_llm/monitor/recovery_gazebo_runs/20260921T194635_5b778116/control_verification.json)
records a harmless duplicate Start, Stop retaining Gazebo, and rejection of a
repeat transfer without an explicit scene reset. The acceptance harness used
`SystemBridge.start_system()`/`stop_system()` and the same preparation function
as the Run page, with headless Gazebo. It was not a physical robot test.

**Observed Stop during motion:** the separate
[control check](../../cais_spade_llm/monitor/recovery_gazebo_runs/20260921T200209_5d5b706e/control_verification.json)
records the same active arm goal changing from status `2` to `6` after Stop
System. Cancellation reaches the KMR arm/gripper controllers before waiting
for MoveIt cancellation. No task was committed: Storage retained the part,
KMR remained empty, and Gazebo stayed available. Task IDs include the run ID
so delayed acknowledgements or CCA messages cannot match a later run.

**Static and focused checks:** the final combined check passed 354 tests;
three UI assertions changed during the concurrent edits. Re-running the entire
affected Resources/setup group passed all 47 tests, including those three.
Coverage includes delivery, nominal agent transitions, resource DES, setup/pages,
product configuration, Gazebo layout, RViz startup, and `place_insert` release
regressions. `poetry check`, full
`compileall` for `cais_spade_llm` and `ros2`, UI `--help`, and `git diff --check`
passed. `make bootstrap-gazebo` completed all 16 packages. Poetry retains its
existing metadata deprecation warnings.

Current geometry also passed [collision-aware IK and complete Cartesian
approach checks](../../cais_spade_llm/monitor/recovery_gazebo_runs/20260921T200209_5d5b706e/machine_paths.json)
for KMR at M1/M2, `ur5e-1` at M1, and `ur5e-2` at M2. These
planning checks do not establish executed UR5e handling. The enclosure, robot
openings, and fixed-joint grasp remain simulation assumptions. Failed
commissioning attempts are retained with their observations; they are not
successful delivery evidence.

The completed transfer predates the stand-height correction from 0.98 m to
1.00 m; its saved snapshot retains that configuration. The later reach and
active Stop checks use the corrected stand. The retained Gazebo scene is from
the Stop test, with the arm stopped during pickup; reset it before another run.

`completion_conditions` currently supports one selected part with `quantity: 1`.
The separately edited Products/Resources views and removal of the offline-run
UI are preserved; their broader model changes are outside this delivery's
Gazebo acceptance claim.

## Current Implementation Roadmap

Follow [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and the
[M1/M2 layout](MACHINING_STATION_LAYOUT.md) ([drawing](MACHINING_STATION_LAYOUT.svg)).
**Phase 2 completed 2026-09-16.** The compact machine increment above
supersedes its M1/M2 geometry and KMR docking poses. **recovery-framework → run → start simulation**
opens [table_recovery_framework.world](../../ros2/cais_lab_robotics/worlds/table_recovery_framework.world).

The project tabs are **run → setup → results**, with **run** open by default.
Opening the page does not start simulation or execution. **setup** selects the
Product Order, permitted resources, Safety, **Simulation** or **Physical** mode,
existing recovery settings, and failure scenario. **Save setup** writes
[`recovery_framework_setup.json`](../../cais_spade_llm/initialization/recovery_framework_setup.json)
only. **run** displays that saved setup alongside the existing **Start System**,
**Stop System**, readiness, progress, and recovery controls. Dry Run controls
are not available in System Control. **results** inspects Gazebo delivery runs
and recovery sessions, including failed and incomplete attempts,
with search, artifact inspection, and CSV export. It does not load a proposal into
the runtime. Missing measurements are `not recorded`; validation, simulation
execution, and physical outcomes remain separate. Artifacts without a recorded
session identifier remain individually inspectable and are not counted as trials.

**UI/settings phase (2026-09-21):** edit products and orders in **Products**,
inspect capability definitions and Live Robot Status in **Resources**, and edit
and verify requirements in **Safety**. Their setup links return to experiment
selection. The default experiment uses the current NIST product, `quantity: 1`,
all 12 resources, and no failure. M1/square and M2/circular assignments remain.
Raw settings and DES details are expandable. Browsing and saving settings do
not plan tasks, dispatch commands, inject failures, or change custody.

Setup supports **Conveyor breakdown**, **ur5e-1 breakdown**, **Machining breakdown
during part processing**, and **Part slippage** as saved future scenarios.
Part slippage selects an exact permitted manipulator, a NIST component shared by
the Product Order and that resource's eligible parts, and an applicable task.
`before_execute` precedes task execution; `after_execute_before_commit` follows
execution but precedes recording success. Occurrence is once per run. The drop
position and unit quaternion are a configured `world` target, not an observation.
An optional condition can require another resource to hold another exact part.

Use **Example: ur5e-3 / KET4_Square_4mm** or **Example: ur5e-4 / gear_large** to
populate a draft, then enter the drop position. These resources have different
eligible parts. One parameterized configuration covers all 11 components.
Active `lg_slippage` settings and resource bindings are removed; legacy `LG`
records remain historical evidence only. No recovery sequence
is preselected. A valid saved failure is labelled **execution not integrated**;
Start System blocks selected failures and resource restrictions until their
execution is implemented. An excluded resource is unavailable from the start;
a breakdown happens later.

Historical recovery evidence never borrows the current setup to
fill missing fields; a configured experiment is not a recorded Gazebo outcome.

**Accepted Gazebo baseline:** Storage is on the left, M1/M2 remain side by
side with a **1.5845 m horizontal gap to the assembly table**, and both machine robots
share one straight passive Conveyor. `ur5e-1` handles M1 and `ur5e-2` handles M2.
At Assembly Station, **`ur5e-3` is the top robot** (`y=0.50`) and **`ur5e-4` is
the bottom robot** (`y=-0.50`). Each has independent arm/gripper controllers,
joints, frames, and MoveIt groups.

Buffer For Machined parts is a straight, four-zone accumulation conveyor
connected to Conveyor by a short inclined transfer plate and tapered infeed.
The square and round pegs travel directly on the belts, lying horizontally in
a shallow fixed guide channel. When the next zone is empty, the occupied zone
advances its peg; parts do not push one another. Zone 4 stops for `ur5e-3` to
pick directly into assembly using one grasp. No removable carriers, carrier
supply stands, or empty-carrier collection tray are present. A full buffer
applies backpressure to Conveyor; machine robots use their staging nests.
Transport, sensing, stop actuation, and occupancy control remain planned.

`prusa_mk4_2` is beside `ur5e-4` at `(0.50, -0.50, 1.04)`, with its open front
facing that robot. All three exact models, `gear_small`, `gear_medium`, and
`gear_large`, start separately on its bed. Their inventory does not change the
active order. `Exit` remains an empty capacity-one tray, assigned to `ur5e-3`
for the future completed `assembly_board-v1` transfer. The four KET and four
RGOCG parts remain in the rotated Storage kitting trays.

The source world keeps one static `KMR` layout reference at
`Storage_KMR_docking_pose = (-8.24, 2.13, 0)`, with base yaw
`1.57079632679` and local arm yaw `1.57079632679`. The arm mount remains
`(-0.25, 0, 0.70)`. M1/M2 docks are `(-6.85, 2.15, 0)` and
`(-3.45, 2.15, 0)` at the same base yaw. The current enclosure and access poses
are documented in the machine-layout page; previous poses remain historical.

**Phase 3 has started with KMR simulation control.** The recovery launch removes
that one static KMR include from a temporary runtime-world copy and spawns one
articulated `KMR` at the same accepted pose. Its planar KMP omniMove 400 base,
seven LBR iiwa joints, and `KMR_rg2_finger_width` joint publish live state.
Nav2 now plans collision-aware base paths against the committed fixed cell map.
The named `/KMR/dock` action retains the configured Storage–M1 and Storage–M2
routes; M1-to-M2 docking remains rejected until KMR returns to `Storage`.
Arbitrary travel uses a `1.20 m/s` simulation speed override. This exceeds the
KMP omniMove 400 physical limits and cannot support physical cycle-time claims.
Named docking follows the configured,
map-validated Storage–M1/M2 waypoints deterministically while holding the
requested yaw. It travels at up to `1.20 m/s` and slows to `0.40 m/s` for the
last `0.20 m`. A `1.00 m/s²` command ramp prevents the planar simulation plugin
from applying an abrupt velocity step to the articulated KMR. Return to Storage
from an arbitrary pose uses Nav2 to reach the
Storage approach before the deterministic final motion.

At startup, the recovery base controller moves the KMR iiwa to its upright
transport pose and maintains that pose while the base travels. The articulated
KMP root, iiwa, and RG2 links are kinematic and gravity-free in this recovery
simulation so inertial impulses from the planar base plugin cannot interrupt
base motion. Their commanded joints remain available through `ros2_control`.
Dynamic wheel-force, payload, and joint-torque fidelity remain later validation
work.

The recovery MoveIt model adds `KMR_iiwa_arm` and `KMR_rg2_gripper`.
`all_robots` contains the four UR5e arms and `KMR_iiwa_arm`. The recovery
configuration no longer contains `dual_robots`. Recovery-specific RViz
markers wait for live joint and KMR odometry state, then place five orange arm
targets at the current TCP transforms and one blue KMR docking marker at the
current base pose. RViz now starts only after that live-state gate passes, so
the five built-in orange MoveIt goal models also initialize at the current TCP
poses instead of their default poses.
The combined RViz model receives the configured `Storage_KMR_docking_pose`
through a steady-clock preview as soon as the KMR controller starts, so it no
longer appears briefly at the world origin while Gazebo loads. `/KMR/dock` and
the blue actionable base marker still require fresh `/KMR/odom`. The KMR
base is controlled separately from `ros2_control`. Drag and rotate the blue
`KMR_base` marker to stage any target within the map, then right-click it to
plan, execute, plan and execute, reset, or cancel. Named **Dock at Storage**,
**Dock at M1**, and **Dock at M2** actions remain available. Dragging alone
never moves KMR; Nav2 rejects targets outside the map or in fixed obstacles.
For immediate random travel, select **Nav2 Goal** in the RViz toolbar, click a
clear floor position, drag the arrow to set yaw, and release. RViz sends that
goal to the validated proxy through RViz's native `/navigate_to_pose` endpoint;
the same proxy remains available explicitly as
`/KMR/validated_navigate_to_pose`. The arrow disappears when it is released
because release submits the immediate plan-and-execute request. The proxy
requires fresh odometry, the parked iiwa state, available Nav2, and an unused
base-action slot.
The blue marker's separate Plan/Execute path uses
`/KMR/validated_follow_path`. Its menu action captures the blue marker's current
pose directly. For a reviewed request, drag `KMR_base`, choose **Plan KMR
base (path only)**, wait for the path, and then choose **Execute stored KMR base
plan (moves)**. **Plan+Execute KMR base (moves)** performs both steps. **Cancel
KMR base motion** stops either workflow.
The recovery RViz configuration includes the **Navigation 2** panel required by
the **Nav2 Goal** tool. The visible blue `KMR_base` pad is the x/y drag surface;
select **Interact** in the RViz toolbar, left-drag the pad, use its ring for
yaw, and right-click the same pad for its menu. **Publish Point** is not a
navigation command because it provides no yaw.
The duplicate `RobotModel` display is disabled and the display rate is 15 FPS
to reduce rendering load while the MoveIt scene continues to show all five
robots.
The fixed occupancy-map display remains available in RViz but starts disabled
because the WSL OpenGL driver fails its indexed-map shader; Nav2 still loads
and enforces the same map. The planned paths and KMR controls remain visible.
The fixed map does not detect people or moved equipment. The UR controller spawner waits directly for
`/controller_manager`; its successful activation then starts the serialized KMR
spawn and KMR controller spawner without waiting for the stalled UR Gazebo
spawn-client response. All recovery MoveIt arm requests now default to 40%
velocity and 30% acceleration while retaining the configured joint limits.

Direct named M1-to-M2 docking remains outside `predefined_routes`. A future
machining-breakdown recovery event may generate that candidate through the
northern aisle, but execution must first validate Nav2 clearance, the parked
iiwa state, M2 availability, docking tolerance, and MoveIt manipulation.

**Known NIST Products is complete.** Select
`assembly_board-v1-recovery-framework` on the Products page to see all eleven
exact components in **Known NIST Components** and **Selected Parts**. Its default
order uses `"parts": "all"`; saved subset orders preserve the same exact
identifiers. The recovery geometry binds every component to its CAD filename,
Gazebo model, initial source resource, and assembly target without a
Spec2Primitives recognition dependency. `GMC_Laser_Plate_Virtual`, `Gear_Plate`,
and `Gear_Shaft_1`–`Gear_Shaft_3` remain targets rather than selectable parts.

**Next:** integrate NIST failure injection, resource restrictions, and recovery
behavior with resumption. Optional setup plan generation and full assembly
execution remain future work. KMR execution is limited to the explicit delivery
order above. Machining, Conveyor/buffer transport, printing, and recovery trials
remain planned for observed execution. Earlier xArm6 and hardware references
below describe the previous framework and preserve their original identifiers.

## Working Thesis

CAIS-SPADE-LLM provides a layered recovery framework for multi-agent manufacturing
systems where formal recovery, LLM-assisted bridge generation, safety validation,
and resource-agent execution work together after runtime failures.

This paper should focus on the recovery architecture itself. Keep Ontology RAG as
future work or as a short discussion pointer to Journal Paper 3.

## Core Idea

The system starts from a monitored production plan and reacts when execution
deviates from the expected product/resource state. Recovery is handled in layers:

1. DES-style recovery checks whether an enabled recovery path already exists.
2. LLM bridge generation proposes recovery actions when the compiled model is
   too limited.
3. Validation checks generated recovery actions before runtime execution.
4. Resource agents execute the accepted recovery task or primitive program.
5. Operator surfaces and the digital twin help inspect, preview, replay, and
   validate recovery behavior.

## Main Contributions

- A layered recovery framework that separates formal recovery, LLM-assisted
  proposal, validation-stage checks, runtime gating, and execution.
- A bridge validation pipeline that rejects unsafe or infeasible recovery actions
  before they become runtime tasks.
- Resource-agent primitive execution for recovery actions, including dry_run,
  Gazebo, digital twin, and hardware-facing paths where available.
- A practical manufacturing case study with xArm6, UR5e, task execution,
  safety constraints, and recovery after failure.

## Validation-Authority Boundary

The recovery outline uses explicit agent ownership rather than treating the PA
as a proxy validator.

| Recovery operation | Authoritative component | Paper interpretation |
| --- | --- | --- |
| Proposal generation | LLM called by PA | `pure_llm` produces three candidate events and selects one index; `neurosymbolic` produces one to `candidate_proposal_budget` next-event candidates without selecting. |
| Syntax and grounding | PA | Verifies response structure and exact formal-system bindings. |
| Transition feasibility | PA | Verifies exact expected-start agreement and the Product-owned `part_traceability` invariant on the proposed successor, including the responsible RA's exact carried-part location. |
| Recovery admission | PA | Rejects `no_state_change` and `label_only_state_change`; this concrete-effect policy is reported separately from DES transition feasibility. |
| Physical feasibility | Responsible live RA | Uses a fresh RA snapshot and resource-specific feasibility checks. |
| Local plant model | Responsible live RA | Privately supplies `G_r` as a task-level extended finite automaton with exact variables, finite domains, valuation, local alphabet, guards, updates, event classifications, marked conditions, and descriptor fingerprint. The LLM does not receive this descriptor. |
| Candidate safety | Live CCA | Uses current CCA safety rules and `validate_outline_macro_recovery_safety()`. |
| Model-based selection | PA | In `neurosymbolic`, compares PA/RA/CCA-valid successors through exact open recovery obligations, recovery-relevant enabled events, and exact nominal-reentry enabled events. |
| Candidate commit | PA | Rechecks the PA fingerprint. `pure_llm` preserves valid LLM selection; `neurosymbolic` commits the lexicographically preferred symbolic successor and uses a stable exact-effect representative only for a remaining tie. |
| Post-outline safety generation | CCA | Produces the recovery safety bundle after outline completion. |
| Runtime safety gating | CCA | Supervises dispatch and execution; this is not the outline-time candidate check. |
| Primitive execution | Responsible RA | Generates and executes the resource-owned primitive program. |

For each candidate, the PA first performs its own checks. It then sends a
correlated `recovery_outline_physical_validate` request to the exact RA named by
`resource_jid`. The RA refreshes `get_recovery_snapshot()`, validates physical
feasibility, and replies with its snapshot, snapshot fingerprint, local recovery
DES descriptor, and descriptor fingerprint. RA-valid candidates
are sent to the CCA through `recovery_outline_safety_validate`. The CCA returns
the safety findings, active rule identifiers, cleared and remaining condition
identifiers, and its safety-rule fingerprint. Request id, recovery session id,
turn index, PA state fingerprint, and sender JID are checked before PA accepts a
reply. Missing, malformed, stale, incorrectly routed, or timed-out validation
fails closed.

This wording is important for the paper: the PA orchestrates validation but does
not claim RA physical authority or CCA safety authority. Outline validation is
also snapshot-specific. Passing it does not guarantee primitive synthesis,
inverse kinematics, collision-free MoveIt planning, trajectory execution,
Gazebo behavior, or hardware execution.

## Neurosymbolic Selection Method

The experimental comparison has two explicit modes. `pure_llm` is the baseline:
the LLM authors exactly three one-action candidates and returns
`selected_candidate_index`. The selected candidate remains authoritative only
when PA, the responsible RA, and CCA all accept it. `neurosymbolic` keeps LLM
proposal generation mandatory because the recovery event alphabet is incomplete,
but the LLM does not rank or select its one to `candidate_proposal_budget`
proposals.

Use conventional DES notation for the plant:

`G = G_P || (||_{r in R} G_r)`.

`G_P` is the product automaton and each RA owns its local extended finite
automaton `G_r`. Its descriptor declares the exact local variables and finite
domains, current valuation, local event alphabet, controllable and observable
classifications, guards, updates, and local marked conditions. The CCA induces
the supervised plant `S/G`. PA stores and composes these descriptors but does
not infer their semantics from event names, descriptions, resource types, or
Case 3 tokens.

Robot descriptors contain task-level `RobotTaskProgram` transitions rather
than controller primitives, sensing functions, or geometry-computation
functions. Continuous pose and workspace evidence remain outside the finite DES
state and are used by the RA physical validator. `RobotAgent` retains
binary `gripper_state` only as private runtime and physical-validation evidence.
It is not a RobotAgent recovery DES variable or an LLM candidate field. For a
later unexecuted outline transition, RobotAgent may privately derive this
evidence from explicit `held_part`. `PrintingAgent` represents `pause_job`,
`resume_job`, and `cancel_job` through `resource_state` only. Printer snapshots and primitive
execution still retain `job_state` and `active_job` as runtime evidence, but
they are not PrintingAgent DES variables. Another ResourceAgent declares only
the exact finite variables needed by its own enabledness, marked conditions,
safety checks, or physical validator. It must implement the same private
descriptor interface or provide an explicit initialization descriptor; a
missing or invalid descriptor fails closed.

The LLM receives only compact live symbolic state, observations needed for
grounding, reachable locations and named poses, recovery goals, active safety
rules, and exact current-turn validation findings. Event alphabets, guards,
updates, marked conditions, descriptor fingerprints, primitive catalogs, and
accepted-transition stacks are not prompt material. Complete private models
remain auditable in internal session state and outline result artifacts.

The response schema uses the union of RA-declared state-field names so one LLM
response can contain candidates for different resources. PA nevertheless
authorizes each candidate against only the descriptor owned by its exact
`resource_jid` and enforces each field's declared `resource` or `part` scope.
Thus neither `gripper_state` nor `job_state` is accepted by the current
RobotAgent and PrintingAgent recovery descriptors. Resource-specific values
must belong to the responsible RA's exact finite domain. Adding another ResourceAgent changes
its descriptor, not PA validation code. Accepted declared values are preserved
in PA state and the RA/CCA projections.

The LLM can author any nonempty exact `event_name` and may optionally introduce
a new nonempty `resource_state` or `part_state` in the candidate end state.
These state values are uninterpreted labels: they do not extend or mutate the
live RA descriptor. A committed label is retained only in the recovery
session's projected state and must be matched exactly by the following
candidate. A label alone supplies neither transition semantics nor selection
progress. PA accepts it only with another concrete declared effect or when the
exact state value clears a supplied recovery obligation. No validation or
selection stage derives motion, acquisition, release, safety, or progress from
wording inside an LLM-authored event or state name.

### Transition feasibility and explicit successor states

At outline turn `k`, let `q_k` denote the PA's authoritative projected
symbolic state. Each LLM candidate explicitly declares an
`expected_start_state` and an `expected_end_state`. DES enabledness is checked
by exact agreement between every declared start field and the corresponding
field in `q_k`. The declared end state is the proposed successor; PA does not
infer a missing `part_location` from `held_part`.

PA accepts the successor as transition-feasible only when the Product-owned
`part_traceability` invariant also holds. A candidate with `part_name` must
include `held_part`, `part_state`, and `part_location` in both state objects. A
null start `part_location` is permitted when the current location is genuinely
unknown, but an acquisition or release must declare a supplied non-null end
`part_location`. Whenever the responsible resource holds the part, the end
`part_location` must equal that RA's exact declared carried-part location. The
projected holder must agree with explicit `held_part`, and no two projected
resources may hold the same part.

Exact-start and `part_traceability` failures are reported as
`transition_feasibility`. Candidate shape, exact-token, field-scope, and
RA-domain failures belong to `syntax_and_grounding_validation`. The additional
`no_state_change` and `label_only_state_change` policy belongs to
`recovery_admission`, not standard DES transition feasibility.

Once accepted, an LLM-authored event contributes one validated
recovery-session transition `(q_k, event_name, q_{k+1})` extending the nominal
transition relation used by the recovery outline. Newly authored
`resource_state` and `part_state` values remain uninterpreted exact labels and
carry no meaning by name alone.

The recovery relation is extended through atomic custody transitions. An
acquisition explicitly changes `held_part` from null to `part_name` and assigns
the responsible resource's declared carried-part `part_location`. A subsequent
placement or release changes `held_part` from `part_name` to null and assigns
the concrete destination. A candidate that keeps `held_part` null while moving
`part_location` is rejected as `part_relocation_without_carrier`, because it
would conceal acquisition, transport, and release inside one transition and
would bypass the separate PA, RA, and CCA checks for those custody states.

Revision guidance is derived from validation rather than from a scripted
recovery answer. Only after this carrier rejection, PA obtains the acting
resource's exact carried-part location from its existing `ResourceProfile` and
checks that token against the same RA's private `part_location` domain. The
same scoped correction is used for `held_part_location_mismatch` and
`missing_acquisition_location`. Other resources' tokens remain absent. A
resource without such a declared token receives no custody hint and fails
closed. Carrier and workspace findings persist only while their
authoritative symbolic, observation, and capability evidence remains unchanged,
and clear after custody or that evidence changes. Rejected rationale, accepted
event names, candidate order, and the transition stack are not fed to the next
LLM turn.

Recovery completion is Product-owned accounting performed after transition
validation. Reaching a part's exact `goal_location` does not clear its recovery
obligation while the part retains a projected holder or any projected resource
reports `held_part == part_name`. Any exact supplied final part-state condition
must also hold. Conversely, the method does not require a fixed vocabulary such
as `placed` or `assembled` when no exact final state label was supplied.

The outline prompt keeps compact live state, observations, genuinely available
named poses and reachable locations, recovery goals, active safety rules, and
current validation findings. Nominal `origin_location` remains private to
nominal-reentry comparison and debug evidence instead of steering the LLM's
proposal. These live capabilities, goals, and safety rules are instance
conditioning, not hard-coded recovery answers.

For current composed state `q`, let `O(q)` be the exact open recovery
obligations: PA Recovery Goals, continuation blockers, reentry requirements,
and CCA safety-condition identifiers. Let `Q_m^R` be the set of
recovery-compatible marked states. Define the recovery-enabled set as

\[
\Gamma^R_{S/G}(q)
= \Gamma^R_G(q) \cap \Gamma^R_{\mathrm{RA}}(q) \cap \Gamma^R_S(q).
\]

Here, `Gamma^R_G(q)` is the backward-relevant set of bound RA-declared events
whose exact symbolic guards hold; `Gamma^R_RA(q)` is the subset physically
realizable according to the responsible RA; and `Gamma^R_S(q)` is the subset
admitted by the CCA safety projection. RA queries bind the exact affected
resource, part, location, observation, and projected state. Their snapshot and
descriptor fingerprints, and the corresponding CCA fingerprints, make this an
explicitly snapshot-specific enabledness result. Every PA/RA/CCA-valid LLM
event `e_i` yields `q'_i = delta(q, e_i)`.

Let `Gamma^N_{S/G}(q)` denote the CCA-admissible exact nominal-reentry events
whose verified task guards hold for affected resources and parts on their
continuation paths. It is computed from verified task status and guards,
resource/part state, holder and location facts, and CCA projection. Unrelated
unfinished nominal tasks are excluded.

A candidate is progressing when one of these ordered cases holds:

1. `O(q'_i)` is strictly smaller and the successor introduces no obligation;
2. `O(q'_i)` is unchanged and `Gamma^R_{S/G}(q'_i)` is a strict expansion; or
3. both are unchanged and `Gamma^N_{S/G}(q'_i)` is a strict expansion.

Selection applies the same lexicographic order. If symbolic evidence remains
tied or incomparable, the smallest existing exact-effect `candidate_id` is
appended immediately as a stable representative. This reproducibility
convention is arbitrary and is not an optimality claim. No valid/progressing
event returns `need_revision`. Three repeated turns with no valid progressing
candidate under unchanged plant and supervisor fingerprints terminate as
`selection_unresolved`. No LLM preference, candidate position, `event_name`,
`rationale`, resource type, location wording, distance, duration, energy,
execution effort, or random rule is used. PA, RA, and CCA never invent a
symbolic fallback event.

When one candidate is committed, PA retains only rejected PA/RA/CCA findings
that still apply to the committed state and unchanged evidence. For example, a
`workspace_unreachable` result remains available to the next LLM turn after an
unrelated candidate is accepted, then clears when the affected observation,
holder/location state, capability fingerprint, or physical result changes.
Accepted event names, rationales, comparison scores, and the transition stack
remain debug-only; they are not included in later prompts.

CCA owns safety projection and returns `safety_dfa_states_before` and
`safety_dfa_states_after`. The first candidate uses the live CCA DFA state;
later unexecuted outline transitions use the previously selected projected DFA
state. Temporary validation monitors do not mutate the live monitor. Stale rule
or live-state fingerprints fail closed.

The method should be described as one-step receding-horizon logical/symbolic
supervisory selection, not BFS, global shortest-path recovery, optimal
supervisory control, or a formal nonblocking supervisor over a complete event
alphabet. Cost-based optimal supervisory control is reserved for a future
comparison mode. Related DES literature separates logical requirements from
optional cost or throughput objectives; representative comparisons include
[Automatica](https://www.sciencedirect.com/science/article/pii/S0005109824001274),
[IFAC optimal DES control](https://www.sciencedirect.com/science/article/pii/S1474667017512197),
and [IFAC throughput control](https://www.sciencedirect.com/science/article/pii/S1474667015374036).
Outline validation remains snapshot-specific and does not
guarantee primitive, trajectory, Gazebo, or hardware execution.

## Current System Strengths

- Recovery is tied to product/resource state rather than free-form text.
- DES enabledness and validation-stage checks are separate from downstream
  runtime gating.
- The system already has bridge-visible primitive programs and resource-owned
  execution surfaces.
- Monitor, Teach, Preview in Gazebo, Replay in Twin, and digital twin provide
  useful operator-facing recovery workflow pieces.

## Main Limitation To Admit

The current recovery framework has dynamic context retrieval, but the retrieval
surface is mostly procedural and ref-based. The LLM can request useful context,
but the framework does not yet use a first-class PPR graph to proactively decide
which Product, Process, Resource, primitive, fact, and constraint context should
be retrieved before generation.

This limitation becomes the motivation for Journal Paper 3.

## Paper Outline

1. Introduction
   - Flexible manufacturing needs recovery after runtime failure.
   - Pure formal recovery is safe but limited.
   - Pure LLM recovery is flexible but hard to trust.
   - This paper combines layered recovery with validation and execution.

2. Related Work
   - Multi-agent manufacturing systems.
   - Product agents and resource agents.
   - DES and supervisory control for manufacturing.
   - LLM planning and robot task generation.
   - Runtime safety validation and recovery.

3. System Architecture
   - Product agent, resource agents, and central controller.
   - Task planning, execution monitoring, and failure detection.
   - Resource-agent execution modes.
   - ROS2/Gazebo/MoveIt and digital twin context.

4. Layered Recovery Framework
   - Failure context.
   - DES recovery path.
   - LLM bridge proposal path.
   - Validation-stage checks.
   - Runtime dispatch and resource-agent execution.

5. Primitive Recovery Execution
   - Bridge events and primitive_steps.
   - Resource-owned primitive catalogs.
   - Primitive validation, event_facts, and projected state.
   - Recovery macro execution.

6. Experiments
   - Normal execution vs failure execution.
   - DES-only recovery vs LLM bridge recovery vs full validated recovery.
   - dry_run, Gazebo, digital twin, and hardware-facing demonstrations where
     available.

7. Discussion
   - What the framework handles well.
   - Limits of ref-based dynamic retrieval.
   - Why PPR Ontology RAG is the next step.

8. Conclusion
   - Layered recovery improves flexibility while preserving validation.

## Experiment Plan

Use small, repeatable failure cases:

- Resource enters failed state during a task.
- Part is held by the wrong resource.
- Destination or expected state does not match the plan.
- Primitive proposal misses required event_facts.
- Recovery action is not enabled from the current symbolic state.

Compare:

- No recovery.
- DES-only recovery.
- LLM bridge without full validation.
- Full layered recovery framework.

Measure:

- Recovery success rate.
- Invalid proposal rejection rate.
- Safety violation count.
- Recovery latency.
- Number of LLM turns.
- Primitive program validity.

## Figures To Produce

- Layered recovery architecture.
- Validation-stage vs runtime-gating diagram.
- Recovery sequence example.
- Primitive recovery execution trace.
- Experiment result table or bar chart.

## Immediate Writing Tasks

- Freeze the title and contribution list.
- Extract one representative recovery scenario for the running example.
- Write the before/after recovery example.
- Decide which experiments can be repeated reliably before submission.
