# Resource-owned Prepare and check

`Prepare and check` on **recovery-framework → recovery → Gazebo safety preparation**
captures registered owners, prepares motion, and saves a detached CCA analysis.
`dispatch_authorized` is always `false`. A result named `allowed` belongs to the
finite offline composition problem; it is not a CCA grant or an active proof.
The existing admission path, Safety-page selection, reviews and approvals remain
unchanged. No system is started by opening the page or pressing this action.

The selected predefined document supplies the reviewed AP meanings, formulas and
scopes. Compilation does not call an LLM. Existing active physical definitions must
agree with the selected document; native monitor histories are copied into the
same branch analysis where present. The physical and native clocks stay distinct.
Missing task continuation or physical history cannot be replaced by initial DFA
states. A new physical requirement needs an explicit prospective checkpoint.

## Input and saved evidence

The page accepts a JSON object with `recovery_id` and `programs`. Each program has
its exact `resource_id` and nonempty `primitive_steps`. Each step retains `primitive`,
`params` and `source`; `source` contains the original `outline_id`, `des_event_id`,
`event_name` and integer `step_index`. Use the bound values from the recovery plan.
The page does not accept safety verdicts, monitor states, synthetic-mode switches,
geometry or trajectories as authority.

Each attempt writes a new `result.json` beneath
`cais_spade_llm/monitor/recovery_safety_preparation/<attempt_id>/`. It retains the
checkpoint and capture times, owner/JID/run/launch identities, resource/product
revisions, ledgers, reservations, pending tasks, monitor states, selected definitions,
original programs, prepared trajectory points and timings, source references and
fingerprints. `resource_support` reports observation/preparation status and the
declared model for each configured resource. Available composition evidence includes each rule's local AP values,
bindings, observation indices, completion witness or counterexample. Missing
evidence is recorded under `unresolved`. Past attempts are not overwritten.

## Current observation and preparation contract

The [adapter](../cais_spade_llm/recovery_framework/gazebo_safety_preparation.py) uses
every resource from `build_environment_models(scene)`. It requires exactly one
registered owner per configured resource. It captures symbolic state under the
existing admission lock, reads observations, and rechecks revisions and history.
It does not call `EnvironmentAdmission.synchronize`, register a composition,
install a proof, or issue `allow`.

Shared orchestration reads only owner hooks and registered entity readers. It
does not inspect robot-controller fields or require joints, a gripper, or an
attachment field from fixed equipment. Each active motion contract declares the
physical facts it needs. `ResourceAgent` defaults return `NEEDS_CONTEXT`;
`RobotAgent` delegates to its initialized controller while holding its motion lock
for preparation. The controller is not initialized as a side effect of checking.

With the registered reader, the controller reads its configured joint identities
from the stamped physics snapshot. Fixed tool frames that Gazebo has merged out
of its link tree are derived from those observed joints and configured FK; the
record labels that source explicitly. URDF and interpolation parameters are
rechecked during capture and reobservation. Legacy capture without this reader
retains its joint/TF/entity checks and 1 mm / 0.001 rad agreement limits. The generic
`recovery_framework` launch fallback is rejected. Part/entity observations use
configured model identities and simulation/receipt timestamps. Fixed equipment
is not assigned invented gripper states. The capture window defaults to two
seconds; stale, inconsistent or incomplete evidence remains unavailable.

Configured `scene.safety_geometry` supplies the existing 3D observation geometry
format for supplied evidence. The registered live path uses `scene.safety_preparation`:
explicit reviewed region bounds, exact model/link identities, controller observation
channels and owner coverage contracts. `/GETRECOVERYSTATE` reads model/link poses,
joints, finite collision bounds and attachment identities together on Gazebo's
physics thread. Resource and part envelopes retain their observation and configuration
sources; configured URDF/mesh hashes bind prepared moving geometry. Missing finite
part geometry remains unavailable; an empty carrier link is not assigned a guessed box. Static world poses
do not supply safety-region bounds or future stationary intervals. Observing an
empty gripper does not establish a detached physical part. Existing controller
attachment fields remain last acknowledged state and do **not** establish custody.
The new read-only attachment-owner snapshot supplies the complete observed attachment
set, instance identity, revision and simulation timestamp. This first motion-only
slice requires empty custody, reconciled with owner records; it does not prepare
grasp/release or infer completion from a part pose.

The [existing controller](../cais_spade_llm/resources/robot/gazebo_pick_place_controller.py)
supports UR5e `move_cartesian(x, y, z, speed, qx, qy, qz, qw)` with fixed orientation
and the existing complete Cartesian/collision-checking policy. It does not rewrite
these commands into KMR `target`/`seed` parameters. The controller's existing
Cartesian planning/timing/validation operation is separated from dispatch; the
normal execution wrapper still dispatches as before. Preparation uses explicit
start joints, never consumes the execution cache, and preserves every trajectory
point, derivative and nanosecond. Later motion steps may plan from projected
endpoint joints without updating actual resource state. Preparation uses a detached
controller object with the existing read/planning clients; its command-evidence and
timing fields are never written back over the live controller's execution feedback.

UR5e helper, named-pose, grasp/release, orientation-changing and other unresolved
effects remain `NEEDS_CONTEXT` in this first slice. The complete original program
is retained even when only its initial motion steps can be prepared. No helper
outputs, durations or effects are invented. Existing KMR checking is unchanged.

## Registering another resource

Trusted application code calls `register_resource_provider(identifier, provider)`
from [resource_safety_preparation.py](../cais_spade_llm/resources/resource_safety_preparation.py).
The provider declares a positive integer `version`, `validate_programs(...)`, and
`create_owner(...)`. Configuration supplies only a registered identifier, never
Python source, a module path, or an executable object.

`scene.resource_models` maps each exact additional resource identity to
`{"model": ..., "owner_provider": ...}`. `model` uses the existing environment
model fields: `resource_id`, `resource_type`, `assignments`, `state_variables`,
`current_valuation`, `marked_state_conditions`, and `events`. Each event retains
its exact `event_id`, `event_name`, `parameter_bindings`, `participants`, `guards`,
`updates`, `product_effects`, `controllable`, and `observable` declarations.
Programs remain in `resource_programs.resources`, using `functions` and
`primitives`. The provider validates native program contracts. Registration does
not change `recovery_selectable` or supply an execution capability.

`build_environment_models(scene)` appends these models to the existing population.
Duplicate identities, inconsistent shared events, invalid participants, absent
programs, and unknown providers fail explicitly. Existing owners retain their
identities; additional owners are created through the registered factory. The
same complete-population grounding expands mutex to 78 pairs when one resource
is added to the current 12. An explicitly bound requirement keeps its bindings.
Missing owner, geometry, or stationary/motion coverage never excludes a resource.
Existing parameterized declarations such as `inventory.{part_name}` retain their
meaning: the initial valuation must explicitly cover the exact registered part
references. No inventory values are filled from an expected outcome.

Owners supply `capture_recovery_safety_state`, `prepare_recovery_safety_program`,
`recovery_safety_configuration`, `get_recovery_safety_primitive_model`,
`validate_recovery_safety_step`, and optionally `get_recovery_entity_reader`.
The reader returns exact entity identity, frame, pose, simulation stamp and time,
and monotonic receipt time. Preparation retains the requested program, checkpoint
and program fingerprints, exact parameters, and event/step provenance. Validation
binds each checked step to that owner's exact saved preparation. No shared check
assumes that prepared commands contain `joint_trajectory`.

## Supported owner models

`PrimitiveModel(id, version, configuration, evaluate)` is a trusted pure model;
its callback stays outside JSON. Optional keyword-only `primitive_models` binds
these objects to exact resources in observation, reviewed-checking, automatic
grounding, and composition calls. The callback validates native actions and
parameters and returns `trajectory`, `base_trajectory`,
`part_trajectories`, `transfers`, and `resource_updates`. Version 2 also permits the
optional `continuous_motion` record, bound to the exact owner configuration and
prepared joint trajectory. Version 3 additionally permits explicit `custody_effects`:
grasp/release at a prepared step's completion boundary, with matching part pose,
grasp transform, containment and deposited stationary coverage. Carried parts
require configured `local_bounds`; their motion follows bounded tool FK and their
grasp transform. The continuous custody clock is `continuous_physical_boundaries_v2`.
These supplied effects do not establish a live attachment operation or a product
completion. The owner cannot return AP values,
rules, process-completion records, or an admission verdict. `None` body motion and
empty effect collections explicitly assert no such effect for that interval.
Unsupported native actions must be rejected by the owner model.

Resource-body paths and transported-part paths are separate. Supported paths have
complete fixed-orientation piecewise-linear coverage, including intermediate
waypoints. Stationary coverage must fill every remaining interval. Joints and
grasp transforms are required only for contracts that use them. Containment
transfers occur at a joint completion boundary and require identical part/source/
destination records from both participants, consistent inventories, and unique
custody. Containment never becomes gripper custody; a contained transported part
has its own region-entry evidence. The existing resource-occupancy AP includes
the resource's declared body/envelope and carried parts, with unchanged meanings.

The new owner path permits `stationary_only` equipment to work while its body
stays fixed. Untagged 1D and KMR evidence keep their existing contracts, including
legacy `stationary_only` behavior. Historical tagged UR5e evidence resolves
through controller-owned compatibility registration. Unknown tags, missing
owner models, and incompatible contracts remain unavailable.

Grounded owner state changes and process effects require exact configured task
associations, supported literal-bound programs, resource guards/updates and
completion evidence. Ordered process requirements come from the trusted model's
frozen `configuration.requirements`, using the existing requirement records.
State evidence must agree with physical observations. `M1` `machine_part` uses
its declared five-second `dwell`; `dwell` alone leaves `processCompleted` unchanged.
Its trim completion can satisfy a pending `F` requirement through the same native
and physical composition witness without requiring assembly `state`/`target`
fields. Native completion conditions require explicit owner state support;
legacy checks retain their previous completion-field restrictions.
`Conveyor` `advance_conveyor` / `move_relative` and `Buffer For Machined parts`
`advance_part` / `move_relative` use their native speed parameters. Other effects
or unresolved program bindings require additional supported evidence.

Model identity, version, configuration, resulting effects, and provenance enter
trace/composition fingerprints. Configuration cannot change between steps or
during evaluation. Incompatible continuations are rejected. Motion, transfers,
native updates, and acknowledged product effects share the existing observation
clock. Predicted effects stay in detached analysis; task and physical histories
remain distinct and still require one common completion witness.

All added equipment and additional-resource providers are synthetic test code in
[test_resource_safety_preparation.py](../test/test_resource_safety_preparation.py).
The seven-joint mock also reaches `Prepare and check` without a robot controller.
These examples establish extension through owner contracts, not live support for
those resources. The earlier UR5e preparation remains the available controller
implementation; the former standalone preparation module has been removed.

## Conditional checking and remaining live gaps

The deterministic 1D and fixed-orientation piecewise-linear paths keep their
existing meanings and clocks. Historical UR5e Cartesian evidence still requires
separately supplied complete coverage; a joint plan never becomes a straight TCP
segment. `cartesian_coverage_fingerprint` retains its primitive-relative timing.

The new continuous path accepts the controller's observed `splines` contract.
It preserves positions, velocities, accelerations and nanosecond times and uses
linear, cubic or quintic interpolation according to the supplied derivatives.
Mixed derivative availability and unsupported interpolation are unavailable.
Polynomial bounds use rational Bernstein coefficients. Outward-rounded interval
FK encloses configured link envelopes, including interior motion that endpoint
or point sampling would miss. URDF origins, collision meshes, stationary component
bounds, root pose, joint identities and frozen non-commanded joint positions are
part of the recorded contract. This is a check of those declared envelopes and
interpolation, not a claim of physical tracking or mesh-contact accuracy.

Physical occupancy has definite or possible values. Each rule retains its own
`ap001` / `ap002` bindings. `ap_evidence.value` is Boolean when definite and `null`
when unresolved; `possible_values` preserves the alternatives. Bounds refine up
to four subdivision levels within the same analysis budget. Exhausted refinement
or numerical uncertainty never becomes a passing result by sampling.

`continuous_physical_boundaries_v1` retains initial, primitive and supplied task
boundaries. An open interval denotes possible finite AP words, including unknown
physical contact order. Numerical subdivision points are certificate boundaries,
not new semantic observations. The composition propagates all resulting DFA states
and unsafe uncontrollable alternatives through the same search. Boolean/G/F/U
formulas are supported; `X` remains unavailable. Acceptance requires a shared
completion witness for every retained physical alternative and the detached native
monitors. A definite conflict is `held`; unresolved uncertainty or pending behavior
outside the supplied continuation is `inconclusive`. No terminal empty observation
is appended. Accepted prefixes replay only in the same frozen problem; histories
from a different geometry, contract, formula or clock cannot supply DFA states.

The conditional soundness argument requires every actual path described by the
owner contract to lie within its bounds. Rational polynomial bounds enclose each
supported interpolation segment, interval FK encloses the declared component
geometry, and each possible physical valuation is retained in the interval's AP
alphabet. DFA closure over possible words overapproximates the unknown contact
sequence without treating numerical samples as clock ticks. Because the supported
fragment excludes `X`, numerical subdivision cannot authorize a continuation by
inserting next-time observations. The same search retains unsafe uncontrollable
successors and accepts only when every retained monitor state can complete safely.
Inaccurate geometry, missing effects or an unsupported predicate invalidate these
premises; registering a provider establishes none of them by itself.

`LiveSafetyPreparation` is registered as CCA's
`recovery_safety_preparation_provider` from trusted scene configuration. It reads
all configured owners, validates an idle checkpoint and the unchanged initialization
ledger, assembles the exact prepared programs and explicit finite start offsets,
and provides stationary contracts for waits and completed candidates. A snapshot
alone cannot establish future stationarity. Existing running work, unknown previous
physical history or unprovided acknowledgement replay remains unavailable. No
predicted process completion is committed. Reobservation checks pose, orientation,
joints, geometry, attachments, run/program identities and actual histories. This
slice has no observation-error envelope: even a small unmodeled state change
requires preparation again. Controller status must come from the current publisher;
an absent initial action-status message is not an empty goal set.

## Supplied candidates and the dedicated test

[The candidate input](../test/fixtures/gazebo_safety_preparation/candidates.json)
is explicitly mock. `GAZEBO_MOTION_SAFE` supplies `ur5e-4` approach, entry and retreat
using native `move_cartesian` parameters. `GAZEBO_MOTION_CONFLICT` also supplies
`ur5e-3`, with explicit simultaneous start offsets. Names describe test intentions,
not certified outcomes. **Load supplied candidate** resolves the retreat pose from
observations; **Prepare and check** retains the existing request format. Required
idle/geometry/history evidence is checked before planning. Every report retains
`dispatch_authorized=False`; the selected Safety-page file and approvals are not
changed by loading the candidate.

For a fresh, dedicated test simulation, build the installed observation service
and launch without dispatching manufacturing work:

```bash
make bootstrap-gazebo
source /opt/ros/humble/setup.bash
source /home/jongh/ros2_ws/install/setup.bash
ros2 launch cais_lab_robotics recovery_framework_gazebo.launch.py launch_gazebo_gui:=false launch_rviz:=false
```

In another shell with the same ROS setup, explicitly select the supplied safety
input for the test runner. Use a new output directory each time:

```bash
poetry run python scripts/check_gazebo_safety_preparation.py \
  --predefined-safety cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt \
  --output-directory /tmp/recovery-safety-test
```

[The runner](../scripts/check_gazebo_safety_preparation.py) creates registered owners
and a CCA without starting SPADE dispatch behaviors. It tests the same adapter,
retains both candidates, and invalidates one required owner declaration for the
negative check when inputs could be resolved. It exits successfully only for the
required `allowed` / `held` / `NEEDS_CONTEXT` outcomes. It is a diagnostic test;
its new context cannot replace history from an existing production run.

On **2026-10-05**, **620 focused tests passed in 174.27s**. The ROS build completed
16 packages. A dedicated session returned a stamped snapshot of 36 models and
zero attachments. Both candidate attempts captured all 12 resources, but returned
`NEEDS_CONTEXT`: observed UR5e link motion, missing controller idle-status records,
and missing finite collision geometry for the empty `assembly_board-v1` carrier.
The missing-owner negative check also returned `NEEDS_CONTEXT`. A later retry timed
out reading the service. The [evidence index](gazebo_safety_preparation_2026-10-05.json)
records canonical report paths and hashes, including that retry.

**Live acceptance is incomplete.** No candidate reached planning plus a meaningful
live mutex result. In particular, the candidate names do not establish a safe
moving continuation or a definite conflict. No primitives, grants, attachment
mutations, entity repositioning, failure injection or LLM calls were dispatched by
the checking path. Precedence's before/after/simultaneous-completion regressions
remain; a motion-only trace without `KET4_Square_4mm` entry would check only the
no-entry case, not a non-vacuous precedence experiment.

Guarded execution, observed feedback and recovery/resumption remain later gates.
The desired slippage experiment still has `ur5e-4` dropping `gear_small` into
`ur5e-3`'s region and `KET4_Square_4mm` initially in `Buffer For Machined parts`;
this work neither creates nor claims that failure checkpoint. Preserve the legacy
`receiving_region_entry` mismatch and **20,000 states / 2 seconds** defaults.

The final affected suites passed **125 tests in 16.53s**, overlapping the 620-test
run. Compilation, CLI help, focused Ruff, `poetry check` (existing metadata
warnings), JSON/report fingerprints, local links and `git diff --check` passed.
The 57 protected existing files and 26 roadmap headings were preserved.

The subsequent **2026-10-05 part_slippage preflight** adds explicit
`safety_preparation.part_geometry` declarations for static constituent models.
`assembly_board-v1` uses the observed collision bounds of its configured
`GMC_Laser_Plate_Virtual` and `Gear_Plate` fixtures. The carrier and every
constituent must be static in the same stamped physics snapshot. Empty, absent,
moving, or changed constituents remain unavailable. The configuration, observation
identities, model fingerprints and resulting envelope are retained and rechecked.
No carrier dimensions or fixture membership are inferred from the safety wording.

Fixed resources now accept their configured `static_body_and_idle_containment`
preparation contract only with observed static geometry and idle evidence.
They need no joints or grippers. This conditional preparation coverage still
retains `future_execution_tracking: not_established`; it grants no execution.

The [three Gazebo companions](../test/fixtures/part_slippage/gazebo/README.md)
retain the original seven events and 19 primitive references per case.
[check_part_slippage_gazebo.py](../scripts/check_part_slippage_gazebo.py) captures
their live prerequisites with the unchanged predefined document. Its reports
explicitly separate a missing-evidence result from a safety rejection, and retain
`dispatch_authorized=False`. It does not resolve unsupported native programs,
stage parts, start SPADE dispatch, issue grants, or record unaccepted trials.

The [live report](../cais_spade_llm/monitor/recovery_gazebo_runs/part-slippage-20261006T021725Z/REPORT.md)
records 12 resource observations and 12 part envelopes for each final capture;
the board-envelope gap was resolved. All three attempts remain `NEEDS_CONTEXT`
because of idle evidence and missing live admission providers. The first final
capture also exceeded its freshness window. No requested checkpoint was staged
and no mutex/precedence counterexample or safe recovery was demonstrated. Native
grasp/release effects, continuous carried-part evidence, prepared execution and
validated feedback remain required. No requested MP4 was published.

The follow-up connection work adds an opt-in
`cais_lab_robotics/RecoveryJointTrajectoryController` with a read-only
`~/recovery_state` service. It calls the existing controller implementation and
reports controller incarnation, command revision, simulation timestamp, holding
state, active/pending goals and exact joints. The launch argument
`recovery_observations:=true` enables this controller for the dedicated test;
normal controller types remain the default. `/KMR/recovery_state` reports its
owned base activity. The diagnostic launch used `launch_nav2:=false`; base-owner
observations cannot substitute for independent navigation owners when Nav2 is enabled.

`GazeboSafetyReader` accepts explicit `controller_state_services` declarations
whose `covers` lists must cover every configured action-status endpoint exactly
once. Missing, stale, active, non-holding or incompatible observations remain
unavailable. This owner contract resolves startup cases where an action server has
not yet published a status message. It establishes present idle activity, not a
guarantee that future geometry stays constant. The dedicated read collected all
11 controller rows for 17 endpoints within 0.234 seconds of its physics snapshot.
Cold discovery first failed with the ordinary two-second timeout; a diagnostic
15-second service timeout permitted discovery without relaxing observation freshness.

CCA now passes trusted owner `primitive_models` to composition and fingerprints
their descriptors. Serialized context cannot supply callbacks. Changed contracts
invalidate a registered proof. `RobotAgent` delegates a registered macro through
the common grant-consuming path only when an owner execution provider and its
initialized controller both support exact prepared execution. That live controller
hook remains unavailable, so the new path still refuses these live starts.
Measured continuous-history replay, execution tracking, native grasp/release
preparation and validated custody feedback remain unfinished.

The requested recording is **one silent, captioned 20× MP4** containing the three
separately staged trials. Both predefined specifications stay active in each
trial's composition. `export_combined_20x` joins complete validated captures,
preserves distinct run identities and captions, and uses the existing exporter
for full decoding and 20× duration validation. Trial sources are removed only
after that succeeds; metadata stays with the capture evidence. Its synthetic
encoding test is not recovery footage. No trial video has been published.
