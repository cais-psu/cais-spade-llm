# Phase 5 validation and revision

Composition produces a **validated program proposal within a recorded rigid vertical geometry and motion model**. The separate **Run in Gazebo** action can execute that exact program in simulation after fresh checks. Composition itself executes no robot commands, and neither a modeled pass nor completed commands establish assembly success. The original Phase 4 completion and saved programs remain immutable. See [implementation status](IMPLEMENTATION_PLAN.md) and [research evaluation](COMPOSITION_EVALUATION.md).

## One composition process

```text
Compose Primitive Program
→ RA authors and immediately displays a candidate
→ inspect its selected dependencies and missing inputs
→ capture measured robot context through the selected RA adapter
→ calculate selected helpers and validate the ordered program
→ findings to RA, which requests context, revises its program, or reports unsupported capability
→ PA investigates missing product inputs and RA-authored supplemental requests
→ RA explicitly binds supplemental evidence or changes its program; repeat checks, or stop
```

`agents/ra/refinement.py` owns this flow. `primitive_composition.py` records each exact RA submission, including rejected responses; a validator never inserts, removes or repairs steps. New independent runs start from completion and captured context. Only revisions inside the same pinned run receive that run's previous candidate, accepted supplemental evidence and findings. Historical drafts and programs do not become initial inputs.

The composition projection retains partial formal contracts: grasp/release expose only `held_part`; `model_name` is absent. Primitive descriptions and helper outputs supply domain knowledge without prescribing an execution order. The shared runtime signatures, robot actions and recovery sequences retain their existing behavior.

## Evidence and dependencies

`program_dependencies.py` follows RA-selected `value_ref` and `result_ref` dependencies. It reports missing fields, incompatible bindings, deferred outputs and downstream dependencies without changing the program. Required nested fields are enumerated even when their containing geometry object is absent. The composition contracts include the strict calculators' placement requirements, including `target_reference.target_point` and `target_origin_pose.x/y/z`; captured runtime signatures and controller fallbacks remain unchanged.

Missing product measurements in the selected `product_geometry`, `target_pose` and `detected_parts` inputs automatically reach PA with their exact step, parameter path, quantity schema, reason and selected evidence references. Missing robot/control inputs and incompatible or unverified bindings return to RA for correction. RA may add `request_context` entries for supplemental facts using `step_index`, `quantity`, `authority` and `reason`. PA entries retain their wording and order, with the first request for each exact step/quantity kept when duplicated. A request for the same step, primitive, selected parameters and quantity is not repeated with unchanged product evidence, including evidence just returned by that investigation. Changed RA bindings can therefore request measurements against the newly selected inputs. The host supplies no replacement binding or task sequence.

`agents/pa/primitive_context.py` adds a bounded investigation after completion. It reuses approved retrieval/document/CAD/RGB-D/calibration producers. Supplemental native retrieval validates typed records and writes separate audits; it never merges ontology deltas, replaces ProductContextView, or repeats resource allocation. PA chooses its evidence tools and feature associations for the requested facts. Returned records and unresolved reasons go back to RA for its next decision. The validator's `part`, `goal`, `scene` and `specification` requirements remain separate acceptance checks for its declared scope; they do not prescribe an extraction sequence. RA can request the supplemental evidence needed for those checks. Conflicting identity or changed goal requires the existing PA authority gate.

PA's evidence catalog includes hash-verified discovery metadata for already issued records, including partial evidence from earlier batches. PA reviews requests derived from RA-selected contracts and RA-authored supplemental requests, reuses applicable records, and may investigate other needs when one measurement is blocked. Its unresolved feedback retains requested quantities and step references and distinguishes uninvestigated requests from attempted measurements that remain ambiguous or unavailable. PA still chooses tools, their order, and when to finish; neither unused operations nor an ambiguous measurement triggers an automatic retry.

`tools/assembly_geometry.py` measures neutral CAD planar/circular features, oriented dimensions and observed support-plane candidates. PA binds selected measured instances/features to the exact accepted assembly relationship. Final part origin is derived from selected opposing seating planes and mating axes, independently of RA's candidate. The visible shaft centre alone is insufficient. An ambiguous pose does not establish an orientation or final pose. For the same observed candidate and approved calibration, `inspect_features` may return `part_height_m` as the world-vertical CAD extent from the already highest-ranked qualified pose hypothesis. This estimate uses no height-agreement threshold or averaging. Its source, registration metadata and warning remain recorded; the record stays ambiguous. Only the selected height scalar may use this partial record, never a full pose, support height or complete `part`/`goal`/`scene` evidence. The estimate does not establish sensor accuracy, and a raw CAD-local Z dimension is still insufficient.

Records retain units, coordinate frame, semantic reference point, observation time, uncertainty and source hashes. Derived source chains and mesh bytes are rechecked. Scene coverage requires every candidate in the selected segmentation records to have registered geometry, plus measured support planes. Missing coverage stays unknown. Unobserved space and contact mechanics are explicitly outside this model; cross-camera identity fusion is not invented.

Missing calculator inputs remain distinct from physical-validation requirements: scene coverage supports collision checks, while the specification supplies acceptance criteria. Live and restored UI feedback shows incomplete assembly validation and preserves all findings and PA reasons in the trace. A failed pure calculation leaves independent subsequent calculations available; an unvalidated movement or custody transition still blocks later state-dependent checks. Missing evidence cannot pass validation or authorize execution.

These repairs support composition from available information within the existing primitive and validation scope. They add no USB cable, rgb cable or hex nuts controllers or evaluators. Broader composition experiments require the corresponding available capabilities; evaluate composition changes separately from measurement availability, with equal evidence and tools across comparison methods.

Assembly tolerances come from exact unit-bearing approved document spans selected by PA, or an explicit experiment specification. To supply the latter, set `validation_specification_path` in [the profile](config/phase5_validation.json) to a JSON file under `cases/`. It must be an accepted `AssemblyValidationSpecification` with `family: vertical_gear_assembly`, positive `position_tolerance_m` and `axis_tolerance_rad`, and an explicit `yaw_required` policy; yaw-required cases also need `orientation_tolerance_rad`. Declare `requires_threading` and `requires_force_control` honestly. The host snapshots and hashes the selected specification. No default assembly tolerances are supplied by RA or the calculator.

## Measured robot context and numerical calculation

`adapters/in_process_robot_agent.py` retains the exact-resource/lifecycle/readiness gate. `robot_validation_context.py` reads joint states, EE and TCP poses, their full relative transform, controller policy, tool links and URDF/SRDF/planning parameters through read-only ROS interfaces. Owned ROS contexts avoid changing another runtime's executor. Configuration names are not measured feedback. Default sensor age and capture skew limits are two seconds; scene age is 120 seconds, using ROS time for observations. Wall-clock capture age is checked separately.

The approved shared extraction is `resources/robot/target_calculations.py`, used by the existing controller and the strict Spec2Primitives calculation adapter. The adapter evaluates only helper calls selected by RA, using resolved exact arguments and the predicted preceding EE pose. It creates no controller/action clients, performs no detection or simulator lookup, and rejects missing dimensions, unresolved destination tokens, incompatible frames and unsupported tool offsets instead of using runtime fallbacks.

Calculation records retain resolved arguments, actual extracted outputs, source hashes, robot policy/configuration hashes, preceding pose and cache key. Cache reuse requires matching inputs, sources and relevant robot context. The authored program remains separate from resolved values. `approach_pose`, `target_pose`, `pre_insert_pose`, and `insert_pose` remain distinct. Absent conditional outputs produce findings; reaching pre-insertion cannot satisfy a seated assembly requirement.

## Finite modeled validation

`agents/ra/program_validation.py` returns `passed`, `failed`, or `unknown` findings. Coverage is `rigid_vertical_gear_assembly_direct_cartesian`. The supported evaluators handle selected pick/place calculations, `move_cartesian`, and grasp/release custody. A known catalog primitive without an evaluator remains an explicit unsupported finding.

| Check | What is established within this model |
| --- | --- |
| Bindings | Supplied types/references, required calculation fields, actual conditional results, selected measurement provenance, units and frames. |
| Declared conditions/effects | Supported custody and pose transitions; unknown conditions/effects cannot silently pass. No planner searches these contracts. |
| Grasp | Necessary TCP/part overlap and gripper-width envelope, followed by an explicit rigid part-to-tool assumption. This is not force closure, friction or gripper-contact validation. |
| Ordered motion | Complete collision-checked Cartesian segments from predicted prefix joints/pose, orientation, robot limits and carried mesh. |
| Release/support | The predicted carried part reaches the specified seating relationship before release. Support is a nominal geometric assumption, not a dynamics result. |
| Outcome | Final part-origin position and selected axis within independently supplied tolerances, required orientation where declared, and destination custody. |

`adapters/isolated_moveit.py` starts a private worker with execution disabled and execution capabilities excluded. Its planning scene is separate from the live scene; robot start states and hypothetical carried/released geometry are supplied explicitly. Shared collision-object and joint-state inputs cannot mutate this frozen scene. Each authored movement must complete as one Cartesian segment. Controller fallback detours receive no credit as RA intermediate movements; runtime controller behavior remains unchanged.

Missing geometry, unavailable services, timeouts or incomplete paths cannot pass. A failed planning search is not proof that no feasible path exists. After a pass, another measured robot capture and source/scene-age checks can invalidate the result. Checks do not monitor the world indefinitely: the verdict is tied to the recorded snapshot and validity interval, not future execution authorization.

Robot contexts are saved before comparison, including the capture that causes rejection. The comparison checks the existing configuration/model hashes, frame and tool fields, `held_part`, EE/TCP transform, and all captured joint names and positions under the unchanged thresholds. Timestamp changes and joint-name ordering alone are not differences. Comparison events pin both captures and retain every differing field, its previous/current values, and applicable numerical differences and thresholds. A mismatch between refinement decisions still stops as `stale`; a mismatch after a provisional pass still produces an `unknown` `final_freshness` finding. Live and restored diagnostics use the same recorded comparison message, with full differences in the trace; reopening verifies the compared captures against their saved pins. A reported comparison difference does not by itself establish physical robot motion.

The shared [dual_moveit_gazebo.launch.py](../../ros2/cais_lab_robotics/launch/dual_moveit_gazebo.launch.py) leaves the MoveIt executable's node name unset in its `Node` declaration. The executable retains `/move_group`, while its helper retains `/moveit_simple_controller_manager`. Setting `name='move_group'` applied a global node-name remap to both, creating competing `/move_group/list_parameters` and `/move_group/get_parameters` services. Captures could therefore read different parameter sets and report `model_parameters_sha256` changes while the robot stayed stationary. The repair separates those services; parameter contents, comparison fields and thresholds remain unchanged.

After this launch-file change, run `make bootstrap-gazebo`, verify the installed launch file, and restart Dual Gazebo through its existing UI controls before composing again. Missing product, scene or specification evidence remains a separate validation gap. See the [recorded rollout and verification limit](IMPLEMENTATION_PLAN.md#robot-context-parameter-service-collision-2026-09-08): the service fixture passed, but the live retry reached its deadline before a second capture could test comparison across decisions.

## Budgets, persistence and UI

Defaults in the pinned profile are three candidate versions, two PA batches, twelve PA evidence operations total, a five-minute run deadline, 32 steps per candidate, and the existing twelve model reads per authoring decision. Each PA batch receives at most six operations. Context-only RA decisions are also bounded by the candidate-plus-batch limit. Private-worker startup has a 30-second allowance; subsequent service/planning calls use 10/20-second limits; cancellation tears down the private worker before completing cleanup.

Runs stop on pass, repeated unchanged failures, unavailable/unsupported evidence, authority conflicts, stale context, exhausted budgets, cancellation or invalid submissions. Stop reasons and every first-pass/revised result remain recorded. A later Compose click starts a new run; no saved program is regenerated automatically.

`composition/refinement_runs/run_*/` contains the request/profile, events, measured contexts, PA requests/exchanges/evidence, candidate references, calculation/validation records and final result. Candidates retain their existing writer and per-request catalog. The UI shows the first proposal immediately, progress, the latest numbered program and short findings; earlier versions and traces expand. Worker checks remain outside the UI event loop, duplicate clicks join/exclude the active run, and reconnect reads the persisted progress.

## Subsequent work and acceptance limits

### Run in Gazebo

`spec2primitives_ui.py` → `PrimitiveExecutionRuntime.run` → fresh `validate_program` → `GazeboExecutionSession` → immutable execution records.

The button runs the displayed saved program, with no new LLM call. It requires `validated_for_declared_scope`, the exact selected context-only RobotAgent, current catalog/configuration/evidence, the isolated Gazebo environment, and stopped hardware stacks. One application execution is allowed, with a filesystem lock excluding another application. Preparation revalidates the unchanged steps and retains complete timed Cartesian trajectories; the separate action client executes those trajectories while the private validation worker remains unable to execute. Helper results retain the strict calculation semantics. Grasp/release require gripper feedback and explicit link-attacher acknowledgments; no recovery wrapper, movement detour or placement snap is used.

The execution-only CAD binding reads the configured installed Gazebo fixture/model definitions and verifies the accepted STL bytes. For example, the existing `gear_medium` model uses `Gear_Medium.STL`. Its SDF mesh scale and visual transform are applied before comparing the live instance pose with the accepted observed CAD pose. The matching thresholds are configured independently of assembly tolerances in [gazebo_execution.json](config/gazebo_execution.json): 10 mm position and 0.1 rad orientation. Exactly one match is required. Neither simulator names nor simulator state becomes recognition/composition evidence.

Execution artifacts append under `contexts/<interaction>/execution/run_*/`: the pinned request, preparation/validation, instance binding, step arguments/results, event chain, final result and optional final RGB-D capture. Reconnecting only reads these records. Stop cancels an active trajectory and waits for a terminal result; unavailable acknowledgments remain unknown. Unknown command/custody outcomes block later execution for both robots. Later context reads project only acknowledged `held_part` and `gripper_state`. A program whose commands were dispatched cannot be replayed; capture current context and compose a new program.

Read [the coordinator](agents/ra/program_execution.py), [the transport and CAD binding](adapters/gazebo_execution.py), [the read-only execution state](agents/ra/execution_state.py), and [the focused execution tests](tests/test_primitive_execution.py). UI regression coverage is in [test_pa_ui_connection.py](tests/test_pa_ui_connection.py). Existing saved programs and validation reports are preserved; old reports need fresh validation to obtain executable trajectory timing.

Hardware execution, force/contact validation, independently observed assembly success, continuous scene monitoring, broader geometry and threaded hex nuts remain subsequent work. Final RGB-D is captured for inspection without an automated success claim.

Controlled fixtures establish the dataflow, numerical parity and failure gates; they do not establish live LLM accuracy. A live no-motion pilot must report missing evidence and unavailable services as observed, separately from fixtures. Publication claims require the held-out comparisons, independent evaluation and outcome measurements in [COMPOSITION_EVALUATION.md](COMPOSITION_EVALUATION.md).
