# Phase 5 validation and revision

The implemented milestone is a **validated program proposal within a recorded rigid vertical geometry and motion model**. It does not execute robots or establish physical assembly success. The original Phase 4 completion and saved programs remain immutable. See [implementation status](IMPLEMENTATION_PLAN.md) and [research evaluation](COMPOSITION_EVALUATION.md).

## One composition process

```text
Compose Primitive Program
→ RA authors and immediately displays a candidate
→ inspect its selected dependencies and missing inputs
→ product/scene needs to PA; measured robot context through the selected RA adapter
→ RA explicitly binds supplemental evidence or changes its program
→ calculate selected helpers and validate the ordered program
→ findings to RA for bounded revision, or stop
```

`agents/ra/refinement.py` owns this flow. `primitive_composition.py` records each exact RA submission, including rejected responses; a validator never inserts, removes or repairs steps. New independent runs start from completion and captured context. Only revisions inside the same pinned run receive that run's previous candidate, accepted supplemental evidence and findings. Historical drafts and programs do not become initial inputs.

The composition projection retains partial formal contracts: grasp/release expose only `held_part`; `model_name` is absent. Primitive descriptions and helper outputs supply domain knowledge without prescribing an execution order. The shared runtime signatures, robot actions and recovery sequences retain their existing behavior.

## Evidence and dependencies

`program_dependencies.py` follows RA-selected `value_ref` and `result_ref` dependencies. It distinguishes missing or incompatible evidence, deferred outputs, and downstream calculations blocked by upstream gaps. Context requests contain the affected step/parameter, required quantity/schema, reason, authority and existing references. A numeric intermediate control target is a proposal to check, not automatically a measured product fact. The host supplies no replacement binding.

`agents/pa/primitive_context.py` adds a bounded investigation after completion. It reuses approved retrieval/document/CAD/RGB-D/calibration producers. Supplemental native retrieval validates typed records and writes separate audits; it never merges ontology deltas, replaces ProductContextView, or repeats resource allocation. PA receives fact requests, not a sequence to repair. Conflicting identity or changed goal requires the existing PA authority gate.

`tools/assembly_geometry.py` measures neutral CAD planar/circular features, oriented dimensions and observed support-plane candidates. PA binds selected measured instances/features to the exact accepted assembly relationship. Final part origin is derived from selected opposing seating planes and mating axes, independently of RA's candidate. The visible shaft centre alone is insufficient. An ambiguous pose does not establish an orientation or final pose. Height may be reported separately only if every retained hypothesis for the same observed candidate agrees within recorded numerical precision; pose ambiguity and registration uncertainty remain visible.

Records retain units, coordinate frame, semantic reference point, observation time, uncertainty and source hashes. Derived source chains and mesh bytes are rechecked. Scene coverage requires every candidate in the selected segmentation records to have registered geometry, plus measured support planes. Missing coverage stays unknown. Unobserved space and contact mechanics are explicitly outside this model; cross-camera identity fusion is not invented.

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

## Budgets, persistence and UI

Defaults in the pinned profile are three candidate versions, two PA batches, twelve PA evidence operations total, a five-minute run deadline, 32 steps per candidate, and the existing twelve model reads per authoring decision. Each PA batch receives at most six operations. Context-only RA decisions are also bounded by the candidate-plus-batch limit. Private-worker startup has a 30-second allowance; subsequent service/planning calls use 10/20-second limits; cancellation tears down the private worker before completing cleanup.

Runs stop on pass, repeated unchanged failures, unavailable/unsupported evidence, authority conflicts, stale context, exhausted budgets, cancellation or invalid submissions. Stop reasons and every first-pass/revised result remain recorded. A later Compose click starts a new run; no saved program is regenerated automatically.

`composition/refinement_runs/run_*/` contains the request/profile, events, measured contexts, PA requests/exchanges/evidence, candidate references, calculation/validation records and final result. Candidates retain their existing writer and per-request catalog. The UI shows the first proposal immediately, progress, the latest numbered program and short findings; earlier versions and traces expand. Worker checks remain outside the UI event loop, duplicate clicks join/exclude the active run, and reconnect reads the persisted progress.

## Subsequent work and acceptance limits

Simulator-identifier resolution belongs to a future execution adapter for the already recognized physical instance. Do not guess it from `part_name` or feed it into recognition. Robot execution, force/contact validation, independently observed success, continuous scene monitoring, broader geometry and threaded hex nuts remain subsequent work.

Controlled fixtures establish the dataflow, numerical parity and failure gates; they do not establish live LLM accuracy. A live no-motion pilot must report missing evidence and unavailable services as observed, separately from fixtures. Publication claims require the held-out comparisons, independent evaluation and outcome measurements in [COMPOSITION_EVALUATION.md](COMPOSITION_EVALUATION.md).
