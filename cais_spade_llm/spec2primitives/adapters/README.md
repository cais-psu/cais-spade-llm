# Adapters

`dual_gazebo.py` manages the no-hardware `gazebo_dual_spec2primitives` process through the narrow supplied runtime protocol. It does not import `SystemBridge`; the shared public surface remains unchanged.

`ui_runtime.py` composes the PA grounding runtime, configured document/observation VLMs, approved calibration and request-owned capture. PA controls evidence investigation. Real sensor metadata remains internal while randomized observation projections cross model boundaries.

Current Phase 4 requires configured capability and live MoveIt position plans for all grounded locations. Each reachability request carries the configured robot planning profile. `in_process_robot_agent.py` validates the request and Gazebo readiness, then calls `MoveItPlanOnlyRuntime.validate_state_locations` through the injected runtime, without starting or contacting a RobotAgent. `GetMotionPlan` uses position-only goals and never executes a trajectory. There is no workspace-box or fixed-radius fallback.

`read_resource_base_pose` reads live TF for the base frames in `config/resource_base_frames.json`, expressed in the grounded target frame. PA's allocation tools use it for advisory distance measurements. This read does not activate a RobotAgent or alter MoveIt eligibility; unavailable TF supplies no preference. The adapter does not synchronize the collision scene or validate grasping/lifting.

The active RA adapter performs exact selected-RA context capture and direct primitive-program authoring. It can reuse/start only that context-only RobotAgent under existing simulation readiness gates. Current completion and evidence lineage are required before activation, recovered context or composition. The isolated RA call receives the reconstructed context and returns a read-only evidence request or program proposal with available parameters. Shared agents remain read-only. The owned refinement adapters add measured context, numerical calculation and private scene validation; no robot execution is added.

See [PA](../agents/pa/README.md), [RA](../agents/ra/README.md), and [bias validation](../BIAS_VALIDATION.md).

Fresh selected-RA state capture adds `motion_context` directly from `controller_config.move_group`: exact `frame_id`, `ee_link` and `tcp_link`, with unavailable values left null. It does not initialize the controller, measure transforms or infer execution identifiers. The full runtime snapshot remains captured; a separate composition projection removes recovery metadata before initial input and subsequent evidence reads.

Nested parameter declarations survive the shared catalog analyzer and owned conversion. `robot_validation_context.py`, `target_calculation.py` and `isolated_moveit.py` implement measured feedback, strict numerical evaluation and a private worker with execution disabled. The future identifier adapter remains separate. See [VALIDATION_AND_REVISION.md](../VALIDATION_AND_REVISION.md).

The owned composition projection omits `model_name` from new program interfaces while retaining the original backend signatures in snapshots. A future execution adapter must bind the already-recognized physical instance to its simulator identifier and supply it to those backend calls. The current adapter performs no identifier lookup, inference from `part_name`, or robot execution.
