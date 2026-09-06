# Adapters

`dual_gazebo.py` manages the no-hardware `gazebo_dual_spec2primitives` process through the narrow supplied runtime protocol. It does not import `SystemBridge`; the shared public surface remains unchanged.

`ui_runtime.py` composes the PA grounding runtime, configured document/observation VLMs, approved calibration and request-owned capture. PA controls evidence investigation. Real sensor metadata remains internal while randomized observation projections cross model boundaries.

Current Phase 4 requires configured capability and live MoveIt position plans for all grounded locations. Each reachability request carries the configured robot planning profile. `in_process_robot_agent.py` validates the request and Gazebo readiness, then calls `MoveItPlanOnlyRuntime.validate_state_locations` through the injected runtime, without starting or contacting a RobotAgent. `GetMotionPlan` uses position-only goals and never executes a trajectory. There is no workspace-box or fixed-radius fallback.

The active RA adapter performs exact selected-RA Phase 5.1 context capture and Phase 5.2A unbound structural authoring. It can reuse/start only that context-only RobotAgent under existing simulation readiness gates. Current completion and evidence lineage are required before activation, recovered context or drafts. Shared agents remain read-only; no binding or execution is added.

See [PA](../agents/pa/README.md), [RA](../agents/ra/README.md), and [bias validation](../BIAS_VALIDATION.md).
