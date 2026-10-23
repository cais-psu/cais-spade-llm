"""Check an offline composition against an observed, owned Gazebo failure."""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from pathlib import Path

from cais_spade_llm.recovery_framework.diagnostics import write_record
from cais_spade_llm.recovery_framework.recovery_observations import (
    copy_generation_records,
    fingerprint,
    observation_fingerprint,
)


async def validate_saved_proposal(recovery, proposal: dict, observations: dict, *, source_state: dict | None = None) -> None:
    """Enable execution only after state matching and controller feasibility checks.

    Args:
        recovery: The ProductAgent-owned simulation recovery session.
        proposal: Completed offline composition; never modified in place.
        observations: Immutable source facts used to generate the composition.
        source_state: Original stage paths when first importing a saved result.
    """
    from cais_spade_llm.recovery_framework.part_slippage_recovery import diagnostic_setup

    if proposal.get("validation_contract_version") == 2:
        raise ValueError("Outline validation does not authorize execution; runtime CCA integration is pending")
    runtime, context = recovery.runtime, recovery.runtime.context
    if recovery.offline or not diagnostic_setup(runtime.setup) or not runtime.diagnostic_cca_bypass:
        raise ValueError("Saved recovery execution requires the selected simulation run")
    if context.pending_tasks or context.reservations or runtime.active_executions:
        raise ValueError("Interrupted work must release execution ownership first")
    if recovery._stopped:
        raise ValueError("Reset the stopped example before testing recovery")
    if source_state is not None:
        directory = recovery._new_artifact_directory()
        paths = copy_generation_records(source_state, Path(source_state["observation_path"]).parent, directory)
        recovery._outline_request = {}
        recovery._publish(**paths, imported_recovery=True,
                          outline=source_state.get("outline", []), primitives=source_state.get("primitives", []))
    recovery._proposal = {}
    revision = context.revision
    recovery._publish(status="validating_saved_recovery", message="Checking saved composition against fresh Gazebo observations.")
    report = {"status": "blocked", "safety_status": "skipped", "execution_authorized": False,
              "checked_at_unix": time.time(),
              "source_observation_fingerprint": observation_fingerprint(observations), "checks": []}
    try:
        owners = await _check_source(runtime, proposal, observations)
        saved_nodes = observations.get("pending_nodes")
        if saved_nodes is None:
            raise ValueError("NEEDS_CONTEXT: saved interrupted task definitions are missing")
        fields = ("function_name", "resource_jid", "params", "status", "sequence_index")
        if ([{key: row.get(key) for key in fields} for row in saved_nodes]
                != [{key: row.get(key) for key in fields} for row in recovery._pending_nodes()]):
            raise ValueError("Unfinished task state changed; regenerate from the current observations")
        snapshots = await _resource_snapshots(runtime, owners, proposal, observations)
        await _check_steps(owners, snapshots, proposal, observations, report)
        if recovery._stopped:
            raise asyncio.CancelledError()
        if context.revision != revision:
            raise ValueError("Execution context changed during validation; repeat Test in Gazebo")
        tasks = proposal["macro_tasks"]
        report.update(status="passed", execution_authorized=True)
        recovery._imported_proposal = deepcopy(proposal)
        recovery._imported_observations = deepcopy(observations)
        recovery._proposal = {**deepcopy(proposal), "offline_recovery": False}
        recovery._context_revision = context.revision
        write_record(Path(recovery._state["artifact_directory"]) / "proposal.json", recovery._proposal)
        recovery._publish(status="ready_to_execute", outline_status="completed", primitive_status="completed",
                          outline=recovery._state.get("outline") or tasks,
                          primitives=recovery._state.get("primitives") or tasks,
                          message="Saved composition checked against Gazebo. Review, then Execute recovery.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report["reason"] = str(exc)
        recovery._publish(status="blocked", message=str(exc))
        raise
    finally:
        directory = Path(recovery._state["artifact_directory"])
        path = directory / "gazebo_validation.json"
        if path.is_file():
            path = directory / f"gazebo_validation_{len(list(directory.glob('gazebo_validation*.json'))) + 1:04d}.json"
        write_record(path, report)


async def _check_source(runtime, proposal: dict, observations: dict) -> dict:
    from cais_spade_llm.recovery_framework.failure_effects import _observe_part

    context = runtime.context
    if proposal.get("source_observation_fingerprint") != observation_fingerprint(observations):
        raise ValueError("Saved composition does not match its source observations")
    if observations.get("jids") != runtime.jids:
        raise ValueError("Resource identities changed; regenerate from the current observations")
    if (observations.get("program_revision") != runtime.program_revision
            or fingerprint(observations.get("inputs")) != fingerprint(context.inputs)):
        raise ValueError("Configuration or resource programs changed; regenerate recovery")
    for rid, saved in observations.get("resource_states", {}).items():
        actor = context.resources.get(rid)
        if actor is None or actor.valuation != saved:
            raise ValueError(f"{rid} resource state changed; regenerate from the current observations")
    owners = {str(owner.jid): owner for owner in runtime.resource_agents}
    for part, saved in observations["part_tracker"].items():
        current = context.part_tracker.get(part, {})
        for field in ("location", "current_process", "processCompleted"):
            if current.get(field) != saved.get(field):
                raise ValueError(f"{part}.{field} changed; regenerate from the current observations")
        if saved.get("observed_pose"):
            controller = next((owner._controller for owner in owners.values()
                               if getattr(owner, "execution_mode", None) == "simulation"
                               and getattr(getattr(owner, "_controller", None), "execution_mode", None) == "simulation"), None)
            if controller is None:
                raise ValueError("No simulation controller can observe the saved part")
            pose = await asyncio.to_thread(_observe_part, controller, context.geometry[part]["model_name"])
            if any(pose.get(key) != value for key, value in saved["observed_pose"].items()):
                raise ValueError(f"{part}.observed_pose changed; regenerate from the current observations")
    return owners


async def _resource_snapshots(runtime, owners: dict, proposal: dict, observations: dict) -> dict:
    context = runtime.context
    saved_resources = observations["prepared_recovery_request"]["recovery_resources"]
    snapshots = {}
    resource_jids = {task["resource_jid"] for task in proposal.get("macro_tasks", [])}
    resource_jids.update(jid for jid, entry in saved_resources.items() if entry.get("resource_type") == "robot")
    for jid in sorted(resource_jids):
        owner = owners.get(jid)
        if (owner is None or owner.agent_name not in context.permitted_resources
                or owner.execution_mode != "simulation"
                or getattr(getattr(owner, "_controller", None), "execution_mode", None) != "simulation"):
            raise ValueError("Saved composition has no owning simulation controller")
        requested_at = time.time()
        snapshot = await asyncio.to_thread(owner.get_recovery_snapshot)
        if snapshot.get("controller_ready") is not True:
            raise ValueError(f"NEEDS_CONTEXT: {jid} controller is not ready")
        saved = saved_resources[jid]["recovery_snapshot"]
        if saved.get("current_pose"):
            observed_at = snapshot.get("current_pose_captured_at")
            if not isinstance(observed_at, (int, float)) or not requested_at <= observed_at <= time.time():
                raise ValueError(f"NEEDS_CONTEXT: {jid} has no fresh pose observation")
        for field in ("current_state", "held_part", "gripper_state", "current_location", "current_pose"):
            if field in saved and snapshot.get(field) != saved[field]:
                raise ValueError(f"{jid}.{field} changed; regenerate from the current observations")
        snapshots[jid] = snapshot
    return snapshots


async def _check_steps(owners: dict, snapshots: dict, proposal: dict, observations: dict, report: dict) -> None:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_primitives import (
        snapshot_matches_expected,
        validate_and_project_steps_with_trace,
    )

    saved_resources = observations["prepared_recovery_request"]["recovery_resources"]
    tasks = proposal.get("macro_tasks") or []
    if not tasks:
        raise ValueError("Saved composition contains no macro_tasks")
    completed = set()
    for task in tasks:
        if task["outline_id"] in completed or not set(task.get("predecessors", [])).issubset(completed):
            raise ValueError("Saved composition has inconsistent task ordering")
        jid, part = task["resource_jid"], task.get("part_name")
        owner, snapshot = owners[jid], snapshots[jid]
        matches, reason = snapshot_matches_expected(snapshot, task.get("expected_snapshot") or {})
        if not matches:
            raise ValueError(f"Saved composition start state changed: {reason}")
        catalog = await asyncio.to_thread(owner.recovery_execution_primitive_catalog)
        if fingerprint(catalog) != fingerprint(saved_resources[jid]["execution_primitive_catalog"]):
            raise ValueError("Primitive definitions changed; regenerate recovery")
        grounding = observations["prepared_recovery_request"].get("grounding_context", {})
        trace = validate_and_project_steps_with_trace(task["primitive_steps"], catalog, snapshot,
                                                    grounding_context=grounding)
        if not trace["valid"]:
            raise ValueError(f"NEEDS_CONTEXT: {trace['validation_error']}")
        for step in trace["step_results"]:
            result = await asyncio.to_thread(
                owner.check_recovery_primitive_feasibility, primitive=step["primitive"],
                params=step["resolved_params"], task=task, recovery_snapshot=step["start_snapshot"],
                part_context=grounding.get("parts", {}).get(part, {}),
                grounded_action={"part_name": part, "target": task.get("target_ref") or {}},
            )
            report["checks"].append({"outline_id": task["outline_id"], "primitive": step["primitive"], "result": result})
            if result.get("allowed") is not True:
                raise ValueError(result.get("reason") or "NEEDS_CONTEXT: controller feasibility is unavailable")
        snapshots[jid] = trace["projected_snapshot"]
        completed.add(task["outline_id"])
