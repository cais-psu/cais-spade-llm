from __future__ import annotations

"""Explicit Gazebo evidence collection for a pending outline proposal."""

import asyncio
import inspect
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from cais_spade_llm.recovery_framework.diagnostics import write_record
from cais_spade_llm.recovery_framework.recovery_observations import (
    capture_observations,
    fingerprint,
    observation_fingerprint,
)


def checkpoint_differences(saved: dict, fresh: dict) -> list[str]:
    """Require matching recorded initial facts before reusing a pending proposal."""
    differences = []
    for field in ("jids", "inputs", "program_revision", "resource_states", "part_tracker", "pending_nodes",
                  "context_revision", "acknowledgements", "transitions", "monitor_history"):
        if field not in saved or field not in fresh:
            differences.append(f"Missing {field}; use new observations")
        elif saved[field] is None or fresh[field] is None:
            differences.append(f"Unavailable {field}; use new observations")
        elif fingerprint(saved[field]) != fingerprint(fresh[field]):
            differences.append(f"{field} changed; use new observations")
    old, current = saved.get("physical_evidence", {}), fresh.get("physical_evidence", {})
    if not old.get("snapshot") or not current.get("snapshot"):
        differences.append("Missing physical_evidence.snapshot; use new observations")
    for field in ("snapshot", "resource_models", "geometry"):
        if not old.get(field) or not current.get(field):
            differences.append(f"Missing physical_evidence.{field}; use new observations")
    old_resources, new_resources = old.get("resources", {}), current.get("resources", {})
    for rid, jid in saved.get("jids", {}).items():
        old_resource, new_resource = old_resources.get(rid), new_resources.get(rid)
        if not isinstance(old_resource, dict) or not isinstance(new_resource, dict):
            differences.append(f"Missing physical evidence for {rid}")
            continue
        for field in ("controller_identity", "frame", "attachment", "custody_complete"):
            if (field in old_resource) != (field in new_resource) or _facts(old_resource.get(field)) != _facts(new_resource.get(field)):
                differences.append(f"{rid}.{field} changed")
        entry = saved.get("prepared_recovery_request", {}).get("recovery_resources", {}).get(jid, {})
        is_robot = entry.get("resource_type") == "robot" or "held_part" in saved.get("resource_states", {}).get(rid, {})
        for field in (("joint_names", "joint_positions") if is_robot else ()):
            old_value = old_resource.get(field, old_resource.get("observation_state", {}).get(field))
            new_value = new_resource.get(field, new_resource.get("observation_state", {}).get(field))
            if not old_value or not new_value:
                differences.append(f"Missing {rid}.{field}")
            elif old_value != new_value:
                differences.append(f"{rid}.{field} changed")
    # Ignore acquisition clocks, but compare every state and configuration value.
    for field in ("snapshot", "resource_models", "geometry"):
        if _facts(old.get(field)) != _facts(current.get(field)):
            differences.append(f"physical_evidence.{field} changed")
    return list(dict.fromkeys(differences))


def _facts(value):
    if isinstance(value, dict):
        return {key: _facts(item) for key, item in value.items() if key not in {
            "captured_at", "observed_monotonic", "captured_monotonic", "simulation_time", "simulation_stamp",
            "checkpoint", "source_kind", "provenance",
        }}
    if isinstance(value, list):
        return [_facts(item) for item in value]
    return value


def pending_outline_event(recovery) -> dict:
    """Read the proposal with its retained, observation-derived start conditions."""
    session = recovery._prepared.get("multi_turn_session_state", {})
    pending = session.get("pending_outline_validation") or {}
    response = pending.get("response") or pending
    event = pending.get("task")
    if not isinstance(event, dict) and pending.get("kind") != "nominal":
        event = response.get("next_transition")
    if not isinstance(event, dict):
        raise ValueError("No pending LLM proposal is available for Gazebo evidence")
    if pending.get("kind") == "event" and isinstance(response.get("next_transition"), dict):
        if any(key not in event or event[key] != value for key, value in response["next_transition"].items()):
            raise ValueError("The pending validation task changes the retained LLM proposal")
    return deepcopy(event)


def _saved_planning_scene(observations: dict) -> dict | None:
    """Verify a saved scene's binding before explicit private model planning."""
    scene = observations.get("planning_scene_capture") or {}
    if scene.get("status") != "captured":
        return None
    physical = observations.get("physical_evidence") or {}
    recorded = physical.get("evidence_fingerprint")
    actual = fingerprint({key: value for key, value in physical.items() if key != "evidence_fingerprint"})
    if not recorded or recorded != actual or scene.get("physical_evidence_fingerprint") != recorded:
        raise ValueError("Saved planning scene is not bound to the unchanged physical capture")
    if not all(scene.get(key) for key in ("planning_scene", "start_state", "scene_fingerprint",
                                         "model_fingerprint", "robot_description", "robot_description_semantic")):
        raise ValueError("Saved planning scene or robot model is incomplete")
    return deepcopy(scene)


async def prepare_gazebo_evidence(recovery, runtime, *, cancel_event=None) -> dict:
    """Capture and preview only within an owned, paused simulation checkpoint.

    Fresh measurements remain a separate capture. Offline validation receives
    saved preview records only after compatibility and revision checks succeed.
    """
    if runtime.setup.get("execution_mode") != "simulation":
        raise ValueError("Gazebo outline evidence requires simulation")
    if runtime.conveyor_fault.status != "triggered" or not runtime.stopped:
        raise ValueError("Use Start failure and wait for the paused failure checkpoint")
    if runtime.active_executions or runtime.context.reservations:
        raise ValueError("Wait for owned execution and resource reservations to finish")
    live_recovery = runtime.part_slippage_recovery
    if live_recovery._task is not None and not live_recovery._task.done():
        raise ValueError("Wait for the Gazebo recovery operation to finish")
    event = pending_outline_event(recovery)
    generation, revision = recovery._generation, runtime.context.revision
    proposal_fingerprint = fingerprint(event)
    source_fingerprint = observation_fingerprint(recovery._observations)
    directory = Path(recovery.snapshot()["artifact_directory"])
    attempt = directory / "gazebo_evidence" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    report = {"status": "NEEDS_CONTEXT", "source_observation_fingerprint": source_fingerprint,
              "proposal_fingerprint": proposal_fingerprint, "execution_authorized": False,
              "command_sent": False}
    try:
        archived_scene = _saved_planning_scene(recovery._observations)
        if archived_scene is not None:
            # This explicit action evaluates the recorded hypothetical scene.
            # Later live poses cannot establish an unexecuted event's outcome.
            fresh = deepcopy(recovery._observations)
            report.update(evidence_source="saved planning scene", physical_execution_verified=False,
                          live_checkpoint_validation="not performed for archived model prediction")
        else:
            fresh = await asyncio.to_thread(capture_observations, runtime, live_recovery._failure)
            _check_cancelled(cancel_event)
            fresh["pending_nodes"] = live_recovery._pending_nodes()
            fresh_path = attempt / "observations.json"
            write_record(fresh_path, fresh)
            _check_cancelled(cancel_event)
            template = live_recovery._observations.get("prepared_recovery_request")
            if isinstance(template, dict):
                fresh["prepared_recovery_request"] = await _prepare_saved_capture(runtime, fresh, template, fresh_path)
            write_record(fresh_path, fresh)
            if runtime.context.revision != revision:
                raise ValueError("Gazebo checkpoint changed during evidence capture")
            differences = checkpoint_differences(recovery._observations, fresh)
            report["checkpoint_differences"] = differences
            if differences:
                report.update(status="new_observations_required", observation_path=str(fresh_path))
                return report
        owner = next((item for item in runtime.resource_agents if str(item.jid) == event.get("resource_jid")), None)
        prepare = getattr(owner, "prepare_recovery_outline_event_evidence", None)
        if not callable(prepare):
            raise ValueError(f"NEEDS_CONTEXT: {event.get('resource_jid')} has no declared event preview")
        session = recovery._prepared["multi_turn_session_state"]
        resource = recovery._prepared["recovery_resources"][event["resource_jid"]]
        pending = session.get("pending_outline_validation") or {}
        physical = deepcopy(pending.get("projected_physical_evidence") or
                            session.get("projected_physical_evidence") or
                            recovery._prepared.get("physical_evidence") or fresh["physical_evidence"])
        kwargs = dict(task=event,
            recovery_snapshot=deepcopy(pending.get("recovery_snapshot") or session.get("derived_resource_snapshots", {}).get(
                event["resource_jid"], resource.get("recovery_snapshot") or {})),
            recovery_des_model=deepcopy(resource.get("recovery_des_model") or {}), physical_evidence=physical,
            resource_evidence=deepcopy(fresh.get("resource_evidence", {}).get(event["resource_jid"], {})),
            observations=fresh, cancel_event=cancel_event)
        if inspect.iscoroutinefunction(prepare):
            evidence = await prepare(**kwargs)
        else:
            evidence = await asyncio.to_thread(prepare, **kwargs)
        _check_cancelled(cancel_event)
        if (recovery._stopped or recovery._generation != generation or runtime.context.revision != revision
                or fingerprint(pending_outline_event(recovery)) != proposal_fingerprint):
            raise ValueError("Discarded stale Gazebo preview")
        if not isinstance(evidence, dict) or evidence.get("command_sent") is not False:
            raise ValueError("Owner preview must explicitly record command_sent=false")
        if archived_scene is None:
            evidence["resource_evidence_capture"] = deepcopy(fresh)
        else:
            evidence["evidence_source"] = "saved planning scene"
            evidence["physical_execution_verified"] = False
        evidence["source_observation_fingerprint"] = source_fingerprint
        write_record(attempt / "event_evidence.json", evidence)
        report.update(status=evidence.get("status", "NEEDS_CONTEXT"), evidence=evidence)
        return report
    except asyncio.CancelledError:
        report.update(status="cancelled", reason="Stopped by operator")
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
        report["reason"] = str(exc)
        raise
    finally:
        write_record(attempt / "preview_report.json", report)


def _check_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError()


async def _prepare_saved_capture(runtime, fresh: dict, template: dict, path: Path) -> dict:
    """Persist raw evidence and preparation errors before publishing ready inputs."""
    write_record(path, fresh)
    try:
        return await _fresh_preparation(runtime, fresh, template)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        write_record(path.parent / "preparation_error.json", {
            "error_type": type(exc).__name__, "reason": str(exc),
            "observation_path": str(path), "context_revision": fresh.get("context_revision"),
            "captured_at": fresh.get("captured_at"), "recorded_at": datetime.now(timezone.utc).isoformat()})
        raise


async def _fresh_preparation(runtime, fresh: dict, template: dict) -> dict:
    """Rebuild prepared facts through the existing non-LLM preparation path."""
    from cais_spade_llm.recovery_framework.recovery_observations import captured_resource_snapshots

    planner = runtime.agent.process_planner
    tracker = deepcopy(fresh["part_tracker"])
    states = {name: row.get("state") for name, row in tracker.items()}
    locations = {name: row.get("location") for name, row in tracker.items()}
    resources = {}
    for rid, values in fresh["resource_states"].items():
        jid = fresh["jids"][rid]
        owner_state = fresh.get("resource_evidence", {}).get(jid, {}).get("state", {})
        resources[jid] = deepcopy(values)
        if "resource_state" in values:
            resources[jid]["current_state"] = deepcopy(values["resource_state"])
        elif "_current_state" in owner_state:
            resources[jid]["current_state"] = deepcopy(owner_state["_current_state"])
        if "resource_location" in values:
            resources[jid]["current_location"] = deepcopy(values["resource_location"])
        if "unavailable_resources" in fresh:
            resources[jid]["availability"] = "unavailable" if rid in fresh["unavailable_resources"] else values.get("availability", "available")
    missing = [fresh["jids"][rid] for rid, values in fresh["resource_states"].items()
               if "resource_state" in values and values["resource_state"] is None]
    if missing:
        raise ValueError("NEEDS_CONTEXT: recorded resource_state or owner _current_state is missing for " + ", ".join(missing))
    recorded_snapshots = captured_resource_snapshots(fresh)
    robot_ids = {row["resource_id"] for row in fresh.get("inputs", {}).get("scene", {}).get("robots", [])}
    for owner in getattr(runtime, "resource_agents", []):
        physical = fresh.get("physical_evidence", {}).get("snapshot", {}).get("resources", {}).get(owner.agent_name, {})
        jid = str(owner.jid)
        if jid not in recorded_snapshots:
            raise ValueError("NEEDS_CONTEXT: recorded owner snapshot is unavailable for " + owner.agent_name)
        snapshot = deepcopy(recorded_snapshots[jid])
        if owner.agent_name in robot_ids or getattr(owner, "static_capabilities", {}).get("resource_type") == "robot":
            pose = physical.get("current_pose")
            if isinstance(pose, list) and len(pose) == 7:
                pose = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose, strict=True))
            if isinstance(pose, dict):
                snapshot["current_pose"] = pose
        else:
            for key in ("current_pose", "base_pose", "joint_names", "joint_positions"):
                if key in physical and key in snapshot and fingerprint(snapshot[key]) != fingerprint(physical[key]):
                    raise ValueError("NEEDS_CONTEXT: conflicting captured " + owner.agent_name + "." + key +
                        ": " + json.dumps({"snapshot": snapshot[key], "physical": physical[key],
                            "source": {field: fresh.get("physical_evidence", {}).get("resources", {}).get(owner.agent_name, {}).get(field)
                                       for field in ("source", "frame", "simulation_time", "observed_monotonic", "launch_id")}},
                            ensure_ascii=False))
        if "held_part" in physical and snapshot.get("held_part") != physical["held_part"]:
            raise ValueError("NEEDS_CONTEXT: captured custody conflicts with owner state for " + owner.agent_name)
        if "availability" in resources.get(jid, {}):
            snapshot["availability"] = deepcopy(resources[jid]["availability"])
        recorded_snapshots[jid] = snapshot
    failure_context = deepcopy(template["failure_context_raw"])
    if "failure" in fresh:
        from cais_spade_llm.agents.intelligent_product.replanner.failure_context import build_failure_event

        failure = fresh["failure"]
        part = failure["part_name"]
        rid = next(rid for rid, jid in fresh["jids"].items() if jid == template["ra_jid"])
        failure_context = build_failure_event(
            failed_task_id=failure_context.get("failed_task_id", ""),
            failed_resource_jid=template["ra_jid"],
            failed_function_name=failure_context.get("failed_function_name", ""),
            final_status="failed:" + fresh["scenario"], part_name=part,
            observations=deepcopy(failure),
            affected_entities=[{"entity_type": "part", "entity_id": part,
                                "state": tracker[part]["state"], "location": tracker[part]["location"]}],
            state_before=failure.get("resource_values_before", {}).get(rid),
            state_after={**deepcopy(fresh["resource_states"][rid]),
                         "part_state": tracker[part]["state"], "part_location": tracker[part]["location"]})
    stuck = planner._build_resource_search_state(resource_jid=template["ra_jid"], resource_states=resources,
        default_resource_state="", part_states=states, part_locations=locations)
    prepared = await planner.prepare_recovery_request(
        stuck_state=stuck, P_id=deepcopy(template["P_id"]), ra_jid=template["ra_jid"],
        goal_state=template["goal_state"], tools_catalog=deepcopy(template["tools_catalog"]),
        part_tracker=tracker, obligation_targets=deepcopy(template["obligation_targets"]),
        recovery_feedback="", resource_states=resources, default_resource_state="",
        part_states=states, part_locations=locations, recovery_safety_context={},
        failure_context=failure_context,
        recorded_resource_snapshots=recorded_snapshots)
    context = getattr(runtime, "context", None)
    if context is not None and fresh.get("context_revision") is not None and context.revision != fresh["context_revision"]:
        raise ValueError("NEEDS_CONTEXT: runtime observations changed during preparation")
    for jid, evidence in fresh.get("resource_evidence", {}).items():
        if jid in prepared.get("recovery_resources", {}):
            prepared["recovery_resources"][jid]["resource_evidence"] = deepcopy(evidence)
    return prepared


async def prepare_owner_event_evidence(owner, *, task: dict, recovery_snapshot: dict,
                                       recovery_des_model: dict, physical_evidence: dict,
                                       resource_evidence: dict, observations: dict, cancel_event=None,
                                       preview_call=None, allow_live_capture: bool = True) -> dict:
    """Prepare owner evidence using the supplied scene and preview interface.

    Resource-owned projections use recorded models directly. Robot and KMR
    previews retain their separate scene and motion preparation. Offline callers
    prohibit live scene acquisition.
    """
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    if resource_evidence.get("execution_mode", getattr(owner, "execution_mode", None)) != "simulation":
        raise ValueError("Owner outline previews require simulation")
    _check_cancelled(cancel_event)
    from cais_spade_llm.recovery_framework.recovery_observations import projection_capture_compatibility

    compatibility = projection_capture_compatibility(physical=physical_evidence, task=task,
                                                      resource_evidence=resource_evidence)
    if compatibility["status"] == "NEEDS_CONTEXT":
        return {"status": "NEEDS_CONTEXT", "command_sent": False, "execution_authorized": False,
                "capture_compatibility": deepcopy(compatibility),
                "reason": "; ".join(row["reason"] for row in compatibility["checks"]
                                    if row["status"] == "NEEDS_CONTEXT")}
    resource_type = resource_evidence.get("resource_type")
    if resource_type == "kmr" and resource_evidence.get("environment_models"):
        from cais_spade_llm.recovery_framework.kmr_outline_evidence import prepare_kmr_event_evidence

        return await prepare_kmr_event_evidence(owner, task=task, recovery_snapshot=recovery_snapshot,
            recovery_des_model=recovery_des_model, physical_evidence=physical_evidence,
            resource_evidence=resource_evidence, observations=observations, cancel_event=cancel_event,
            preview_call=preview_call, allow_live_capture=allow_live_capture)
    if resource_type != "robot":
        projector = getattr(owner, "project_recovery_outline_event", None)
        if not callable(projector):
            return {"status": "NEEDS_CONTEXT", "command_sent": False, "execution_authorized": False,
                    "reason": "Resource-owned event projection is unavailable",
                    "context_requests": [f'/resources/{task.get("resource_jid")}/outline_event_evidence']}
        projection = projector(task=deepcopy(task), recovery_snapshot=deepcopy(recovery_snapshot),
            recovery_des_model=deepcopy(recovery_des_model), physical_input={
                "physical_evidence": deepcopy(physical_evidence), "resource_evidence": deepcopy(resource_evidence)})
        _check_cancelled(cancel_event)
        result = {"command_sent": False, "execution_authorized": False, "geometry_evidence": projection}
        if projection.get("owner_effect_rejected") is True:
            return {**result, **projection}
        return {**result, "status": "prepared" if projection.get("status") == "projected" else "NEEDS_CONTEXT",
                "context_requests": deepcopy(projection.get("context_requests") or []),
                "reason": projection.get("reason") or "; ".join(projection.get("unresolved") or [])}
    from cais_spade_llm.resources.resource_safety_preparation import (
        finalize_outline_event_preview,
        outline_event_preview_progress,
        prepare_outline_event_departure,
        prepare_outline_event_preview_request,
        install_outline_cartesian_route,
    )

    prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=recovery_snapshot,
        recovery_des_model=recovery_des_model, physical_evidence=physical_evidence,
        resource_evidence=resource_evidence, static_capabilities=getattr(owner, "static_capabilities", None))
    end = task.get("expected_end_state") or {}
    destination = end.get("resource_location", end.get("current_location"))
    move_group = (resource_evidence.get("controller_config") or {}).get("move_group") or {}
    needs_named_pose = (
        prepared.get("reason") == "No declared owner motion policy supports these event end conditions"
        and isinstance(destination, str) and bool(destination)
        and isinstance(move_group.get("group_name"), str) and bool(move_group["group_name"])
    )
    if prepared["status"] != "ready" and not needs_named_pose:
        return prepared
    motions = prepared["motion_requests"]
    responses, scene_capture, mounting_evidence, root_evidence = [], None, None, None
    route_calculations = []
    if motions or needs_named_pose:
        from cais_spade_llm.recovery_framework.recovery_motion_preview import (
            apply_declared_mounting_contacts, capture_preview_scene, preview_recovery_motion,
            named_pose_joint_target, preview_scene_fingerprint, project_preview_scene, prepare_recorded_root_transforms,
        )
        run_preview = preview_call or preview_recovery_motion
        if not allow_live_capture and preview_call is None:
            raise ValueError("Offline owner evidence requires a saved-model preview interface")

        controller = resource_evidence.get("controller_config") or {}
        move_group = controller.get("move_group") or {}
        settings = resource_evidence.get("inputs", {}).get("scene", {}).get(
            "safety_preparation", {}).get("outline_preview") or {}
        required = ("action_name", "planning_scene_service", "preview_parameter_service", "max_translation_step",
                    "max_rotation_step", "max_joint_step", "velocity_scaling", "acceleration_scaling", "timeout_seconds")
        missing = [field for field in required if field not in settings]
        if missing:
            return {"status": "NEEDS_CONTEXT", "command_sent": False,
                    "reason": "Missing declared outline preview settings: " + ", ".join(missing)}
        scene_capture = _saved_planning_scene(observations)
        if scene_capture is None:
            if not allow_live_capture:
                return {"status": "NEEDS_CONTEXT", "command_sent": False,
                        "execution_authorized": False,
                        "reason": "Saved planning_scene_capture is missing; capture it through Start failure"}
            scene_capture = await capture_preview_scene(execution_mode="simulation",
                planning_scene_service=settings["planning_scene_service"],
                preview_parameter_service=settings["preview_parameter_service"], timeout_seconds=settings["timeout_seconds"])
        elif preview_scene_fingerprint(scene_capture["planning_scene"]) != scene_capture["scene_fingerprint"]:
            raise ValueError("Saved planning scene fingerprint changed")
        _check_cancelled(cancel_event)
        if needs_named_pose:
            physical_evidence = deepcopy(physical_evidence)
            destination = task["expected_end_state"].get("resource_location",
                task["expected_end_state"].get("current_location"))
            try:
                target = named_pose_joint_target(scene_capture["robot_description_semantic"],
                    group_name=move_group.get("group_name"), named_pose=destination)
            except ValueError as exc:
                return {**prepared, "reason": str(exc), "context_requests": [
                    "/resources/" + task["resource_jid"] + "/outline_event_evidence"]}
            names = physical_evidence["snapshot"]["resources"][owner.agent_name]["joint_names"]
            if set(target["name"]) != set(names) or len(names) != len(target["name"]):
                raise ValueError("NEEDS_CONTEXT: SRDF named pose does not match captured joint identities")
            positions = dict(zip(target["name"], target["position"], strict=True))
            physical_evidence["resource_models"][owner.agent_name].setdefault("named_joint_poses", {})[destination] = {
                "joint_names": deepcopy(names), "positions": [positions[name] for name in names],
                "source": "captured robot_description_semantic", "model_fingerprint": scene_capture["model_fingerprint"],
            }
            prepared = prepare_outline_event_preview_request(task=task, recovery_snapshot=recovery_snapshot,
                recovery_des_model=recovery_des_model, physical_evidence=physical_evidence,
                resource_evidence=resource_evidence, static_capabilities=getattr(owner, "static_capabilities", None))
            if prepared["status"] != "ready":
                return prepared
            motions = prepared["motion_requests"]
        initial = observations["physical_evidence"]
        bindings = _prepare_scene_object_bindings(observations=observations, scene_capture=scene_capture,
                                                 task=task, resource_id=owner.agent_name)
        if bindings["status"] != "ready":
            return bindings
        prepared["physical_evidence"]["scene_object_bindings"] = bindings["scene_object_bindings"]
        prepared["physical_evidence"]["support_collision_object_bindings"] = _support_collision_object_bindings(
            scene_capture, initial)
        recorded_scene_configuration = observations.get("inputs", {}).get("scene", {})
        owner_contract = prepared["physical_evidence"]["resource_models"][owner.agent_name][
            "event_projection_contracts"][prepared["event_record"]["contract_id"]]
        if owner_contract.get("version") == 7 and owner_contract.get("nominal_snap") is not None:
            prepared["physical_evidence"]["planning_scene_capture"] = deepcopy(scene_capture)
            prepared["physical_evidence"]["planning_scene_configuration"] = {
                key: deepcopy(recorded_scene_configuration[key]) for key in
                ("robots", "planning_scene_mounting_contacts") if key in recorded_scene_configuration}
        roots = prepare_recorded_root_transforms(scene_capture, prepared["physical_evidence"])
        root_evidence = roots["evidence"]
        prepared["native_requests"] = []
        capture_fingerprint = fingerprint(physical_evidence)
        for index in range(len(motions)):
            progress = outline_event_preview_progress(prepared_preview=prepared, motion_results=responses)
            if progress.get("continuous_mating_rejected") is True or progress.get("physical_endpoint_rejected") is True:
                break
            if progress.get("pending_nominal_snap") is not None:
                from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_scene_endpoint

                pending = progress["pending_nominal_snap"]
                if prepared.get("nominal_snap_result") is not None:
                    return {"status": "NEEDS_CONTEXT", "command_sent": False, "execution_authorized": False,
                        "reason": pending.get("reason"), "scene_endpoint_result": prepared["nominal_snap_result"],
                        "motion_results": responses}
                candidate = deepcopy(progress["physical_evidence"])
                candidate["snapshot"] = deepcopy(pending["candidate_state"])
                endpoint_scene = project_preview_scene(scene_capture, candidate)
                if "planning_scene_mounting_contacts" in recorded_scene_configuration:
                    mounted = apply_declared_mounting_contacts(endpoint_scene["planning_scene"],
                        robot_description=scene_capture["robot_description"],
                        robots=recorded_scene_configuration.get("robots", []), physical_evidence=initial,
                        declarations=recorded_scene_configuration["planning_scene_mounting_contacts"])
                    endpoint_scene["planning_scene"] = mounted["planning_scene"]
                    mounting_evidence = mounted["evidence"]
                endpoint_request = {
                    "planning_scene": deepcopy(endpoint_scene["planning_scene"]),
                    "start_state": deepcopy(endpoint_scene["start_state"]),
                    "scene_fingerprint": preview_scene_fingerprint(endpoint_scene["planning_scene"]),
                    "model_fingerprint": scene_capture["model_fingerprint"], "resource_id": owner.agent_name,
                    "capture_fingerprint": capture_fingerprint,
                    "proposal_fingerprint": recovery_validation_fingerprint(task),
                    "group_name": move_group.get("group_name"), "link_name": move_group.get("ee_link"),
                    "frame_id": move_group.get("frame_id"),
                    **{field: settings[field] for field in required if field not in {
                        "action_name", "planning_scene_service", "preview_parameter_service"}},
                }
                if roots["recorded_root_transforms"]:
                    endpoint_request["recorded_root_transforms"] = deepcopy(roots["recorded_root_transforms"])
                endpoint_request = bind_scene_endpoint(endpoint_request, pending["scene_endpoint_record"])
                _check_cancelled(cancel_event)
                prepared["nominal_snap_result"] = await run_preview(endpoint_request,
                    execution_mode="simulation", action_name=settings["action_name"],
                    timeout_seconds=settings["timeout_seconds"], cancel_event=cancel_event)
                _check_cancelled(cancel_event)
                progress = outline_event_preview_progress(prepared_preview=prepared, motion_results=responses)
                if progress.get("physical_endpoint_rejected") is True:
                    break
                if progress.get("pending_nominal_snap") is not None:
                    return {"status": "NEEDS_CONTEXT", "command_sent": False, "execution_authorized": False,
                        "reason": progress["pending_nominal_snap"].get("reason"),
                        "scene_endpoint_result": prepared["nominal_snap_result"], "motion_results": responses}
            custody_kind = (prepared.get("custody_operation") or {}).get("kind")
            departure = custody_kind == "grasp" and index == prepared["custody_motion_index"]
            if custody_kind is not None and index == prepared["custody_motion_index"]:
                prepared = prepare_outline_event_departure(prepared_preview=prepared, progress=progress)
            physical = progress["physical_evidence"]
            projected = project_preview_scene(scene_capture, physical)
            if "planning_scene_mounting_contacts" in recorded_scene_configuration:
                mounted = apply_declared_mounting_contacts(projected["planning_scene"],
                    robot_description=scene_capture["robot_description"],
                    robots=recorded_scene_configuration.get("robots", []), physical_evidence=initial,
                    declarations=recorded_scene_configuration["planning_scene_mounting_contacts"])
                projected["planning_scene"] = mounted["planning_scene"]
                mounting_evidence = mounted["evidence"]
            motion = prepared["motion_requests"][index]
            request = {
                "planning_scene": deepcopy(projected["planning_scene"]), "start_state": deepcopy(projected["start_state"]),
                "scene_fingerprint": preview_scene_fingerprint(projected["planning_scene"]),
                "model_fingerprint": scene_capture["model_fingerprint"],
                "resource_id": owner.agent_name,
                "capture_fingerprint": capture_fingerprint, "proposal_fingerprint": recovery_validation_fingerprint(task),
                "group_name": move_group.get("group_name"), "link_name": move_group.get("ee_link"),
                "frame_id": move_group.get("frame_id"),
                **{field: settings[field] for field in required if field not in {
                    "action_name", "planning_scene_service", "preview_parameter_service"}},
                **deepcopy(motion),
            }
            if roots["recorded_root_transforms"]:
                request["recorded_root_transforms"] = deepcopy(roots["recorded_root_transforms"])
            if departure:
                from cais_spade_llm.resources.recovery_support_departure import prepare_support_departure
                from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_support_departure

                supporting = prepare_support_departure(resource_id=owner.agent_name,
                    state=physical["snapshot"], evidence=physical)
                if supporting.get("status") != "prepared":
                    return {"status": "NEEDS_CONTEXT", "command_sent": False,
                        "execution_authorized": False, "reason": supporting.get("reason"),
                        "support_departure": supporting, "motion_results": responses}
                request = bind_support_departure(request, supporting)
            contract = prepared["physical_evidence"]["resource_models"][owner.agent_name][
                "event_projection_contracts"][prepared["event_record"]["contract_id"]]
            mating_interval = (physical_evidence.get("physical_preview_contract_version") in {6, 7}
                and index in contract.get("placement_arrival", {}).get("mating_motion_indices", [])
                and contract.get("placement_target", {}).get("simulation_mating_contact") is not None)
            if mating_interval:
                from cais_spade_llm.resources.recovery_mating_contact import prepare_mating_contact
                from cais_spade_llm.recovery_framework.recovery_motion_preview import bind_mating_contact

                target = contract["placement_target"]
                end = motion["waypoints"][-1]
                target_tool_pose = [*(end["position"][axis] for axis in ("x", "y", "z")),
                                    *(end["orientation"][axis] for axis in ("x", "y", "z", "w"))]
                mating = prepare_mating_contact(resource_id=owner.agent_name, part_name=task["part_name"],
                    state=physical["snapshot"], evidence=physical,
                    owner_evidence=contract["mating_owner_evidence"],
                    target_origin_pose=target["target_origin_pose"], target_tool_pose=target_tool_pose)
                if mating.get("status") != "prepared":
                    return {"status": "NEEDS_CONTEXT", "command_sent": False,
                        "execution_authorized": False, "reason": mating.get("reason"),
                        "mating_contact": mating, "motion_results": responses}
                request = bind_mating_contact(request, mating)
            response = await run_preview(request, execution_mode="simulation",
                action_name=settings["action_name"], timeout_seconds=settings["timeout_seconds"], cancel_event=cancel_event)
            failure = response.get("cartesian_failure_evidence") or {}
            if (physical_evidence.get("physical_preview_contract_version") in {5, 6, 7}
                    and request["mode"] == "cartesian" and not departure and not mating_interval
                    and not (custody_kind is not None and index == prepared["custody_motion_index"])
                    and failure.get("complete_inputs") is True
                    and failure.get("failure_kind") in {"ik_search_failed", "joint_jump"}):
                from cais_spade_llm.recovery_framework.recovery_motion_preview import (
                    prepare_cartesian_route_search, archive_cartesian_route,
                )

                _check_cancelled(cancel_event)
                search_request = prepare_cartesian_route_search(request,
                    prepared["physical_evidence"]["resource_models"][owner.agent_name]["static_capabilities"])
                search_result = await run_preview(search_request, execution_mode="simulation",
                    action_name=settings["action_name"], timeout_seconds=settings["timeout_seconds"], cancel_event=cancel_event)
                proof = archive_cartesian_route(search_request, search_result,
                    previous_request=request, previous_result=response)
                route_calculations.append(proof)
                _check_cancelled(cancel_event)
                if proof.get("candidate"):
                    prepared = install_outline_cartesian_route(prepared_preview=prepared, motion_index=index,
                        search_request=search_request, route_proof=proof, progress=progress)
                    request.update(deepcopy(prepared["motion_requests"][index]))
                    response = await run_preview(request, execution_mode="simulation",
                        action_name=settings["action_name"], timeout_seconds=settings["timeout_seconds"], cancel_event=cancel_event)
            prepared["native_requests"].append(deepcopy(request))
            responses.append(response)
            _check_cancelled(cancel_event)
            if responses[-1].get("status") != "PREVIEW_COMPLETE":
                break
    result = finalize_outline_event_preview(prepared_preview=prepared, motion_results=responses,
        owner_projector=getattr(owner, "project_recovery_outline_event", None), resource_evidence=resource_evidence)
    if "physical_evidence" not in result:
        result["physical_evidence"] = deepcopy(prepared["physical_evidence"])
    result["motion_results"] = responses
    result["native_requests"] = deepcopy(prepared.get("native_requests", []))
    if prepared.get("nominal_snap_result") is not None:
        result["scene_endpoint_result"] = deepcopy(prepared["nominal_snap_result"])
    if route_calculations:
        result["route_calculations"] = route_calculations
    if mounting_evidence is not None:
        result["mounting_contact_evidence"] = mounting_evidence
    if root_evidence is not None:
        result["recorded_root_evidence"] = root_evidence
    if scene_capture is not None:
        result["planning_scene_capture"] = scene_capture
    return result


def _support_collision_object_bindings(scene_capture: dict, physical_evidence: dict) -> dict:
    """Associate captured support collisions using the exact SDF producer identities."""
    from cais_spade_llm.recovery_framework.part_collision import observed_collision_identities

    objects = {row["id"] for row in scene_capture["planning_scene"].get("world", {}).get("collision_objects", [])}
    bindings = {}
    for name, model in physical_evidence.get("physics_evidence", {}).get("models", {}).items():
        if model.get("static") is not True or not model.get("pose"):
            continue
        try:
            identity = observed_collision_identities(name, {"name": name, "frame": "world", "pose": model["pose"]})
        except (ValueError, FileNotFoundError):
            continue
        for declared in identity["collision_objects"]:
            link = model.get("links", {}).get(declared["link"], {})
            # GetScopedName and the MoveIt producer have different declared
            # formats. Construct both from the same SDF identity, never a rewrite.
            native_id = "::".join((declared["model"], declared["link"], declared["collision"]))
            matches = [row for row in link.get("collisions", []) if row.get("id") == native_id]
            if len(matches) != 1 or declared["id"] not in objects:
                continue
            bindings[native_id] = {"object_id": declared["id"], "source": {
                "producer": identity["source"], "source_files": identity["source_files"],
                "producer_sha256": identity["producer_sha256"], "identity": deepcopy(declared),
                "physics_collision": deepcopy(matches[0]),
                "scene_fingerprint": scene_capture["scene_fingerprint"]}}
    return bindings


def _prepare_scene_object_bindings(*, observations: dict, scene_capture: dict,
                                  task: dict, resource_id: str) -> dict:
    """Resolve needed attachment bindings without removing any scene geometry."""
    from cais_spade_llm.recovery_framework.recovery_motion_preview import SceneObjectBindingError, build_scene_object_bindings

    physical = observations["physical_evidence"]
    snapshot = physical["snapshot"]
    required = {rid for rid, row in snapshot.get("resources", {}).items()
                if "held_part" in row and row["held_part"] is not None}
    missing_facts = []
    try:
        payloads, targets = _scene_owner_records(observations, scene_capture)
        required.update(row["contained_by"] for row in snapshot.get("parts", {}).values()
                        if row.get("contained_by") in targets)
        if resource_id in targets or any(task.get(field, {}).get("held_part") is not None
                                         for field in ("expected_start_state", "expected_end_state")):
            required.add(resource_id)
        for row in scene_capture["planning_scene"]["robot_state"].get("attached_collision_objects", []):
            owners = [rid for rid, target in targets.items() if target.get("link_name") == row.get("link_name")]
            if len(owners) != 1:
                missing_facts.append({"ref": "/planning_scene_capture/planning_scene/robot_state/attached_collision_objects",
                                      "reason": f"Recorded attachment owner is missing or ambiguous for {row.get('link_name')}"})
                raise ValueError(f"Recorded attachment owner is missing or ambiguous for {row.get('link_name')}")
            required.add(owners[0])
        for rid in sorted(required):
            if rid not in targets:
                jid = observations.get("jids", {}).get(rid, rid)
                missing_facts.append({"ref": f"/resources/{jid}/resource_evidence/controller_config/attach",
                                      "reason": f"Attachment declaration is missing for {rid}"})
            configuration = physical.get("resource_models", {}).get(rid, {}).get("configuration", {})
            if not configuration.get("reference_link"):
                missing_facts.append({"ref": f"/physical_evidence/resource_models/{rid}/configuration/reference_link",
                                      "reason": f"Recorded reference_link is missing for {rid}"})
            row = snapshot.get("resources", {}).get(rid, {})
            links = physical.get("resources", {}).get(rid, {}).get("link_states", {})
            if row.get("current_pose") is None and configuration.get("reference_link") not in links:
                missing_facts.append({"ref": f"/physical_evidence/snapshot/resources/{rid}/current_pose",
                                      "reason": f"Recorded reference pose is missing for {rid}"})
        if missing_facts:
            raise ValueError("; ".join(item["reason"] for item in missing_facts))
        bindings = build_scene_object_bindings(scene_capture, physical, payload_evidence=payloads,
            attachment_targets=targets, required_attachment_resources=required)
    except (KeyError, ValueError) as exc:
        # These calls validate saved data. Keep their exact failure and source;
        # programming exceptions such as TypeError still surface as stage errors.
        if isinstance(exc, SceneObjectBindingError):
            missing_facts = [{"ref": ref, "reason": str(exc)} for ref in exc.context_requests]
        elif not missing_facts:
            missing_facts = [{"ref": ref, "reason": str(exc)} for ref in
                             ("/physical_evidence", "/planning_scene_capture")]
        refs = list(dict.fromkeys(item["ref"] for item in missing_facts))
        return {"status": "NEEDS_CONTEXT", "reason": f"{type(exc).__name__}: {exc}",
            "missing_facts": missing_facts,
            "context_requests": refs, "required_attachment_resources": sorted(required),
            "command_sent": False, "execution_authorized": False, "interval_complete": False}
    return {"status": "ready", "scene_object_bindings": bindings,
            "required_attachment_resources": sorted(required), "command_sent": False,
            "execution_authorized": False}


def _scene_owner_records(observations: dict, scene_capture: dict) -> tuple[dict, dict]:
    """Read exact collision identities and declared attachment targets from owners."""
    from cais_spade_llm.recovery_framework.kmr_recovery import recorded_scene_payloads

    payloads, targets = {}, {}
    scene = scene_capture["planning_scene"]
    released = recorded_scene_payloads(observations, scene_capture)
    for name, part in observations["physical_evidence"].get("parts", {}).items():
        identity = part.get("collision_scene_identity")
        if isinstance(identity, dict) and identity.get("status") == "recorded":
            payloads[f"parts/{name}"] = deepcopy(released.get(name, identity))
    for rid, jid in observations["jids"].items():
        evidence = observations.get("resource_evidence", {}).get(jid, {})
        payload = evidence.get("payload_collision")
        if isinstance(payload, dict) and payload.get("collision_objects"):
            payloads[rid] = deepcopy(payload)
        pickup = observations.get("failure", {}).get("pickups", {}).get(rid, {})
        # Historical acknowledgements contribute identities only; current scene
        # geometry and current observed transforms establish their placements.
        result = pickup.get("observations", {}).get("controller_result", {}).get("observations", {})
        for index, row in enumerate(result.get("simulation_execution", {}).get("primitive_results", [])):
            recorded = (row.get("result") or {}).get("command_evidence", {}).get("payload_collision")
            if not recorded:
                recorded = row.get("command_evidence", {}).get("payload_collision")
            if isinstance(recorded, dict) and recorded.get("collision_objects"):
                payloads[f"{rid}/{index}"] = deepcopy(recorded)
        configuration = evidence.get("controller_config") or {}
        attach = configuration.get("attach") or {}
        link = attach.get("primary_attach_link")
        physical = observations["physical_evidence"].get("resources", {}).get(rid, {})
        attached_model = physical.get("attachment", {}).get("model_name")
        current = [row for row in scene["robot_state"].get("attached_collision_objects", [])
                   if any(row.get("object", {}).get("id") == item.get("id")
                          for value in payloads.values() if value.get("model_name") == attached_model
                          for item in value.get("collision_objects", []))]
        if current:
            links = {row["link_name"] for row in current}
            if len(links) != 1:
                raise ValueError(f"Conflicting attachment links for {rid}")
            link = next(iter(links))
            touch = sorted({name for row in current for name in row.get("touch_links", [])})
        else:
            prefix = configuration.get("payload_collision", {}).get("robot_prefix")
            touch = [name for name in scene.get("allowed_collision_matrix", {}).get("entry_names", [])
                     if isinstance(prefix, str) and name.startswith(prefix + "rg2")]
            touch.extend(attach.get("attach_link_candidates", []))
        if link and touch:
            targets[rid] = {"link_name": link, "touch_links": list(dict.fromkeys(touch)),
                            "source": "captured owner attachment declaration and current planning scene"}
    return payloads, targets
