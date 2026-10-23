"""Scenario gates around ordinary negotiated robot execution."""

from __future__ import annotations

import asyncio
import math
import time
from copy import deepcopy
from pathlib import Path


class SlippagePickupGate:
    """Release the preset's two admitted pickup approaches together."""

    def __init__(self, runtime, configuration: dict) -> None:
        self.runtime = runtime
        other = configuration["additional_condition"]
        self.parts = {configuration["resource_id"]: configuration["part_name"],
                      other["resource_id"]: other["part_name"]}
        self.arrived: dict[str, str] = {}
        self.released = asyncio.Event()

    async def wait(self, task: dict) -> None:
        """Wait for both owning tasks after their normal safety approvals."""
        rid = task["resource_id"]
        if (task["event_name"] != "pick_approach" or rid not in self.parts
                or task["parameters"].get("part_name") != self.parts[rid] or self.released.is_set()):
            return
        context = self.runtime.context
        if task.get("run_id") != context.run_id or context.pending_for(task["task_id"]) != task:
            raise ValueError("Pickup gate requires the current admitted resource task")
        if self.runtime.stopped:
            raise asyncio.CancelledError()
        previous = self.arrived.get(rid)
        if previous is not None and previous != task["task_id"]:
            raise ValueError("Pickup gate received another task for the same resource")
        self.arrived[rid] = task["task_id"]
        context.negotiations.append({
            "kind": "slippage_pickup_ready", "run_id": context.run_id,
            "task_id": task["task_id"], "resource_id": rid, "timestamp": time.time(),
        })
        if set(self.arrived) == set(self.parts):
            context.negotiations.append({
                "kind": "slippage_pickups_released", "run_id": context.run_id,
                "tasks": dict(self.arrived), "timestamp": time.time(),
            })
            self.released.set()
        await self.released.wait()
        recording_ready = getattr(self.runtime, "slippage_recording_ready", None)
        if recording_ready is not None:
            await recording_ready.wait()
        if self.runtime.stopped or context.pending_for(task["task_id"]) != task:
            raise asyncio.CancelledError()


async def execute_generated_recovery(recovery) -> None:
    """Execute the reviewed program on the owning simulation controllers.

    Args:
        recovery: The current ProductAgent-owned Part slippage session.

    Formal safety admission is explicitly skipped for this run. Controller
    outcomes, custody, configured placement geometry and program ownership are
    still required before unfinished nominal work can resume.
    """
    from cais_spade_llm.recovery_framework.diagnostics import write_record
    from cais_spade_llm.recovery_framework import ROOT
    from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent
    from cais_spade_llm.recovery_framework.part_slippage_recovery import diagnostic_setup
    from cais_spade_llm.resources.gazebo_programs import current_program_revision

    runtime, context = recovery.runtime, recovery.runtime.context
    if getattr(recovery, "offline", False) or recovery._proposal.get("offline_recovery"):
        raise ValueError("Offline proposals require Test in Gazebo before execution")
    if not runtime.diagnostic_cca_bypass or not diagnostic_setup(runtime.setup):
        raise ValueError("Generated diagnostic recovery requires the selected simulation run")
    if context.pending_tasks or context.reservations or runtime.active_executions:
        raise ValueError("Interrupted nominal work has not released its execution ownership")
    tasks = deepcopy(recovery._proposal.get("macro_tasks") or [])
    if not tasks:
        raise ValueError("No completed primitive composition is available")
    owners = {str(owner.jid): owner for owner in runtime.resource_agents}
    completed, records = set(), []
    identifiers = [task["outline_id"] for task in tasks]
    if len(set(identifiers)) != len(tasks):
        raise ValueError("Primitive composition repeats an outline_id")
    for task in tasks:
        owner = owners.get(task["resource_jid"])
        controller = getattr(owner, "_controller", None)
        # The KMR runs its primitives on the Gazebo worker bound to this runtime.
        mobile = isinstance(owner, KMRResourceAgent)
        prepared = (getattr(owner, "environment_runtime", None) is runtime if mobile else
                    getattr(owner, "execution_mode", None) == "simulation"
                    and getattr(controller, "execution_mode", None) == "simulation")
        if owner is None or owner.agent_name not in context.permitted_resources or not prepared:
            raise ValueError("Recovery requires the resource's prepared simulation controller")
        if not set(task.get("predecessors") or []).issubset(completed):
            raise ValueError("Primitive composition is not ordered after its predecessors")
        if recovery._stopped:
            raise asyncio.CancelledError()
        if runtime.program_revision and current_program_revision(ROOT / runtime.scene_file) != runtime.program_revision:
            raise ValueError("The saved resource programs changed; regenerate recovery")
        task_id = f"{context.run_id}_recovery_{recovery._generation}_{task['outline_id']}"
        record = {
            "task_id": task_id, "run_id": context.run_id, "outline_id": task["outline_id"],
            "resource_id": owner.agent_name, "safety_status": "skipped",
            "status": "executing", "started_at_unix": time.time(),
            "primitive_steps": deepcopy(task["primitive_steps"]),
        }
        records.append(record)
        recovery._publish(execution=records, message=f"Executing {task['macro_name']} ({len(records)}/{len(tasks)}).")
        if mobile:
            # A release record belongs to the macro that produced it.
            owner._released_part = None
        else:
            controller._simulation_task_id = task_id
            controller._simulation_task_run_id = context.run_id
            controller._simulation_task_part = task.get("part_name")
            controller._simulation_fault_request = None
            controller._simulation_fault_evidence = None
        execution = asyncio.create_task(owner.execute_recovery_macro(
            macro_name=task["macro_name"], primitive_steps=task["primitive_steps"],
            expected_start_state=task.get("task_metadata", {}).get("in_state") or None,
            expected_snapshot=task.get("expected_snapshot"),
            out_state=task.get("task_metadata", {}).get("out_state") or None,
            product_jid=runtime.product_jid, task_id=task_id,
            part_name=task.get("part_name"),
            observed_pose=deepcopy(context.part_tracker.get(task.get("part_name"), {}).get("observed_pose")),
            **task.get("task_params", {}),
        ))
        runtime.active_executions.add(execution)
        try:
            result = await execution
            record.update(status=result.get("status", "failed"), controller_result=deepcopy(result))
            if result.get("status") != "completed":
                raise ValueError(result.get("content") or "A recovery primitive did not complete")
            if recovery._stopped:
                raise asyncio.CancelledError()
            reconcile = _reconcile_mobile_completion if mobile else _reconcile_completion
            record["observations"] = await reconcile(runtime, owner, task, task_id, result)
            completed.add(task["outline_id"])
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            runtime.active_executions.discard(execution)
            record["completed_at_unix"] = time.time()
            write_record(Path(recovery._state["artifact_directory"]) / "execution.json", records)
            recovery._publish(execution=records)
    for owner in owners.values():
        actor = context.resources[owner.agent_name]
        declaration = actor.model["state_variables"].get("resource_state")
        if declaration and actor.valuation.get("resource_state") not in declaration["domain"]:
            raise ValueError(f"{owner.agent_name} has no observed nominal reentry state")


async def _reconcile_completion(runtime, owner, task: dict, task_id: str, result: dict) -> dict:
    from cais_spade_llm.recovery_framework.failure_effects import _observe_part, observe_robot
    from cais_spade_llm.recovery_framework.workflow_execution import _part_geometry, _destination_geometry, _observe_robot_home

    context, controller = runtime.context, owner._controller
    pose = await observe_robot(runtime, owner.agent_name)
    observed = {"robot_pose": pose, "held_part": owner._held_part,
                "attached_model": controller._attached_model}
    if owner._held_part is not None:
        if controller._attached_model != context.geometry[owner._held_part]["model_name"]:
            raise ValueError("Recovery custody does not match the observed controller attachment")
    elif controller._attached_model is not None:
        raise ValueError("Recovery reports an empty hand with a retained controller attachment")
    end = task.get("expected_end_state") or {}
    if "held_part" in end and end["held_part"] != owner._held_part:
        raise ValueError("Observed recovery custody differs from the generated end state")
    values = deepcopy(context.resources[owner.agent_name].valuation)
    values.update(resource_state=owner._current_state, held_part=owner._held_part)
    for field in ("resource_location", "task_ctx.origin_resource_location", "task_ctx.destination_location", "task_ctx.part_name"):
        if field in end:
            value = end[field]
            declaration = context.models[owner.agent_name]["state_variables"][field]
            if "domain" in declaration and value not in declaration["domain"]:
                raise ValueError(f"Recovery cannot resume from {field}={value}")
            values[field] = deepcopy(value)
    if end.get("resource_location") == "home":
        observed["home"] = await asyncio.to_thread(_observe_robot_home, owner)
        if observed["home"].get("fresh_stable") is not True:
            raise ValueError("Recovery home completion was not observed")
    part = task.get("part_name")
    entry = deepcopy(context.part_tracker.get(part))
    buffer_zone = None
    released_buffer_fields = []
    if entry is not None:
        if owner._held_part == part:
            if entry["location"] == "Buffer For Machined parts":
                released_buffer_fields = [field for field, value in context.resources[entry["location"]].valuation.items()
                                          if field.startswith("zone_") and field.endswith("_part") and value == part]
                if len(released_buffer_fields) != 1:
                    raise ValueError("Recovery pickup does not identify the previous Buffer zone custody")
            entry.update(location=owner.agent_name)
        elif end.get("part_location") == context.product_name:
            target = _part_geometry(context, part).get("target_origin_pose")
            if not target:
                raise ValueError(f"No configured assembly pose for {part}")
            position = await asyncio.to_thread(_observe_part, controller, context.geometry[part]["model_name"])
            observed["part_pose"] = position
            if any(not math.isfinite(position[axis]) or abs(position[axis] - target[axis]) > .006
                   for axis in ("x", "y", "z")):
                raise ValueError(f"{part} was not observed at its configured assembly target")
            entry.update(location=context.product_name, observed_pose=position)
            fact = {"process": "assembly", "target": context.geometry[part]["target"]}
            if fact not in entry["processCompleted"]:
                entry["processCompleted"].append(fact)
        elif end.get("part_location") and end["part_location"] != entry["location"]:
            destination = end.get("part_location")
            if destination not in {"Buffer For Machined parts", "Exit"}:
                raise ValueError("Recovery part transition requires observed custody or configured placement")
            target = _destination_geometry(context, part, destination, owner.agent_name)["target_origin_pose"]
            position = await asyncio.to_thread(_observe_part, controller, context.geometry[part]["model_name"])
            if any(not math.isfinite(position[axis]) or abs(position[axis] - target[axis]) > .006
                   for axis in ("x", "y", "z")):
                raise ValueError(f"{part} was not observed at {destination}")
            observed["part_pose"] = position
            entry.update(location=destination, observed_pose=position)
            if destination == "Buffer For Machined parts":
                buffer_zone = next((zone["zone"] for zone in context.inputs["scene"][destination]["zones"]
                                    if math.dist(zone["pose"][:3], [target[axis] for axis in ("x", "y", "z")]) < .006), None)
                if buffer_zone is None or context.resources[destination].valuation[f"zone_{buffer_zone}_part"] not in (None, part):
                    raise ValueError("Observed recovery placement does not identify a free Buffer zone")
        entry["last_task"] = task_id
    with context.admission_lock:
        actor = context.resources[owner.agent_name]
        before, products_before = context.snapshot(), deepcopy(context.part_tracker)
        actor.valuation = values
        actor.revision += 1
        actor.evidence = "resource"
        owner._position = pose
        if entry is not None:
            context.part_tracker[part] = entry
        if buffer_zone is not None or released_buffer_fields:
            buffer = context.resources["Buffer For Machined parts"]
            for field in released_buffer_fields:
                buffer.valuation[field] = None
            if buffer_zone is not None:
                buffer.valuation[f"zone_{buffer_zone}_part"] = part
            buffer.revision += 1
            buffer.evidence = "resource"
        context.revision += 1
        runtime.agent.part_tracker = deepcopy(context.part_tracker)
        context.transitions.append({
            "recovery_task_id": task_id, "safety_status": "skipped",
            "acknowledgement": {
                "task_id": task_id, "run_id": context.run_id, "resource_id": owner.agent_name,
                "event_id": task["outline_id"], "event_name": task.get("event_name", task["macro_name"]),
                "parameters": {**deepcopy(task.get("task_params", {})), "part_name": part},
                "status": "completed", "evidence": "resource",
            },
            "before": before, "after": context.snapshot(),
            "product_before": products_before, "product_after": deepcopy(context.part_tracker),
            "observations": deepcopy(observed), "controller_result": deepcopy(result),
        })
    return observed


def _guard_holds(values: dict, field: str, rule: dict, params: dict) -> bool:
    # The same typed comparison the environment model uses when it projects an event.
    if len(rule) != 1:
        return False
    operator, expected = next(iter(rule.items()))
    if operator.endswith("_from_param"):
        expected = params.get(expected)
    value = values.get(field)
    equal = field in values and type(value) is type(expected) and value == expected
    if operator in {"equals", "equals_from_param"}:
        return equal
    return operator in {"not_equals", "not_equals_from_param"} and not equal


def _declared_consequence(context, event_names: set[str], location_field: str, location: str,
                          part: str) -> tuple[str, dict] | None:
    """Return an owner's own declared consequence of a part being taken from, or put at, a location.

    The consequence is read from the owner's recorded handoff events, whichever
    robot they were declared for, with ``{part_name}`` fields bound to the part.
    """
    from cais_spade_llm.resources.environment_models import _bound

    params = {"part_name": part}
    for rid, model in context.models.items():
        resource = context.resources.get(rid)
        if resource is None:
            continue
        for event in model.get("events", []):
            bindings = event.get("parameter_bindings", {})
            if (event.get("event_name") not in event_names
                    or bindings.get(location_field) != {"equals": location}
                    or bindings.get("resource_id", {}).get("equals") == rid):
                continue
            if all(_guard_holds(resource.valuation, _bound(field, params), rule, params)
                   for field, rule in event.get("guards", {}).items()):
                return rid, {_bound(field, params): params[rule["set_from_param"]] if "set_from_param" in rule
                             else deepcopy(rule["set"]) for field, rule in event.get("updates", {}).items()}
    return None


def _observed_release(context, part: str, pose: list, settled: bool) -> tuple[str | None, dict, dict]:
    """Assign custody of a released part from where it was observed to rest.

    Returns:
        The receiving resource (None for a surface no receiver declares, or a
        part still moving when observation ended), its declared updates, and
        the part's tracker entry with its observed pose.
    """
    from cais_spade_llm.resources.environment_models import product_state_from_custody
    from cais_spade_llm.resources.environment_recovery import observed_receiver, storage_placement_conditions
    from cais_spade_llm.resources.nominal_conveyor import conveyor_place_parameters

    resting = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), map(float, pose), strict=True))
    entry = deepcopy(context.part_tracker[part])
    receiver = observed_receiver(context.inputs["scene"], context.models.get("Conveyor"), part, pose) if settled else None
    if receiver is None:
        # Like a dropped part: located only by its observed pose until picked again.
        entry.update(location=None, observed_pose=resting)
        return None, {}, entry
    rid, area = receiver
    values = context.resources[rid].valuation
    if rid == "Conveyor":
        if values.get("belt_stopped") is not True or values.get("loading_reserved_by") is not None:
            raise ValueError("The Conveyor must be stopped and unreserved to receive a part")
        order = conveyor_place_parameters(context.models[rid], values, part, area)
        updates = {f"part_location.{part}": area,
                   **{key.removeprefix("next_"): value for key, value in order.items()}}
    elif rid == "Buffer For Machined parts":
        if values.get(area) is not None:
            raise ValueError(f"{rid} {area} is already occupied")
        updates = {area: part}
    elif rid == "Storage":
        conditions = storage_placement_conditions(context.models[rid], context.inputs["scene"], values, part)
        updates = {field: rule["set"] for field, rule in conditions["updates"].items()}
    else:
        declared = _declared_consequence(context, {"place_release"}, "destination_location", area, part)
        if declared is None or declared[0] != rid:
            raise ValueError(f"{area} does not accept {part} in its current state")
        updates = declared[1]
    entry.update(**product_state_from_custody(part, rid, {**values, **updates}, context.product_name),
                 observed_pose=resting)
    return rid, updates, entry


async def _reconcile_mobile_completion(runtime, owner, task: dict, task_id: str, result: dict) -> dict:
    """Commit a KMR macro from its own measured base pose, custody and released part.

    A picked part's source and a released part's receiver come from the owners'
    declared consequences and from where the part was observed, not from the
    generated end state.
    """
    context = runtime.context
    state = deepcopy(getattr(owner, "_primitive_state", {}))
    actor = context.resources[owner.agent_name]
    declared = context.models[owner.agent_name]["state_variables"]
    values = deepcopy(actor.valuation)
    held_before = values.get("held_part")
    for field in ("held_part", "resource_location"):
        if field in state:
            values[field] = deepcopy(state[field])
    held_after = values.get("held_part")
    end = task.get("expected_end_state") or {}
    if "held_part" in end and end["held_part"] != held_after:
        raise ValueError("Observed recovery custody differs from the generated end state")
    if "domain" in declared.get("resource_location", {}) and values.get("resource_location") not in declared["resource_location"]["domain"]:
        raise ValueError(f"Recovery cannot resume from resource_location={values.get('resource_location')}")
    if held_before is not None and held_after is not None and held_after != held_before:
        raise ValueError("A KMR macro cannot exchange one held part for another")
    released = None
    if held_before is not None and held_after is None:
        record = getattr(owner, "_released_part", None) or {}
        if record.get("part_name") != held_before or not record.get("pose"):
            raise ValueError("A KMR release requires the observed resting place of the part")
        released = (held_before, *_observed_release(context, held_before, record["pose"], record.get("settled") is True))
    picked = source = None
    if held_before is None and held_after is not None:
        picked = held_after
        if task.get("part_name") not in (None, picked):
            raise ValueError(f"The KMR picked {picked}, not the generated event's {task['part_name']}")
        entry = deepcopy(context.part_tracker[picked])
        # A part located only by its observed pose has no owner to update.
        if entry["location"] is not None:
            source = _declared_consequence(context, {"pick_grasp", "pick_part"}, "origin_resource_location",
                                           entry["location"], picked)
            if source is None:
                raise ValueError(f"{entry['location']} declares no consequence of {picked} being taken")
        entry.update(location=owner.agent_name, last_task=task_id)
        entry.pop("observed_pose", None)
    values["resource_state"] = "carrying" if held_after else "idle"
    if "approached_part" in declared:
        values["approached_part"] = None
    observed = {"base_pose": deepcopy(state.get("base_pose")), "resource_location": values.get("resource_location"),
                "held_part": held_after}
    if released is not None:
        observed["released"] = {"part_name": released[0], "receiver": released[1],
                                "observed_pose": deepcopy(released[3]["observed_pose"])}
    with context.admission_lock:
        before, products_before = context.snapshot(), deepcopy(context.part_tracker)
        actor.valuation = values
        actor.revision += 1
        actor.evidence = "resource"
        changes = [source] if source is not None else []
        if released is not None and released[1] is not None:
            changes.append((released[1], released[2]))
        for rid, updates in changes:
            context.resources[rid].valuation.update(updates)
            context.resources[rid].revision += 1
            context.resources[rid].evidence = "resource"
        if picked is not None:
            context.part_tracker[picked] = entry
        if released is not None:
            context.part_tracker[released[0]] = {**released[3], "last_task": task_id}
        context.revision += 1
        runtime.agent.part_tracker = deepcopy(context.part_tracker)
        context.transitions.append({
            "recovery_task_id": task_id, "safety_status": "skipped",
            "acknowledgement": {
                "task_id": task_id, "run_id": context.run_id, "resource_id": owner.agent_name,
                "event_id": task["outline_id"], "event_name": task.get("event_name", task["macro_name"]),
                "parameters": {**deepcopy(task.get("task_params", {})), "part_name": picked or task.get("part_name")},
                "status": "completed", "evidence": "resource",
            },
            "before": before, "after": context.snapshot(),
            "product_before": products_before, "product_after": deepcopy(context.part_tracker),
            "observations": deepcopy(observed), "controller_result": deepcopy(result),
        })
    owner._released_part = None
    return observed
