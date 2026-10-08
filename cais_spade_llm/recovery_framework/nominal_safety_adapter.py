from __future__ import annotations

"""Prepare saved nominal robot programs without executing their physical steps."""

import math
from copy import deepcopy
from dataclasses import asdict

from cais_spade_llm.resources.resource_safety_preparation import fingerprint

_MOTION = {"move_cartesian", "move_relative", "move_to_named_pose"}
_READS = {"detect_parts", "compute_pick_targets", "compute_place_targets", "get_current_pose"}
_PREDICATES = {"", "always", "held_part_empty", "held_part_exists",
               "task_ctx_key_truthy", "move_insert_required", "execution_mode",
               "named_pose_available"}
_POSE_KEYS = ("x", "y", "z", "qx", "qy", "qz", "qw")


def _finite(values) -> None:
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Nominal motion requires finite numeric values")


def resolve_motion(primitive: str, params: dict, start_pose: list, configuration: dict) -> dict:
    """Resolve supported resource operations to their configured Cartesian command.

    This function performs no planning or execution. Named configurations are
    accepted only for the native Cartesian mode and its fixed-orientation path.
    """
    _finite(start_pose)
    if len(start_pose) != 7:
        raise ValueError("Nominal motion requires a complete observed start pose")
    if primitive == "move_cartesian":
        return deepcopy(params)
    if primitive == "move_relative":
        if set(params) - {"dx", "dy", "dz", "speed"}:
            raise ValueError("Unknown move_relative parameters")
        offsets = [params.get(key, 0.0) for key in ("dx", "dy", "dz")]
        _finite(offsets)
        resolved = dict(zip(_POSE_KEYS, [*[start_pose[i] + offsets[i] for i in range(3)],
                                        *start_pose[3:]], strict=True))
        if "speed" in params:
            resolved["speed"] = params["speed"]
        return resolved
    if primitive != "move_to_named_pose":
        raise ValueError(
            f"{primitive} has no complete native continuous preparation contract; "
            "gripper geometry, custody timing and placement corrections require owner evidence"
        )
    if set(params) - {"pose_name", "speed"} or not isinstance(params.get("pose_name"), str):
        raise ValueError("Unknown move_to_named_pose parameters")
    settings = configuration["controller_configuration"]
    cartesian = settings.get("cartesian_motion", {})
    if settings.get("retain_observed_clear_pose_as_home") is True:
        raise ValueError("Conditional retention of a clearance pose requires its own preparation contract")
    positions = configuration["named_positions"].get(params["pose_name"])
    names = configuration["joint_names"]
    if not isinstance(positions, (list, tuple)) or len(positions) != len(names):
        raise ValueError("Configured named joint positions are unavailable")
    _finite(positions)
    from cais_spade_llm.resources.continuous_motion import (
        interval, kinematic_transforms, representative_pose,
    )

    geometry = configuration["continuous_motion"]
    transforms = kinematic_transforms(
        geometry, {name: interval(value) for name, value in zip(names, positions, strict=True)})
    target = representative_pose(transforms[geometry["reference_link"]])
    if cartesian.get("home_preserve_orientation") is True:
        from cais_spade_llm.recovery_framework.geometry import rotate

        if rotate(start_pose[3:], [0.0, 0.0, 1.0])[2] > -math.cos(0.02):
            raise ValueError("Cartesian home requires the configured downward gripper")
        target[3:] = start_pose[3:]
    elif abs(sum(a * b for a, b in zip(target[3:], start_pose[3:], strict=True))) < 1 - 1e-10:
        raise ValueError("Named pose changes orientation; its native waypoint contract is unavailable")
    else:
        target[3:] = start_pose[3:]
    # The native Cartesian named-pose branch uses the configured time scale,
    # independently of the joint-duration argument calculated by its caller.
    return dict(zip(_POSE_KEYS, target, strict=True))


def model_configuration(controller) -> dict:
    """Return the exact owner configuration required by motion lowering."""
    return {
        "joint_names": list(controller.arm_joint_names),
        "controller_configuration": deepcopy(controller.controller_config),
        "continuous_motion": deepcopy(controller.recovery_safety_observer.motion_configuration),
        "named_positions": deepcopy(controller.named_positions),
    }


def prepare_step(controller, *, primitive: str, params: dict, start: dict, binding: dict) -> dict:
    """Prepare a supported native motion while preserving its original operation."""
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import prepare_ur5e_motion

    try:
        if primitive == "move_to_named_pose" and start["attachment"]["model_name"] is not None:
            raise ValueError("Named-pose preparation requires the configured empty gripper")
        resolved = resolve_motion(primitive, params, start["current_pose"], model_configuration(controller))
        record = prepare_ur5e_motion(controller, primitive="move_cartesian", params=resolved,
                                    start=start, binding=binding)
        record.update(primitive=primitive, params=deepcopy(params),
                      execution_kind="arm_joint_trajectory",
                      resolved_motion={"primitive": "move_cartesian", "params": resolved})
        if record["status"] == "prepared" and primitive == "move_to_named_pose":
            settings = controller.controller_config.get("cartesian_motion", {})
            if settings.get("home_preserve_orientation") is not True:
                endpoint = record["joint_trajectory"]["points"][-1]["positions"]
                expected = controller.named_positions[params["pose_name"]]
                if any(abs(math.remainder(a - b, 2 * math.pi)) > 0.02
                       for a, b in zip(endpoint, expected, strict=True)):
                    raise ValueError("Cartesian endpoint does not reach the configured named joint posture")
        record.pop("preparation_id", None)
        record["preparation_id"] = fingerprint(record)
        return record
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"status": "NEEDS_CONTEXT", "primitive": primitive, "params": deepcopy(params),
                "start": deepcopy(start), "binding": deepcopy(binding),
                "command_sent": False, "reason": str(exc)}


def motion_effects(*, primitive, params, evidence, start_time, end_time, configuration) -> dict:
    """Validate original operation semantics against its exact continuous command."""
    from cais_spade_llm.resources.continuous_motion import ContinuousMotion
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import _continuous_motion_effects

    raw = evidence["continuous_motion"]
    motion = ContinuousMotion(raw["joint_trajectory"], raw["configuration"])
    resolved = resolve_motion(primitive, params, motion.reference_pose(0), configuration)
    native = evidence["resolved_motion"]
    if native["primitive"] != "move_cartesian":
        raise ValueError("Native motion lowering changed")
    # FK roundoff may differ from the observed pose used by preparation.
    expected, actual = deepcopy(resolved), deepcopy(native["params"])
    if set(expected) != set(actual) or any(
        not math.isclose(expected[key], actual[key], abs_tol=1e-6, rel_tol=0)
        if type(expected[key]) in (int, float) else expected[key] != actual[key]
        for key in expected
    ):
        raise ValueError("Native Cartesian command differs from the original operation")
    return _continuous_motion_effects(
        primitive="move_cartesian", params=actual, evidence=evidence,
        start_time=start_time, end_time=end_time, configuration=configuration)


def _guard(guard, owner, args: dict, state: dict, outputs: dict) -> bool:
    from cais_spade_llm.resources.robot.robot_task_model import _evaluate_guard

    if guard.condition:
        if guard.condition.get("operator", "equals") not in {"equals", "not_equals", "exists"}:
            raise ValueError("Saved nominal guard has an unsupported condition operator")
    elif guard.predicate not in _PREDICATES:
        raise ValueError("Saved nominal guard has no registered predicate")
    return _evaluate_guard(guard, agent=owner, args=args,
                           runtime_state=state, step_outputs=outputs)


class NominalRobotAdapter:
    """Bind full saved nominal programs and observed completion to one owner."""

    def __init__(self, owner, scene: dict) -> None:
        self.owner = owner
        self.scene = deepcopy(scene)
        self.programs: dict[str, dict] = {}

    def prepare(self, task: dict) -> dict:
        """Expand every active saved step, including steps hidden from recovery."""
        from cais_spade_llm.resources.gazebo_programs import saved_robot_definition
        from cais_spade_llm.resources.robot.robot_task_model import _resolve_value
        from cais_spade_llm.resources.robot.robot_task_runtime import (
            _build_runtime_state, _primitive_payload_from_result,
        )

        if task["resource_id"] != self.owner.agent_name:
            raise ValueError("Nominal task names another resource")
        if getattr(self.owner, "execution_mode", None) != "simulation":
            raise ValueError("Nominal physical adapter requires simulation owner contracts")
        if getattr(self.owner, "gazebo_program_scene", self.scene) != self.scene:
            raise ValueError("Saved nominal program configuration changed")
        definition = saved_robot_definition(self.scene, self.owner.agent_name, task["event_name"])
        args = {row.name: deepcopy(row.default) for row in definition.arguments if row.default is not None}
        args.update(deepcopy(task["parameters"]))
        if set(args) - {row.name for row in definition.arguments}:
            raise ValueError("Nominal task contains undeclared function arguments")
        if any(row.required and args.get(row.name) is None for row in definition.arguments):
            raise ValueError("Nominal task is missing required function arguments")
        state, outputs, steps, originals = _build_runtime_state(self.owner), {}, [], []
        for guard in definition.program.entry_guards:
            if not _guard(guard, self.owner, args, state, outputs):
                raise ValueError("Nominal saved program entry guard is false")
        for step in definition.program.steps:
            if not all(_guard(guard, self.owner, args, state, outputs) for guard in step.when):
                continue
            params = _resolve_value(step.params, args=args, runtime_state=state, step_outputs=outputs)
            params = {key: value for key, value in params.items() if not key.startswith("_")}
            if step.executor != "primitive":
                raise ValueError("Saved nominal executor has no physical preparation contract: " + step.executor)
            if step.op in _READS:
                if steps:
                    raise ValueError("A nominal helper after motion requires a fresh staged checkpoint: " + step.id)
                result = getattr(self.owner._controller, step.op)(**deepcopy(params))
                if not isinstance(result, dict) or result.get("success") is not True:
                    raise ValueError("Nominal helper observation unavailable: " + step.id)
                if step.store_as:
                    outputs[step.store_as] = _primitive_payload_from_result(
                        step=step, params=params, primitive_result=result, runtime_state=state)
                continue
            if step.op not in _MOTION:
                raise ValueError(
                    "Saved nominal step requires a complete native preparation contract: " + step.op)
            settings = self.owner._controller.controller_config.get("cartesian_motion", {})
            if ((step.id == "move_above_part" and settings.get("pick_transit_waypoints"))
                    or (step.id == "move_above_destination" and settings.get("transit_waypoints"))
                    or step.id == "retreat_from_source"):
                raise ValueError("Saved nominal routed motion needs its full waypoint preparation: " + step.id)
            steps.append({"primitive": step.op, "params": params,
                          "source": {"function_name": definition.name, "step_id": step.id,
                                     "saved_program_fingerprint": fingerprint(asdict(definition.program))}})
            originals.append(step)
        if not steps:
            raise ValueError("Nominal program has no positively timed modeled native motion")
        key = fingerprint(task)
        self.programs[key] = {"task": deepcopy(task), "definition": definition,
            "args": args, "initial_state": state, "outputs": outputs,
            "primitive_steps": deepcopy(steps), "originals": originals, "completed": False}
        return {"primitive_steps": steps}

    def complete(self, task: dict, results: list) -> dict:
        """Commit task effects only after authenticated owner command completion."""
        from cais_spade_llm.resources.robot.robot_task_model import _resolve_value
        from cais_spade_llm.resources.robot.robot_task_runtime import (
            _apply_effect, _build_runtime_state, _commit_runtime_state, _primitive_payload_from_result,
        )

        record = self.programs[fingerprint(task)]
        if record["task"] != task or record["completed"]:
            raise ValueError("Nominal task changed or completion was already consumed")
        if len(results) != len(record["primitive_steps"]) or not all(row.get("success") is True for row in results):
            raise ValueError("Nominal physical completion is incomplete")
        provider = self.owner.recovery_composition_evidence_provider
        ledger = provider.ledger
        with ledger.lock:
            for result, step in zip(results, record["primitive_steps"], strict=True):
                matching = [row for row in ledger.commands.values()
                    if row["task_id"] == task["task_id"] and row["status"] == "completed"
                    and row["owner_identity"].get("preparation_id") == result.get("preparation_id")
                    and row["command"]["primitive"] == step["primitive"]
                    and row["command"]["params"] == step["params"]
                    and row["command"]["joint_trajectory"] == result.get("joint_trajectory")
                    and row.get("reservation_token") and row.get("controller_goal_id") is not None
                    and row["observations"] == result.get("observations")]
                if len(matching) != 1:
                    raise ValueError("Nominal completion lacks an authenticated owner ledger result")
        if getattr(self.owner, "gazebo_program_scene", self.scene) != self.scene:
            raise ValueError("Saved nominal program configuration changed")
        current = self.owner._controller.capture_recovery_safety_state()
        for field in ("joint_names", "joint_positions", "current_pose", "attachment", "launch_id", "frame"):
            if current[field] != results[-1]["observations"][field]:
                raise ValueError("Nominal final physical observation changed: " + field)
        if _build_runtime_state(self.owner) != record["initial_state"]:
            raise ValueError("Nominal owner task state changed during physical execution")
        state, outputs = deepcopy(record["initial_state"]), deepcopy(record["outputs"])
        for result, step, original in zip(results, record["primitive_steps"], record["originals"], strict=True):
            observed = result["observations"]
            actual = {**result, "absolute_position": dict(zip(_POSE_KEYS, observed["current_pose"], strict=True))}
            if original.store_as:
                outputs[original.store_as] = _primitive_payload_from_result(
                    step=original, params=step["params"], primitive_result=actual, runtime_state=state)
            state["_position"] = actual["absolute_position"]
        for effect in record["definition"].program.effects:
            if all(_guard(guard, self.owner, record["args"], state, outputs) for guard in effect.when):
                _apply_effect(effect, args=record["args"], runtime_state=state, step_outputs=outputs)
        response = _resolve_value(record["definition"].program.success_response,
                                  args=record["args"], runtime_state=state, step_outputs=outputs)
        if not isinstance(response, dict):
            response = {"content": str(response or "")}
        _commit_runtime_state(self.owner, state)
        record["completed"] = True
        return {**response, "status": "completed", "task_id": task["task_id"],
                "physical_steps": deepcopy(results)}


def _install_motion_contract(controller) -> None:
    from types import MethodType

    from cais_spade_llm.resources.resource_safety_preparation import PrimitiveModel, validate_prepared_start

    original_model = controller.get_recovery_safety_primitive_model

    def model(bound):
        observer = getattr(bound, "recovery_safety_observer", None)
        if observer is None or observer.motion_configuration is None:
            return original_model()
        return PrimitiveModel("joint_trajectory_continuous", 2,
                              model_configuration(bound), motion_effects)

    def prepare_program(bound, program: dict, checkpoint: dict, *, resource_jid: str) -> dict:
        rid = program["resource_id"]
        result = {"resource_id": rid, "resource_jid": resource_jid,
                  "program": deepcopy(program), "program_fingerprint": fingerprint(program),
                  "checkpoint_id": checkpoint["checkpoint_id"], "steps": [], "status": "NEEDS_CONTEXT"}
        start = deepcopy(checkpoint["observations"][rid]["physical"])
        start["model_execution"] = checkpoint.get("model_execution") is True
        if start["model_execution"]:
            start["model_execution_regions"] = deepcopy(checkpoint.get("geometry", {}).get("regions"))
        observed = bound.capture_recovery_safety_state()
        if not start["model_execution"]:
            for field in ("joint_names", "joint_positions", "current_pose", "attachment", "launch_id", "frame"):
                if observed[field] != start[field]:
                    return {**result, "reason": "Checkpoint changed before planning: " + field}
        for index, step in enumerate(program["primitive_steps"]):
            binding = {"checkpoint_id": checkpoint["checkpoint_id"], "resource_id": rid,
                       "resource_jid": resource_jid, "program_fingerprint": result["program_fingerprint"],
                       "step_index": index, "source": deepcopy(step.get("source", {})),
                       "run_id": checkpoint["runtime"]["run_id"], "launch_id": start["launch_id"],
                       "model_execution": start["model_execution"]}
            if start["model_execution"]:
                binding.update(model_execution_regions=deepcopy(start["model_execution_regions"]),
                               stationary_contract=deepcopy(start.get("stationary_contract")))
            record = prepare_step(bound, primitive=step["primitive"], params=step["params"],
                                  start=start, binding=binding)
            result["steps"].append(record)
            if record["status"] != "prepared":
                return {**result, "reason": record["reason"]}
            if index == 0 and start["model_execution"]:
                try:
                    validate_prepared_start(observed, start, continuous_motion=record.get("continuous_motion"))
                except (ValueError, KeyError, TypeError) as exc:
                    return {**result, "reason": str(exc)}
            start["joint_positions"] = deepcopy(record["joint_trajectory"]["points"][-1]["positions"])
            start["current_pose"] = deepcopy(record["target_pose"])
        return {**result, "status": "prepared"}

    controller.get_recovery_safety_primitive_model = MethodType(model, controller)
    controller.prepare_recovery_safety_program = MethodType(prepare_program, controller)


def install_nominal_robot_adapter(owner, scene: dict) -> NominalRobotAdapter:
    """Install internal saved-program hooks without changing the public bridge."""
    previous = getattr(owner, "_nominal_safety_adapter", None)
    if previous is not None:
        if previous.scene != scene:
            raise ValueError("Nominal safety adapter configuration changed")
        return previous
    adapter = NominalRobotAdapter(owner, scene)
    _install_motion_contract(owner._controller)
    owner._nominal_safety_adapter = adapter
    owner.prepare_nominal_safety_program = adapter.prepare
    owner.observe_prepared_nominal_completion = adapter.complete
    return adapter
