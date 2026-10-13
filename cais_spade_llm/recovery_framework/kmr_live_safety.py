from __future__ import annotations

"""Registered KMR observations and prepared-motion owner adapter."""

import asyncio
import hashlib
from pathlib import Path
from copy import deepcopy

from cais_spade_llm.recovery_framework.kmr_safety_execution import kmr_motion_effects
from cais_spade_llm.resources.resource_safety_preparation import (
    PreparedRobotEvidence,
    PrimitiveModel,
    validate_prepared_start,
)


def configure_observer(owner, observer, record: dict) -> None:
    """Bind KMR geometry to its configured URDF and controller interpolation."""
    from cais_spade_llm.resources.continuous_geometry import load_continuous_geometry

    config = observer.configuration
    reader = observer.provider.reader
    scene = observer.provider.runtime.context.inputs["scene"]["KMR"]
    description = reader.parameter(config["description_node"], "robot_description")
    interpolation = reader.parameter(config["controller_node"], "interpolation_method")
    model = record["models"][config["model"]]
    observer.motion_configuration = load_continuous_geometry(
        description, root=config["root_link"], reference_link=config["reference_link"],
        joint_names=scene["arm_joint_names"], observed_joints=model["joints"],
        root_pose=model["links"][config["root_link"]]["pose"], interpolation=interpolation)
    owner.recovery_safety_observer = observer
    owner.recovery_safety_description = description


def capture_state(owner, *, max_age: float = 2.) -> dict:
    """Read KMR tool, base, joints, and custody from the configured physics owner."""
    from cais_spade_llm.resources.continuous_motion import interval, kinematic_transforms, representative_pose

    observer = owner.recovery_safety_observer
    if observer.motion_configuration is None:
        raise ValueError(observer.support_error or "KMR continuous geometry unavailable")
    reader = observer.provider.reader
    description = reader.parameter(observer.configuration["description_node"], "robot_description")
    interpolation = reader.parameter(observer.configuration["controller_node"], "interpolation_method")
    if (hashlib.sha256(description.encode()).hexdigest() != observer.motion_configuration["description_sha256"]
            or interpolation != observer.motion_configuration["interpolation"]):
        raise ValueError("KMR geometry or controller interpolation changed")
    record = reader.snapshot(max_age=max_age)
    model = record["models"][observer.configuration["model"]]
    from cais_spade_llm.resources.continuous_geometry import observed_joint_configuration

    config = observed_joint_configuration(observer.motion_configuration, model["joints"],
        model["links"][observer.motion_configuration["root"]]["pose"])
    matrices = kinematic_transforms(config, {name: interval(value["position"])
                                           for name, value in model["joints"].items()})
    names = config["joint_names"]
    native = {"joint_names": list(names), "joint_positions": [model["joints"][name]["position"] for name in names],
              "base_pose": deepcopy(model["links"][config["root"]]["pose"]),
              "current_pose": representative_pose(matrices[config["reference_link"]]),
              "tool_pose_source": "GETRECOVERYSTATE joints and configured FK"}
    native["native_physics_source"] = {key: record[key] for key in
                                       ("instance_id", "simulation_time", "observed_monotonic")}
    native["continuous_motion_configuration"] = deepcopy(config)
    # The base service records its own command state; zero measured velocity
    # alone cannot establish a future fixed-base interval.
    hold = {"version": 2, "status": "NEEDS_CONTEXT"}
    try:
        from ament_index_python.packages import get_package_prefix

        base = reader.controller_state("/KMR")
        hold["controller_observation"] = deepcopy(base)
        installed = Path(get_package_prefix("cais_lab_robotics")) / "lib/cais_lab_robotics/kmr_base_controller.py"
        code_sha = hashlib.sha256(installed.read_bytes()).hexdigest()
        hold["controller_source_sha256"] = code_sha
        if base.get("controller_source_sha256") != code_sha:
            raise ValueError("KMR base controller code differs from its captured source identity")
        link = model["links"][config["root"]]
        physics_node = observer.configuration["physics_description_node"]
        physics_description = reader.parameter(physics_node, "robot_description")
        hold.update({
            "robot_description": description, "root_link": config["root"], "root_pose": deepcopy(link["pose"]),
            "physics_robot_description": physics_description, "physics_description_node": physics_node,
            "physics_description_sha256": hashlib.sha256(physics_description.encode()).hexdigest(),
            "linear_velocity": deepcopy(link["linear_velocity"]), "angular_velocity": deepcopy(link["angular_velocity"]),
            "physics_source": deepcopy(native["native_physics_source"]), "max_age": max_age,
            "launch_id": observer.provider.launch_id})
        from cais_spade_llm.agents.central_controller.recovery_case_evidence import recorded_base_hold

        recorded_base_hold(resource_id="KMR", physical_evidence={"resources":{"KMR":{"base_hold_evidence":hold}}}, model=config)
        hold["status"] = "supported"
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        # Failed hold validation does not erase the captured moving controller
        # state. Later calculations still need its exact command and identity.
        hold.update(status="NEEDS_CONTEXT", reason=str(exc))
    native["base_hold_evidence"] = hold
    captured = observer.capture(max_age=max_age, native=native, record=record)
    captured["observation_state"]["base_pose"] = deepcopy(native["base_pose"])
    return captured


def primitive_model(owner):
    """Return the configured KMR continuous-motion semantics."""
    observer = getattr(owner, "recovery_safety_observer", None)
    if observer is None or observer.motion_configuration is None:
        return None
    from cais_spade_llm.resources.robot.gazebo_pick_place_controller import _outline_custody_configuration

    configuration = {"continuous_motion": deepcopy(observer.motion_configuration)}
    scene = owner.environment_runtime.context.inputs["scene"]["KMR"]
    execution = scene["task_execution"]
    description = getattr(owner, "recovery_safety_description", None)
    if description is not None:
        widths = {"default": execution["closed_gripper_width_m"],
                  **execution.get("closed_gripper_width_m_by_part", {})}
        contracts = {}
        for part, width in widths.items():
            contract = _outline_custody_configuration(description,
                {"joint": scene["gripper_joint"], "open": scene["gripper_stroke_m"], "closed": width},
                observer.motion_configuration, enabled=True)
            if contract is not None:
                contract.update(model_name="KMR", attachment_link=execution["attach_link"],
                    reference_link=execution["tcp_link"], part_link="link",
                    source="KMR.task_execution and captured robot_description",
                    simulation_assumptions=execution["simulation_assumptions"])
                contracts[part] = contract
        configuration["custody_by_part"] = contracts
    return PrimitiveModel("kmr_joint_trajectory_continuous", 2, configuration, kmr_motion_effects)


def _worker_request(owner) -> dict:
    runtime = owner.environment_runtime
    if runtime.stopped:
        raise ValueError("KMR physical execution runtime is stopped")
    return {"inputs": runtime.context.inputs, "valuation": runtime.context.snapshot(),
            "geometry": deepcopy(runtime.context.geometry),
            "probe": getattr(runtime, "kmr_probe", None) or getattr(runtime, "prepared", {}).get("probe"),
            "custody": deepcopy(getattr(owner, "workflow_custody", None))}


async def prepare_program(owner, program: dict, checkpoint: dict) -> dict:
    """Ask the original ROS worker to retain a complete native primitive program."""
    if program["resource_id"] != owner.agent_name:
        raise ValueError("KMR preparation belongs to another resource")
    observer = owner.recovery_safety_observer
    start = checkpoint["observations"][owner.agent_name]["physical"]
    result = await owner.worker.run({**_worker_request(owner), "mode": "safety_prepare",
        "program": deepcopy(program), "start": deepcopy(start),
        "continuous_motion": deepcopy(observer.motion_configuration), "resource_jid": str(owner.jid),
        "binding": {"checkpoint_id": checkpoint["checkpoint_id"],
                    "run_id": owner.environment_runtime.context.run_id,
                    "resource_id": owner.agent_name, "resource_jid": str(owner.jid)}})
    prepared = result["preparation"]
    provider = getattr(owner, "recovery_composition_evidence_provider", None)
    if isinstance(provider, KMRPreparedEvidence):
        provider.register_program(prepared)
    return prepared


def prepare_program_sync(owner, program: dict, checkpoint: dict) -> dict:
    """Support the existing worker-thread preparation API without nested loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(prepare_program(owner, program, checkpoint))
    raise ValueError("KMR preparation requires the asynchronous owner API on the application loop")


class KMRPreparedEvidence(PreparedRobotEvidence):
    """Execute admitted KMR motion through the same persistent ROS worker."""

    async def execute_async_step(self, *, resource_agent, task_id: str, step_index: int,
                                 primitive: str, params: dict, grant: dict) -> dict:
        """Consume one command once and retain its observed execution outcome."""
        if resource_agent is not self.owner:
            raise ValueError("KMR prepared executor owner changed")
        with self.ledger.lock:
            prepared = self.preparations[grant["preparation_id"]]
            if task_id != prepared["request"]["task_id"] or step_index in prepared["executed"]:
                raise ValueError("KMR prepared task changed or command was consumed")
            planned = deepcopy(prepared["steps"][step_index])
            if planned["primitive"] != primitive or planned["params"] != params:
                raise ValueError("KMR dispatch parameters differ from its preparation")
            command_id = prepared["result"]["command_ids"][step_index]
            command = {"primitive": primitive, "params": params,
                       "joint_trajectory": planned["joint_trajectory"]}
            self.ledger.require_authorized(command_id, command=command)
            prepared["executed"].add(step_index)
            self.ledger.start(command_id, command=command, controller_goal_id=None)
            self.dispatch_command = command_id
        try:
            observed = capture_state(self.owner, max_age=0.)
            validate_prepared_start(observed, planned["start"])
            response = await self.owner.worker.run({**_worker_request(self.owner),
                "mode": "safety_execute", "preparation_id": planned["preparation_id"],
                **command})
            result = response["result"]
            goals = result.get("controller_goals", [])
            if len(goals) != 1 or not goals[0].get("goal_id"):
                raise ValueError("KMR execution did not retain its accepted controller goal")
            self.ledger.bind_goal(command_id, list(bytes.fromhex(goals[0]["goal_id"])))
            result["observations"] = capture_state(self.owner, max_age=0.)
        except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            result = {"success": False, "command_id": command_id, "error": str(exc)}
            self.ledger.finish(command_id, success=False, observations={"error": str(exc)})
            self._publish(task_id, step_index, result)
            raise
        finally:
            self.dispatch_command = None
        self.ledger.finish(command_id, success=result.get("success") is True,
                           observations=result["observations"])
        self.owner.record_primitive_evidence(response)
        self._publish(task_id, step_index, result)
        return result

    def request_guard(self, request: dict) -> None:
        """Allow only preparation/read requests or one exact authorized command."""
        if request.get("mode") in {"probe", "safety_prepare"}:
            return
        with self.ledger.lock:
            row = self.ledger.commands.get(self.dispatch_command)
            if (request.get("mode") != "safety_execute" or row is None
                    or row["status"] != "active" or not row.get("reservation_token")
                    or request.get("preparation_id") != row["owner_identity"]["preparation_id"]
                    or any(request.get(key) != value for key, value in row["command"].items())):
                raise ValueError("KMR request lacks its exact live physical grant")
