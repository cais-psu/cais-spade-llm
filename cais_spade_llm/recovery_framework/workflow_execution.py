"""Execution adapters for the full recovery-framework manufacturing order."""

from __future__ import annotations

import asyncio
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.gazebo_worker import GazeboExecutionError, GazeboWorker
from cais_spade_llm.recovery_framework.kmr_agent import KMRResourceAgent
from cais_spade_llm.resources.robot.robot_task_registry import robot_task_registry

_UR_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class WorkflowResourceAgent(ResourceAgent):
    """Own a persistent Gazebo worker for one non-robot workflow resource."""

    def __init__(self, jid: str, password: str, *, name: str, worker: GazeboWorker | None, **kwargs):
        super().__init__(jid, password, name=name, **kwargs)
        self.workflow_worker = worker

    async def teardown(self) -> None:
        """Cancel owned Gazebo execution before the agent is removed."""
        if self.workflow_worker is not None:
            await self.workflow_worker.cancel()


def _robot_configuration(scene: dict, robot: dict) -> tuple[dict, dict, dict]:
    base_path = ROOT / "cais_spade_llm/initialization/resources/robot_ur5e.json"
    base = json.loads(base_path.read_text())["ur5e"]["gazebo"]
    prefix = robot["prefix"]
    controller = deepcopy(base["controller"])
    controller["node_name"] = f"{prefix.rstrip('_')}_workflow_controller"
    controller["arm_joint_names"] = [prefix + name for name in _UR_JOINTS]
    controller["arm_trajectory_topic"] = (
        f"/{prefix}joint_trajectory_controller/joint_trajectory"
    )
    controller["move_group"].update(
        group_name=f"{prefix}ur_manipulator",
        ee_link=f"{prefix}tool0",
        tcp_link=f"{prefix}rg2_gripper_tcp",
    )
    controller["gripper"].update(
        joint=f"{prefix}rg2_finger_width",
        topic=f"/{prefix}rg2_gripper_traj_controller/joint_trajectory",
    )
    attach_links = [
        f"{prefix}rg2_gripper_tcp",
        f"{prefix}tool0",
        f"{prefix}wrist_3_link",
    ]
    controller["attach"].update(
        attach_link_candidates=attach_links,
        release_detach_link_candidates=attach_links,
        primary_attach_link=attach_links[0],
    )
    controller["cartesian_motion"] = deepcopy(robot.get("cartesian_motion", {}))
    controller["motion"]["trajectory_time_scale"] = 1.0
    controller["motion"]["tf_lookup_timeout_sec"] = float(
        robot.get("tf_lookup_timeout_sec", controller["motion"].get("tf_lookup_timeout_sec", 2.0))
    )
    controller["payload_collision"] = {
        "enabled": True,
        "robot_prefix": prefix,
        "observation_link": attach_links[-1],
        "support_contact_allowance_m": scene["KMR"]["task_execution"][
            "attachment_support_contact_allowance_m"
        ],
    }
    controller["background_preparation_enabled"] = bool(
        scene["KMR"]["task_execution"].get("background_preparation_enabled", False)
    )
    controller["retain_observed_clear_pose_as_home"] = bool(
        robot.get("retain_observed_clear_pose_as_home", False)
    )
    named_positions = {
        "home": [float(robot["initial_joint_positions"][name]) for name in _UR_JOINTS]
    }
    static = deepcopy(base["static_capabilities"])
    static["reachability"] = sorted(
        {
            connection["origin_resource_location"]
            for connection in robot.get("handling_connections", [])
        }
        | {
            connection["destination_location"]
            for connection in robot.get("handling_connections", [])
        }
    )
    static["gripper_reach"]["origin_pose"] = {
        axis: float(value) for axis, value in zip(("x", "y", "z"), robot["base_xyz"], strict=True)
    }
    return controller, named_positions, static


def _prepared_controller_matches(controller: Any, config: dict) -> bool:
    """Check that a reusable controller belongs to this exact configured resource."""
    move_group = config["move_group"]
    return bool(
        getattr(controller, "is_usable", lambda: False)()
        and list(getattr(controller, "arm_joint_names", ())) == config["arm_joint_names"]
        and getattr(controller, "arm_trajectory_topic", None)
        == config["arm_trajectory_topic"]
        and getattr(controller, "group_name", None) == move_group["group_name"]
        and getattr(controller, "ee_link", None) == move_group["ee_link"]
        and getattr(controller, "tcp_link", None) == move_group["tcp_link"]
    )


def create_environment_resource_agents(
    scene: dict,
    models: dict,
    cca_jid: str | None,
    *,
    prewarmed_controllers: dict | None = None,
) -> list[ResourceAgent]:
    """Create configured resource types while retaining exact resource identifiers."""
    prewarmed = prewarmed_controllers if prewarmed_controllers is not None else {}
    robot_rows = {row["resource_id"]: row for row in scene["robots"]}
    resources: list[ResourceAgent] = []
    for index, rid in enumerate(models, 1):
        jid = f"recovery-resource-{index}@localhost"
        if rid in robot_rows:
            controller, named_positions, static = _robot_configuration(scene, robot_rows[rid])
            names = sorted(
                set(models[rid]["local_event_alphabet"]) & set(robot_task_registry())
            )
            controller_key = rid.lower()
            prepared_controller = prewarmed.pop(controller_key, None)
            common = {}
            if prepared_controller is not None and _prepared_controller_matches(
                prepared_controller, controller,
            ):
                common["prewarmed_controller"] = prepared_controller
            elif prepared_controller is not None:
                prepared_controller.shutdown()
            resources.append(
                RobotAgent(
                    jid,
                    "none",
                    name=rid,
                    cca_jid=cca_jid,
                    tool_timeout_s=600,
                    function_names=names,
                    static_capabilities=static,
                    execution_mode="simulation",
                    controller_config=controller,
                    named_positions=named_positions,
                    enable_controller_prewarm=True,
                    **common,
                )
            )
            continue
        worker = None
        if rid == "KMR":
            worker = GazeboWorker(label="KMR", timeout_sec=600)
            resources.append(KMRResourceAgent(
                jid, "none", cca_jid=cca_jid, tool_timeout_s=600, worker=worker,
            ))
            continue
        elif rid in {"M1", "M2", "Conveyor", "Buffer For Machined parts"}:
            worker = GazeboWorker(
                module="cais_spade_llm.recovery_framework.workflow_gazebo",
                label=rid,
                timeout_sec=600,
            )
        resources.append(
            WorkflowResourceAgent(
                jid, "none", name=rid, cca_jid=cca_jid, tool_timeout_s=600, worker=worker
            )
        )
    for controller in prewarmed.values():
        try:
            controller.shutdown()
        except Exception:
            pass
    prewarmed.clear()
    return resources


def _part_geometry(context, part_name: str) -> dict:
    raw = context.inputs["geometry"]
    geometry = ProductProfile.geometry_for_part_from_geometry(part_name, raw)
    if geometry:
        contact = raw["assembly_board"].get("mating_contact_by_part", {}).get(part_name)
        if contact:
            expected = raw["parts"]["assembly_target_map"][part_name] + "/collision"
            if contact["target_collision_object"] != expected:
                raise ValueError("Mating contact must identify this part's configured assembly target")
            geometry["simulation_mating_contact"] = deepcopy(contact)
        return geometry
    item = context.geometry[part_name]
    dimensions = item.get("dimensions_m") or [0.05, 0.05, 0.05]
    return {
        "part_name": part_name,
        "part_height_m": float(dimensions[2]),
        "model_name": item.get("model_name") or part_name,
    }


def _pick_geometry(context, part_name: str, origin: str, robot_id: str) -> dict:
    """Add the configured source access contract to a part's pick geometry."""
    geometry = deepcopy(_part_geometry(context, part_name))
    dimensions = context.geometry[part_name].get("dimensions_m")
    if dimensions and "grasp_width_m" not in geometry:
        width = max(float(dimensions[0]), float(dimensions[1]))
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError(f"Invalid configured grasp width for {part_name}")
        geometry["grasp_width_m"] = width
    scene = context.inputs["scene"]
    source = scene.get(origin)
    if source is None:
        source = next(
            (row for row in scene.get("machines", []) if row["resource_id"] == origin),
            None,
        )
    access = dict(source.get("handling_robot_access") or {}) if source else {}
    if access and access.get("resource_id") == robot_id:
        geometry["handling_robot_access"] = access
    return geometry


def _destination_geometry(context, part_name: str, destination: str, robot_id: str) -> dict:
    if destination == context.product_name:
        return _part_geometry(context, part_name)
    scene = context.inputs["scene"]
    if destination == "Conveyor":
        machine = next(
            row for row in scene["machines"] if row["handling_robot"] == robot_id
        )
        pose = machine["conveyor_loading_pose"]
    elif destination.endswith(" staging tray"):
        machine_id = destination.removesuffix(" staging tray")
        pose = next(row["staging_pose"] for row in scene["machines"] if row["resource_id"] == machine_id)
    elif destination == "Exit":
        pose = scene["Exit"]["completed_product_pose"]
    else:
        machine = next(row for row in scene["machines"] if row["resource_id"] == destination)
        pose = machine["workholding_pose"]
    dimensions = context.geometry[part_name].get("dimensions_m") or [0.05, 0.05, 0.05]
    return {
        "part_name": part_name,
        "part_height_m": float(dimensions[2]),
        "model_name": context.geometry[part_name].get("model_name") or part_name,
        "board_center": {"x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2])},
        "slot_xy": [0.0, 0.0],
        "slot_floor_z_m": float(pose[2]) - float(dimensions[2]) / 2.0,
        "target_origin_pose": {"x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2])},
        "target_reference": {"target_point": "part_origin", "surface_role": "support"},
    }


def _robot_arguments(context, agent: RobotAgent, task: dict) -> tuple[str, dict]:
    event_name = task["event_name"]
    program_name = "place_insert" if event_name == "place_release" else event_name
    parameters = task["parameters"]
    allowed = {argument.name for argument in robot_task_registry()[program_name].arguments}
    arguments = {key: deepcopy(value) for key, value in parameters.items() if key in allowed}
    part_name = parameters.get("part_name")
    if part_name:
        if program_name.startswith("place_"):
            arguments["product_geometry"] = _destination_geometry(
                context, part_name, parameters["destination_location"], agent.agent_name
            )
        else:
            arguments["product_geometry"] = _pick_geometry(
                context,
                part_name,
                parameters["origin_resource_location"],
                agent.agent_name,
            )
    arguments["task_id"] = task["task_id"]
    arguments["product_jid"] = context.product_name
    return program_name, arguments


def bind_environment_executors(runtime, resources: list[ResourceAgent]) -> None:
    """Bind every physically executable event to its owning persistent controller."""
    by_name = {resource.agent_name: resource for resource in resources}
    context = runtime.context
    runtime.assembly_worker = GazeboWorker(
        module="cais_spade_llm.recovery_framework.workflow_gazebo",
        label="assembly_board-v1",
        timeout_sec=120,
    )
    runtime.assembly_fixture_registration = None

    async def prepare_execution() -> None:
        """Register shared fixtures once in the ordered production startup."""
        if (runtime.assembly_fixture_registration is not None
                or not any(isinstance(agent, RobotAgent) for agent in resources)):
            return
        registration = await runtime.assembly_worker.run({
            "operation": "register_assembly_fixtures",
            "scene": context.inputs["scene"],
            "geometry": context.geometry,
        })
        if registration.get("status") != "completed":
            raise ValueError(registration.get("error") or "Could not register assembly fixtures to their carrier")
        runtime.assembly_fixture_registration = registration

    runtime.prepare_execution = prepare_execution
    for rid, actor in context.resources.items():
        agent = by_name[rid]
        for event_name in actor.model["local_event_alphabet"]:
            if isinstance(agent, RobotAgent) and event_name in {
                "pick_approach",
                "pick_grasp",
                "place_approach",
                "place_insert",
                "place_release",
                "move_home",
            }:

                async def execute_robot(task: dict, *, owner=agent) -> dict:
                    program_name, arguments = _robot_arguments(context, owner, task)
                    result = await getattr(owner, program_name)(**arguments)
                    simulation_execution = dict(result.get("observations", {})).get(
                        "simulation_execution", {}
                    )
                    evidence = {
                        "resource_id": owner.agent_name,
                        "event_name": task["event_name"],
                        "task_id": task["task_id"],
                        "controller_result": result,
                        "timing": deepcopy(simulation_execution.get("timing", {})),
                    }
                    if task["event_name"] == "move_home" and result.get("status") == "completed":
                        evidence["home_observation"] = await asyncio.to_thread(_observe_robot_home, owner)
                    return evidence

                actor.bind_executor(
                    event_name,
                    execute_robot,
                    _validate_robot_completion,
                    validate_start=_validate_start,
                )
            elif event_name == "machine_part" and getattr(agent, "workflow_worker", None):
                actor.bind_executor(
                    event_name,
                    _worker_executor(runtime, agent),
                    _validate_machine_completion,
                    validate_start=_validate_start,
                )
            elif event_name in {"advance_conveyor", "advance_part"} and getattr(
                agent, "workflow_worker", None
            ):
                actor.bind_executor(
                    event_name,
                    _worker_executor(runtime, agent),
                    _validate_transport_completion,
                    validate_start=_validate_start,
                )
            elif rid == "KMR" and event_name in {
                "pick_part",
                "move_to_resource",
                "place_release",
            }:
                actor.bind_executor(
                    event_name,
                    _kmr_executor(runtime, agent),
                    _validate_kmr_completion,
                    validate_start=_validate_start,
                )


async def _validate_start(task: dict, valuation: dict, geometry: dict) -> bool:
    return bool(task.get("task_id") and valuation and geometry is not None)


def _worker_executor(runtime, agent: WorkflowResourceAgent):
    async def execute(task: dict) -> dict:
        request = {
            "scene": runtime.context.inputs["scene"],
            "geometry": runtime.context.geometry,
            "valuation": runtime.context.snapshot(),
            "task": task,
        }
        try:
            return await agent.workflow_worker.run(request)
        except GazeboExecutionError as exc:
            return exc.result

    return execute


def _kmr_executor(runtime, agent: KMRResourceAgent):
    async def execute(task: dict) -> dict:
        request = {
            "mode": "environment_task", "inputs": runtime.context.inputs,
            "pending": task, "valuation": runtime.context.snapshot(),
            "geometry": runtime.context.geometry,
            "custody": deepcopy(getattr(agent, "workflow_custody", None)),
        }
        agent._kmr_execution_request = deepcopy(request)
        try:
            result = await agent.workflow_worker.run(request)
            agent.workflow_custody = deepcopy(result.get("observations", result))
        except GazeboExecutionError as exc:
            result = exc.result
        if hasattr(agent, 'record_primitive_evidence'):
            agent.record_primitive_evidence(result)
        return result

    return execute


def _identity_matches(task: dict, evidence: dict) -> bool:
    return (
        evidence.get("status") == "completed"
        and evidence.get("resource_id") == task["resource_id"]
        and evidence.get("event_name") == task["event_name"]
        and evidence.get("task_id") == task["task_id"]
    )


def _observe_robot_home(agent: RobotAgent) -> dict:
    """Capture fresh, stable joint feedback at the configured home endpoint."""
    controller = agent._controller
    target = list(agent.named_positions["home"])
    names = list(controller.arm_joint_names)
    tolerance = 0.02
    deadline = time.monotonic() + 2.0
    while True:
        observed, missing = controller._get_arm_joint_positions(timeout_sec=0.0)
        stable = controller._fresh_stable_joint_target(
            dict(zip(names, target, strict=True)), tolerance=tolerance,
        )
        if stable or time.monotonic() >= deadline:
            return {
                "pose_name": "home", "joint_names": names,
                "target_positions": target, "observed_positions": observed,
                "missing_joints": missing, "fresh_stable": stable,
                "tolerance_rad": tolerance, "held_part": agent._held_part,
                "observed_at_unix": time.time(),
            }
        time.sleep(0.02)


def _validate_robot_completion(task: dict, evidence: dict) -> bool:
    result = evidence.get("controller_result", {})
    if not (
        _identity_matches(task, {**evidence, "status": "completed"})
        and result.get("status") == "completed"
    ):
        return False
    if task["event_name"] == "move_home":
        observed = evidence.get("home_observation", {})
        actual = observed.get("observed_positions") or []
        target = observed.get("target_positions") or []
        return (
            observed.get("pose_name") == "home"
            and observed.get("fresh_stable") is True
            and "held_part" in observed and observed["held_part"] is None
            and observed.get("missing_joints") == []
            and len(actual) == len(target) == len(observed.get("joint_names", [])) > 0
            and all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in [*actual, *target])
            and all(abs(math.atan2(math.sin(a - b), math.cos(a - b))) <= 0.02
                    for a, b in zip(actual, target, strict=True))
        )
    return True


def _validate_machine_completion(task: dict, evidence: dict) -> bool:
    observations = evidence.get("observations", {})
    return (
        _identity_matches(task, evidence)
        and observations.get("correct_part_present") is True
        and observations.get("robot_access_clear") is True
        and float(evidence.get("timing", {}).get("simulation_sec", -1.0)) + 1e-6
        >= float(observations.get("processing_time_sec", 0.0))
    )


def _validate_transport_completion(task: dict, evidence: dict) -> bool:
    observations = evidence.get("observations", {})
    return (
        _identity_matches(task, evidence)
        and observations.get("arrival_observed") is True
        and observations.get("source_clear") is True
        and observations.get("robot_clear") is True
    )


def _validate_kmr_completion(task: dict, evidence: dict) -> bool:
    observations = evidence.get("observations", evidence)
    if task["event_name"] == "move_to_resource" and task["parameters"].get("target_resource") == "Storage":
        home = observations.get("home_observation") or {}
        if not (home.get("home_pose_observed") is True and home.get("downward_facing") is True
                and home.get("gripper_open") is True and observations.get("attached") is False
                and observations.get("part_name") is None):
            return False
    return (
        evidence.get("status") == "completed"
        and evidence.get("task_id") == task["task_id"]
        and observations.get("controllers_succeeded") is True
    )
