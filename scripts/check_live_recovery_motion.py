from __future__ import annotations

"""Check one supplied motion in an already launched dedicated Gazebo session.

The candidate is the first move_cartesian of GAZEBO_MOTION_SAFE. Both predefined
requirements remain active, and only a committed CCA grant permits execution.
This does not establish part_slippage recovery or nominal product completion.
Launch the complete scene with recovery_observations:=true and launch_nav2:=false.
"""

import argparse
import asyncio
import json
import logging
import time
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, read_json

logger = logging.getLogger(__name__)
CANDIDATE_ID = "GAZEBO_MOTION_SAFE"
SAFETY_PATH = ROOT / "cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt"
SPECIFICATIONS = {"SAFE_shared_area_mutex", "SAFE_gear_small_before_KET4_Square_4mm"}


def controller_observation_services(topics: list[str]) -> dict:
    """Bind declared endpoints to their native complete idle observation owners."""
    suffix = "/follow_joint_trajectory/_action/status"
    services, base_topics = {}, []
    for topic in topics:
        if topic.endswith(suffix):
            services[topic.removesuffix(suffix) + "/recovery_state"] = {"covers": [topic]}
        else:
            base_topics.append(topic)
    if base_topics:
        services["/KMR/recovery_state"] = {
            "covers": base_topics, "required_values": {"navigation_enabled": False},
        }
    return services


def supplied_task(candidate: dict, resource_jid: str) -> dict:
    """Retain exactly the first configured ur5e-4 command and its provenance."""
    if candidate.get("recovery_id") != CANDIDATE_ID or len(candidate["programs"]) != 1:
        raise ValueError("Expected the configured GAZEBO_MOTION_SAFE candidate")
    program = candidate["programs"][0]
    if program["resource_id"] != "ur5e-4":
        raise ValueError("The bounded supplied motion belongs to ur5e-4")
    step = deepcopy(program["primitive_steps"][0])
    if step["primitive"] != "move_cartesian" or step["source"]["step_index"] != 0:
        raise ValueError("The supplied first step is not move_cartesian")
    return {
        "task_id": step["source"]["outline_id"],
        "outline_id": step["source"]["outline_id"],
        "resource_id": program["resource_id"], "resource_jid": resource_jid,
        "function_name": "execute_recovery_macro", "params": {"primitive_steps": [step]},
        "primitive_steps": [step],
    }


def native_contract_diagnostics() -> dict:
    """Inspect native ownership separately, using only stationary or zero-motion inputs."""
    import rclpy
    from builtin_interfaces.msg import Duration
    from cais_lab_robotics.srv import SetRecoveryMotionContract
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rosidl_runtime_py.convert import message_to_ordereddict
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from cais_spade_llm.recovery_framework import fingerprint

    context = rclpy.context.Context()
    rclpy.init(context=context)
    node = rclpy.create_node("recovery_motion_contract_diagnostics", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    endpoint = "/ur5e_4_joint_trajectory_controller"
    reader = node.create_client(Trigger, endpoint + "/recovery_state")
    contract = node.create_client(SetRecoveryMotionContract, endpoint + "/recovery_motion_contract")
    action = ActionClient(node, FollowJointTrajectory, endpoint + "/follow_joint_trajectory")
    publisher = node.create_publisher(JointTrajectory, endpoint + "/joint_trajectory", 10)
    report = {"evidence_kind": "native_owner_diagnostics", "cca_admission": False,
              "moving_command_sent": False, "physical_execution_verified": False,
              "status": "NEEDS_CONTEXT", "operations": []}
    token = "native_stationary_diagnostic"
    binding = fingerprint({"endpoint": endpoint, "operation": "stationary_diagnostic"})
    armed = False

    def call(client, request):
        if not client.wait_for_service(timeout_sec=10.):
            raise ValueError("Native diagnostic service unavailable: " + client.srv_name)
        future = client.call_async(request)
        executor.spin_until_future_complete(future, timeout_sec=10.)
        if not future.done():
            future.cancel()
            raise TimeoutError("Native diagnostic service timed out: " + client.srv_name)
        return future.result()

    def observe():
        result = call(reader, Trigger.Request())
        if not result.success:
            raise ValueError(result.message)
        return json.loads(result.message)

    def operation(name, *, wrong_identity=False):
        observed = observe()
        request = SetRecoveryMotionContract.Request(
            operation=name, expected_instance_id=("stale" if wrong_identity else observed["instance_id"]),
            expected_command_revision=observed["command_revision"],
            expected_contract_revision=observed["contract_revision"],
            expected_positions=observed["positions"], stationary=True,
            reservation_token=token, binding_fingerprint=binding,
        )
        result = call(contract, request)
        report["operations"].append({"before": observed,
            "request": message_to_ordereddict(request), "response": message_to_ordereddict(result)})
        return result

    try:
        report["initial_observation"] = observe()
        prepared = operation("prepare")
        stale = operation("prepare", wrong_identity=True)
        report["stale_identity_rejected"] = not stale.accepted
        if not prepared.accepted:
            report["reason"] = prepared.reason
            return report
        reservation = operation("arm")
        armed = reservation.accepted
        if not armed:
            report["reason"] = reservation.reason
            return report
        observed = observe()
        trajectory = JointTrajectory(joint_names=observed["joint_names"], points=[
            JointTrajectoryPoint(positions=observed["positions"], time_from_start=Duration(sec=1)),
        ])
        report["foreign_zero_motion_trajectory"] = message_to_ordereddict(trajectory)
        deadline = time.monotonic() + 3.
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.1)
        if publisher.get_subscription_count() == 0:
            raise ValueError("Native diagnostic trajectory subscriber unavailable")
        publisher.publish(trajectory)
        if not action.wait_for_server(timeout_sec=5.):
            raise ValueError("Native diagnostic action server unavailable")
        future = action.send_goal_async(FollowJointTrajectory.Goal(trajectory=trajectory))
        executor.spin_until_future_complete(future, timeout_sec=5.)
        if not future.done():
            raise TimeoutError("Native diagnostic goal acknowledgement unavailable")
        report["foreign_zero_motion_action_accepted"] = future.result().accepted
        if future.result().accepted:
            raise ValueError("Stationary native contract accepted a foreign goal")
        report["after_foreign_commands"] = observe()
        stopped = operation("stop")
        report["stop_accepted"] = stopped.accepted
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            observation = observe()
            if observation["observed_stationary"]:
                break
            executor.spin_once(timeout_sec=.1)
        report["before_release"] = observation
        released = operation("release")
        armed = not released.accepted
        report["release_accepted"] = released.accepted
        if (released.accepted and stopped.accepted and report["stale_identity_rejected"]
                and report["after_foreign_commands"]["rejected_commands"] >= observed["rejected_commands"] + 2):
            report["status"] = "native_owner_checks_passed"
    except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
        report["reason"] = str(exc)
    finally:
        if armed:
            report["native_reservation_retained"] = True
        action.destroy()
        executor.shutdown()
        node.destroy_node()
        context.shutdown()
    return report


async def check_and_execute(live, runtime, task: dict, report: dict) -> None:
    """Execute only a committed exact grant and retain authenticated completion."""
    request = {"recovery_id": CANDIDATE_ID, "recovery_safety_scope_id": CANDIDATE_ID,
               "tasks": [deepcopy(task)]}
    report["request"] = request
    registration = await live.register(request, product_jid=runtime.product_jid)
    report["registration"] = registration
    if registration["status"] != "allowed":
        report["reason"] = registration.get("reason", "Registration unavailable")
        return
    reference = registration["task_refs"][task["task_id"]]
    event = {key: task[key] for key in ("task_id", "resource_jid", "function_name")}
    event["params"] = {**deepcopy(task["params"]), "recovery_composition_ref": reference}
    incomplete_checkpoint = bool(report.get("checkpoint", {}).get("unresolved"))
    decision = await live.check(event, sender=task["resource_jid"], commit=not incomplete_checkpoint)
    report["cca_decision"] = decision
    report["common_composition"] = deepcopy(decision.get("common_composition"))
    report["reason"] = decision.get("reason", "")
    if incomplete_checkpoint:
        report["reason"] = decision.get("reason") or "Complete live checkpoint unavailable"
        report["cca_check_read_only"] = True
        return
    if decision["status"] != "allowed":
        report["status"] = "held" if decision["status"] == "held" else "NEEDS_CONTEXT"
        return
    grant = decision.get("recovery_composition_grant")
    if decision.get("committed") is not True or not isinstance(grant, dict):
        report["reason"] = "CCA did not commit a physical execution grant"
        return
    if (grant.get("primitive_steps") != task["primitive_steps"]
            or grant.get("run_id") != runtime.context.run_id
            or grant.get("recovery_composition_ref") != reference):
        raise ValueError("CCA grant differs from the registered supplied motion")
    report["dispatch_authorized"] = True
    owner = next(row for row in runtime.resource_agents if row.agent_name == task["resource_id"])
    step = task["primitive_steps"][0]
    report["command_sent"] = None
    result = await owner.execute_recovery_composition_step(
        primitive=step["primitive"], params=step["params"], task_id=task["task_id"],
        step_index=0, grant=grant, owner=owner,
    )
    report["execution"] = result
    report["command_sent"] = result.get("command_sent") is True
    completion = {
        **{key: task[key] for key in ("task_id", "resource_jid", "function_name")},
        "run_id": runtime.context.run_id, "recovery_composition_ref": reference,
        "status": "completed" if result.get("success") is True else "failed",
    }
    report["completion"] = completion
    report["observation_result"] = live.observe(completion, sender=task["resource_jid"])
    report["acceptance_complete"] = (
        report["command_sent"] and result.get("success") is True
        and report["observation_result"].get("success") is True
    )
    report["status"] = "completed" if report["acceptance_complete"] else "NEEDS_CONTEXT"


def retain_execution_evidence(live, owners: list, report: dict) -> None:
    """Retain owner preparations, declared assumptions, and actual observations.

    Missing composition results remain unavailable. Diagnostic samples and
    declared model assumptions do not become verified physical bounds here.
    """
    with live.lock:
        report["commands"] = live.commands.snapshot()
        report["physical_history"] = deepcopy(live.monitor.history)
        report["physical_states"] = {key: sorted(value) for key, value in live.monitor.states.items()}
        coverage = deepcopy(getattr(live.preparation, "last_execution_coverage", None))
        report["execution_coverage"] = coverage
        report["model_execution_assumptions"] = (
            deepcopy(coverage.get("model_execution_assumptions")) if isinstance(coverage, dict) else None
        )
        report["common_composition"] = deepcopy(report.get("cca_decision", {}).get("common_composition"))
        report["owner_execution_coverage"] = {}
        report["prepared_programs"] = {}
        for owner in owners:
            provider = getattr(owner, "recovery_composition_evidence_provider", None)
            if provider is None:
                continue
            report["owner_execution_coverage"][owner.agent_name] = deepcopy(
                getattr(provider, "execution_coverage_records", {})
            )
            report["prepared_programs"][owner.agent_name] = {
                "steps": deepcopy(getattr(provider, "steps", {})),
                "preparations": {
                    key: {**deepcopy(row), "executed": sorted(row["executed"])}
                    for key, row in getattr(provider, "preparations", {}).items()
                },
            }
        report["last_results"] = deepcopy(live.last_results)


def _json_value(value):
    """Preserve ROS integer scalars without stringifying unsupported evidence."""
    if isinstance(value, Integral):
        return int(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


async def run_check(output_directory: Path) -> dict:
    """Collect live evidence without starting agents or changing saved inputs."""
    from cais_spade_llm.agents.central_controller.central_controller_agent import (
        CentralControllerAgent,
    )
    from cais_spade_llm.agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
    from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
        initialize_predefined_safety,
        predefined_scope,
    )
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.recovery_framework import environment_runtime
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
        build_supplied_candidate,
        capture_checkpoint,
    )
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import install_live_runtime
    from cais_spade_llm.recovery_framework.workflow_execution import (
        create_environment_resource_agents,
    )

    report = {
        "version": 1, "candidate_id": CANDIDATE_ID, "origin": "mock",
        "candidate_provenance": {"source": "supplied", "llm_generated": False},
        "execution_mode": "simulation", "diagnostic_cca_bypass": False,
        "status": "NEEDS_CONTEXT", "acceptance_complete": False,
        "dispatch_authorized": False, "command_sent": False,
        "part_slippage_recovery_complete": False, "nominal_product_complete": False,
    }
    owners, live = [], None
    try:
        meta = next(iter(read_json(PRODUCT_PATH).values()))
        scene = read_json(SCENE_PATH)
        configuration = scene["safety_preparation"]
        configuration["controller_state_services"] = controller_observation_services(
            configuration["controller_status_topics"])
        report["run_configuration"] = deepcopy(configuration)
        inputs = {"scene": scene, "product_order": read_json(ROOT / meta["product_order_file"]),
                  "geometry": read_json(ROOT / meta["product_geometry_file"])["gazebo"]}
        context = EnvironmentProductContext(**inputs)
        owners = create_environment_resource_agents(scene, context.models, "cca@localhost")
        # Preserve the ordinary runtime's latest report ownership in this CLI harness.
        previous_directory = environment_runtime.RUN_DIRECTORY
        environment_runtime.RUN_DIRECTORY = output_directory / "environment_runs"
        try:
            runtime = environment_runtime.EnvironmentRuntime(SimpleNamespace(jid="assembly_board-v1@localhost"), {
                "inputs": inputs, "diagnostic_cca_bypass": False,
                "setup": {"execution_mode": "simulation", "permitted_resources": list(context.models)},
            }, owners)
        finally:
            environment_runtime.RUN_DIRECTORY = previous_directory
        report["run_id"] = runtime.context.run_id
        cca = CentralControllerAgent("cca@localhost", "none", name="cca", resource_agents=owners,
                                     safety_file=str(SAFETY_PATH))
        initialize_predefined_safety(cca)
        if cca.predefined_safety_error:
            raise ValueError(cca.predefined_safety_error)
        report["predefined_safety"] = deepcopy(cca.predefined_safety)
        identifiers = {row["id"] for row in cca.predefined_safety["catalog"]["specifications"]}
        if identifiers != SPECIFICATIONS:
            raise ValueError("Both unchanged predefined specifications are required")
        for owner in owners:
            controller = getattr(owner, "_controller", None)
            if controller is not None and not controller.wait_for_services(timeout_sec=30):
                raise ValueError("Controller unavailable: " + owner.agent_name)
        live = install_live_runtime(runtime, cca)
        if live is None:
            raise ValueError("CCA live admission is unavailable")
        live.preparation.initialize(model_execution=True)
        report["physics"] = live.preparation.reader.snapshot(refresh=True)
        report["checkpoint"] = capture_checkpoint(runtime, cca, model_execution=True)
        bridge = SimpleNamespace(resource_agents=owners, cca=cca)
        candidate = build_supplied_candidate(bridge, CANDIDATE_ID)
        report["supplied_candidate"] = deepcopy(candidate)
        task = supplied_task(candidate, runtime.jids["ur5e-4"])
        task_id = task["task_id"]
        cca.plan_fsa_monitor = OnlineFsaMonitor({"A": {
            "x0": "checkpoint0", "X": ["checkpoint0", "checkpoint1", "checkpoint2"],
            "Xm": ["checkpoint2"], "Tr": [
                {"from": "checkpoint0", "to": "checkpoint1", "event": task_id + ".start", "task_id": task_id},
                {"from": "checkpoint1", "to": "checkpoint2", "event": task_id + ".done", "task_id": task_id},
            ],
        }})
        cca.plan_fsa_monitor.history_path = output_directory / "online_fsa_trace.jsonl"
        predefined_scope(cca, CANDIDATE_ID)
        await check_and_execute(live, runtime, task, report)
    except (ValueError, KeyError, RuntimeError, OSError, TypeError, TimeoutError) as exc:
        report["reason"] = str(exc)
        logger.error("Live acceptance incomplete: %s", exc)
    finally:
        if live is not None:
            retain_execution_evidence(live, owners, report)
            if live.preparation.reader is not None:
                live.preparation.reader.close()
        for owner in owners:
            controller = getattr(owner, "_controller", None)
            if controller is not None:
                controller.shutdown()
    return report


def main() -> int:
    """Save a fresh bounded acceptance report, preserving all prior evidence."""
    parser = argparse.ArgumentParser(description=(
        "Check one supplied move_cartesian in a dedicated live Gazebo session; "
        "both predefined specifications and a committed CCA grant are required."))
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--native-contract-diagnostics-only", action="store_true",
                        help="Run separate stationary native ownership diagnostics; never CCA acceptance.")
    args = parser.parse_args()
    directory = args.output_directory.resolve()
    if directory.exists():
        parser.error("Use a new output directory to preserve prior evidence")
    directory.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)
    report = (native_contract_diagnostics() if args.native_contract_diagnostics_only
              else asyncio.run(run_check(directory)))
    path = directory / "result.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_value) + "\n", encoding="utf-8")
    logger.info("%s: %s", report["status"], path)
    passed = (report.get("status") == "native_owner_checks_passed" if args.native_contract_diagnostics_only
              else report["acceptance_complete"])
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
