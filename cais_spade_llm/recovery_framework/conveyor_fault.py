"""Once-per-run simulation faults using the existing Conveyor bridge routes."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from cais_spade_llm.recovery_framework import ROOT

logger = logging.getLogger(__name__)
CHECKPOINT = "after_M1_pick_before_release"


def marker(scene: dict, action: str) -> dict:
    """Show or clear the non-colliding Gazebo breakdown marker through ROS."""
    with tempfile.TemporaryDirectory(prefix="cais_conveyor_fault_") as directory:
        path = Path(directory) / "scene.json"
        path.write_text(json.dumps(scene))
        command = (
            "source /opt/ros/humble/setup.bash\n"
            'source "$1"\n'
            "exec /usr/bin/python3 -m cais_spade_llm.recovery_framework.fault_visual "
            '--scene "$2" --action "$3"'
        )
        try:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    command,
                    "conveyor-fault",
                    str(Path.home() / "ros2_ws/install/setup.bash"),
                    str(path),
                    action,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"status": "failed", "error": str(exc)}
        if result.returncode:
            return {"status": "failed", "error": result.stderr.strip()[-1500:]}
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"status": "failed", "error": "Gazebo marker returned no evidence"}


class ConveyorFault:
    """Own the selected simulation fault through the existing bridge interface."""

    def __init__(self, runtime, setup: dict) -> None:
        """Create a fresh once-per-run latch for an explicitly configured checkpoint."""
        from cais_spade_llm.recovery_framework.failure_checkpoints import SUPPORTED_CHECKPOINTS

        self.runtime = runtime
        self.configuration = deepcopy(setup.get("failure_scenario"))
        self.enabled = self.configuration is not None
        if self.enabled and (
            setup.get("execution_mode") != "simulation"
            or self.configuration.get("checkpoint") not in SUPPORTED_CHECKPOINTS.get(
                self.configuration.get("scenario"), ())
        ):
            raise ValueError("Only configured observed failure checkpoints are integrated in simulation")
        self.status = "armed" if self.enabled else "disabled"
        self.revision = 0
        self.evidence: dict = {}
        self.visual: dict = {}
        self._machine_evidence: dict | None = None
        self._machine_task: dict | None = None
        self._robot_task: dict | None = None
        self._robot_evidence: dict | None = None
        self._robot_pickups: dict | None = None
        self._control_directory = None
        self._lock = asyncio.Lock()
        self._injection_task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        """Return fault state independently of injection and visualization outcomes."""
        configuration = self.configuration or {}
        return {
            "scenario": configuration.get("scenario", "Conveyor breakdown"),
            "resource_id": configuration.get("resource_id", "Conveyor"),
            "checkpoint": configuration.get("checkpoint", CHECKPOINT),
            "affected_resources": [configuration.get("resource_id", "Conveyor")] + (
                [configuration["additional_condition"]["resource_id"]]
                if configuration.get("additional_condition") else []
            ),
            "status": self.status,
            "run_active": not self.runtime.stopped,
            "run_id": self.runtime.context.run_id,
            "revision": self.revision,
            "ready": self.enabled and not self.runtime.stopped and self.checkpoint() is not None,
            "evidence": deepcopy(self.evidence),
            "visual": deepcopy(self.visual),
        }

    def checkpoint(self) -> dict | None:
        """Require current resource observations at the configured checkpoint."""
        from cais_spade_llm.recovery_framework.failure_checkpoints import checkpoint

        if not self.enabled:
            return None
        if self.configuration["scenario"] == "Machining breakdown during part processing":
            return deepcopy(self._machine_evidence)
        if (self.configuration["scenario"] == "Part slippage"
                and self.configuration["checkpoint"] == "during_place_lowering"):
            if self._robot_evidence is None:
                return None
            return {**deepcopy(self._robot_evidence), **deepcopy(self._robot_pickups),
                    "placement_motion": deepcopy(self._robot_evidence)}
        return checkpoint(self.runtime.context, self.configuration)

    def holds_task(self, task: dict) -> bool:
        """Gate selected placement while retaining the other acknowledged pickup."""
        from cais_spade_llm.recovery_framework.failure_checkpoints import holds_task

        return self.status == "armed" and holds_task(
            self.runtime.context, self.configuration, task,
        )

    def arm(self, armed: bool) -> dict:
        """Arm or disarm an active untriggered run and update its worker control."""
        if not self.enabled or self.runtime.stopped or self.status == "triggered":
            raise ValueError("Arm requires an active configured simulation; reset after a breakdown")
        self.status = "armed" if armed else "disarmed"
        self.revision += 1
        self._write_control()
        self.runtime.queue_save()
        return self.snapshot()

    def _write_control(self) -> None:
        if self._control_directory is not None:
            path = Path(self._control_directory.name) / "fault.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "run_id": self.runtime.context.run_id, "status": self.status,
                "resource_id": self.configuration["resource_id"],
            }))
            temporary.replace(path)

    def worker_request(self, task: dict) -> dict:
        """Bind a machine worker's checkpoint to this run and selected resource."""
        if (not self.enabled
                or self.configuration["scenario"] != "Machining breakdown during part processing"
                or task["resource_id"] != self.configuration["resource_id"]
                or task["event_name"] != "machine_part"):
            return {}
        if self._control_directory is None:
            self._control_directory = tempfile.TemporaryDirectory(prefix="cais_machine_fault_")
        self._write_control()
        self._machine_task = deepcopy(task)
        return {"run_id": self.runtime.context.run_id, "task_id": task["task_id"],
                "resource_id": task["resource_id"], "checkpoint": self.configuration["checkpoint"],
                "control_path": str(Path(self._control_directory.name) / "fault.json"),
                "evidence_path": str(Path(self._control_directory.name) / "interruption.json")}

    async def accept_machine_failure(self, task: dict, result: dict) -> bool:
        """Latch worker evidence bound to this run, including a cancelled reply."""
        evidence = result.get("failure_injection")
        if not evidence:
            return False
        context = self.runtime.context
        if (not self.enabled or self.status == "triggered"
                or self.configuration["scenario"] != "Machining breakdown during part processing"
                or (context.pending_for(task["task_id"]) != task and self._machine_task != task)
                or task["resource_id"] != self.configuration["resource_id"]
                or any(evidence.get(key) != task.get(key) for key in ("task_id", "resource_id", "run_id"))
                or evidence.get("checkpoint") != self.configuration["checkpoint"]
                or evidence.get("source") not in {"gazebo_workholding_observation", "gazebo_processing_checkpoint"}):
            raise ValueError("Stale or unrelated machining fault evidence")
        duration = evidence.get("processing_time_sec")
        elapsed = evidence.get("simulation_elapsed_sec")
        observed = evidence.get("observed_pose", {})
        observation_failed = (evidence.get("source") == "gazebo_processing_checkpoint"
                              and evidence.get("observation_status") in {"failed", "pending"})
        if (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0
                or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
                or elapsed < duration * 0.5 or evidence.get("process_completed") is not False
                or evidence.get("part_name") != task["parameters"].get("part_name")
                or (not observation_failed and any(
                    not isinstance(observed.get(axis), (int, float)) or not math.isfinite(observed[axis])
                    for axis in ("x", "y", "z")))):
            raise ValueError("Invalid machining interruption observation")
        # The worker owns the trigger instant. A later UI disarm cannot undo that failure.
        self._machine_evidence = deepcopy(evidence)
        await self.trigger()
        return True

    def _placement_progress(self) -> float:
        progress = self.configuration.get("placement_progress", .5)
        if (type(progress) not in (int, float) or not math.isfinite(progress)
                or not .5 <= progress < 1.):
            raise ValueError("Placement interruption progress must be at least halfway and before completion")
        return float(progress)

    def robot_request(self, task: dict) -> dict:
        """Bind the selected lowering executor to observed custody in this run.

        Args:
            task: The pending, normally approved place_approach task.

        Returns:
            Controller fault binding, or an empty request when the gate is inactive.
        """
        from cais_spade_llm.recovery_framework.failure_checkpoints import placement_checkpoint

        if (not self.enabled or self.status != "armed" or self.runtime.stopped
                or self.configuration["scenario"] != "Part slippage"
                or self.configuration["checkpoint"] != "during_place_lowering"):
            return {}
        context = self.runtime.context
        with context.admission_lock:
            if (not task.get("task_id") or context.pending_for(task["task_id"]) != task
                    or placement_checkpoint(context, self.configuration, task) is None):
                return {}
            progress = self._placement_progress()
            self._robot_task = deepcopy(task)
            return {"run_id": context.run_id, "task_id": task["task_id"],
                    "resource_id": task["resource_id"], "part_name": self.configuration["part_name"],
                    "checkpoint": self.configuration["checkpoint"], "placement_progress": progress}

    def _robot_capture_matches(self, task: dict, evidence: dict) -> bool:
        return (isinstance(evidence, dict)
                and task.get("run_id") == self.runtime.context.run_id
                and task.get("resource_id") == self.configuration["resource_id"]
                and task.get("event_name") == "place_approach"
                and task.get("parameters", {}).get("part_name") == self.configuration["part_name"]
                and all(evidence.get(key) == task.get(key) for key in ("run_id", "task_id", "resource_id"))
                and evidence.get("part_name") == self.configuration["part_name"]
                and evidence.get("checkpoint") == self.configuration["checkpoint"]
                and evidence.get("source") == "gazebo_placement_motion"
                and evidence.get("function_name") == "place_approach"
                and evidence.get("step_id") == "descend")

    def _validate_robot_observation(self, evidence: dict) -> None:
        poses = [evidence.get(name) for name in ("started_pose", "target_pose", "observed_pose")]
        if any(not isinstance(pose, dict) or any(
            type(pose.get(axis)) not in (int, float) or not math.isfinite(pose[axis])
            for axis in ("x", "y", "z")
        ) for pose in poses):
            raise ValueError("Invalid placement interruption poses")
        started, target, observed = poses
        descent = started["z"] - target["z"]
        progress, stamp = evidence.get("progress"), evidence.get("observed_at_unix")
        if (descent <= 0 or type(progress) not in (int, float) or not math.isfinite(progress)
                or not self._placement_progress() <= progress < 1.
                or type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0
                or not isinstance(evidence.get("controller_goal_id"), list)
                or len(evidence["controller_goal_id"]) != 16
                or any(type(value) is not int or not 0 <= value <= 255
                       for value in evidence["controller_goal_id"])
                or any(evidence.get(key) is not True for key in
                       ("goal_active", "goal_cancelled", "motion_stopped"))):
            raise ValueError("Invalid or unconfirmed placement interruption")
        actual_progress = (started["z"] - observed["z"]) / descent
        if (not self._placement_progress() <= actual_progress < 1.
                or not math.isclose(progress, actual_progress, rel_tol=0., abs_tol=1e-6)):
            raise ValueError("Placement interruption lacks observed downward progress")

    async def accept_robot_failure(self, task: dict, result: dict) -> bool:
        """Latch a confirmed lowering interruption before normal completion validation.

        Args:
            task: Pending placement task, or the same task retained after Stop.
            result: Owning executor result containing captured failure_injection.

        Returns:
            Whether this result was consumed as a physical failure interruption.
        """
        from cais_spade_llm.recovery_framework.failure_checkpoints import placement_checkpoint

        evidence = result.get("failure_injection")
        if (not evidence or not self.enabled or self.configuration["scenario"] != "Part slippage"
                or self.configuration["checkpoint"] != "during_place_lowering"):
            return False
        context = self.runtime.context
        with context.admission_lock:
            if not self._robot_capture_matches(task, evidence):
                raise ValueError("Stale or unrelated placement fault evidence")
            self._validate_robot_observation(evidence)
            if self.status == "triggered":
                if self._robot_task == task and self._robot_evidence == evidence:
                    return True
                raise ValueError("Stale or unrelated placement fault evidence")
            pending = context.pending_for(task.get("task_id"))
            retained = (self._robot_task == task
                        and task in self.runtime.outcome.get("cancelled_tasks", []))
            pickups = placement_checkpoint(context, self.configuration, task)
            if self.status == "reset" or (pending != task and not retained) or pickups is None:
                raise ValueError("Stale placement task or unacknowledged pickup custody")
            self._robot_task = deepcopy(task)
            self._robot_pickups = pickups
            self._robot_evidence = deepcopy(evidence)
        # Cancellation and disarm after capture cannot undo the physical interruption.
        await self.trigger()
        return True

    async def retain_worker_interruption(self) -> None:
        """Retain confirmed worker or controller evidence after Stop cancels its reply."""
        if self.status in {"triggered", "reset"}:
            return
        if self._robot_task is not None:
            agent = next((agent for agent in self.runtime.resource_agents
                          if agent.agent_name == self.configuration["resource_id"]), None)
            evidence = deepcopy(getattr(getattr(agent, "_controller", None),
                                        "_simulation_fault_evidence", None))
            if evidence and self._robot_capture_matches(self._robot_task, evidence):
                try:
                    await self.accept_robot_failure(self._robot_task, {"failure_injection": evidence})
                except ValueError as exc:
                    self.evidence = {
                        **evidence, "placement_motion": deepcopy(evidence),
                        "injection_status": "interrupted", "error": str(exc),
                        "physical_state_reconciliation_required": True,
                        "pending_tasks": deepcopy(self.runtime.outcome.get("cancelled_tasks", [])
                                                  or [self._robot_task]),
                        "continuations": deepcopy(getattr(self.runtime, "retained_paths", {})),
                        "requirements": deepcopy(self.runtime.context.requirements),
                    }
                    self.runtime.outcome["failure_evidence"] = deepcopy(self.evidence)
                    self.runtime.queue_save()
            return
        if self._control_directory is None or self._machine_task is None:
            return
        path = Path(self._control_directory.name) / "interruption.json"
        if path.is_file():
            await self.accept_machine_failure(
                self._machine_task, {"failure_injection": json.loads(path.read_text())},
            )

    def marker_scene(self) -> dict:
        """Bind a visual marker to the configured resource and observed dropped part."""
        scene = deepcopy(self.runtime.context.inputs["scene"])
        scene["_failure_marker"] = {
            **(self.configuration or {}), "evidence": deepcopy(self.evidence),
        }
        return scene

    async def trigger(self) -> dict:
        """Capture obligations, stop execution, apply the fault, and display evidence."""
        async with self._lock:
            if self.status == "triggered":
                return self.snapshot()
            if (not self.enabled
                    or (self.runtime.stopped and self._machine_evidence is None and self._robot_evidence is None)
                    or (self.status != "armed" and self._machine_evidence is None and self._robot_evidence is None)):
                raise ValueError("Failure injection is not armed in an active simulation")
            evidence = self.checkpoint()
            if evidence is None:
                raise ValueError("Waiting for the configured observed failure checkpoint")
            runtime, configuration = self.runtime, self.configuration
            context = runtime.context
            rid, scenario = configuration["resource_id"], configuration["scenario"]
            with context.admission_lock:
                self.status = "triggered"
                self.revision += 1
                self._write_control()
                self.evidence = {
                    **evidence, "triggered_at_unix": time.time(), "run_id": context.run_id,
                    "checkpoint": configuration["checkpoint"],
                    "resource_values_before": context.snapshot(),
                    "part_tracker_before": deepcopy(context.part_tracker),
                    "pending_tasks": deepcopy(list(context.pending_tasks.values())
                                              or runtime.outcome.get("cancelled_tasks", [])),
                    "continuations": deepcopy(getattr(runtime, "retained_paths", {})),
                    "requirements": deepcopy(context.requirements),
                    "resource_revisions": context.revisions(),
                    "injection_status": "pending",
                }
                actor = context.resources[rid]
                actor.executors.clear()
                actor.revision += 1
                actor.evidence = scenario + "; execution unavailable"
                context.unavailable_resources.add(rid)
                context.revision += 1
                if scenario == "Conveyor breakdown":
                    context.inputs["scene"]["Conveyor"]["transport_enabled"] = False
                runtime.stop(scenario)
                runtime.outcome.update(
                    status="blocked", reason=scenario, failed_resource=rid,
                    failure_evidence=deepcopy(self.evidence),
                    cancelled_tasks=deepcopy(self.evidence["pending_tasks"]),
                )
            self._injection_task = asyncio.create_task(self._finish_injection())
        # Operator cancellation must not discard evidence while a detach is in flight.
        return await asyncio.shield(self._injection_task)

    async def _finish_injection(self) -> dict:
        runtime, configuration = self.runtime, self.configuration
        context = runtime.context
        rid, scenario = configuration["resource_id"], configuration["scenario"]
        try:
            await runtime.cancel_owned()
            if scenario == "Part slippage":
                from cais_spade_llm.recovery_framework.failure_effects import slip_part

                await slip_part(runtime, configuration, self.evidence)
            elif scenario == "ur5e-1 breakdown":
                from cais_spade_llm.recovery_framework.failure_effects import observe_robot

                self.evidence["robot_pose"] = await observe_robot(runtime, rid)
            if scenario == "Machining breakdown during part processing" and self.evidence.get("source") != "gazebo_workholding_observation":
                raise ValueError(self.evidence.get("observation_error", "Machine checkpoint observation was interrupted"))
            self.evidence["injection_status"] = "completed"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
            self.evidence.update(injection_status="failed", error=str(exc),
                                 physical_state_reconciliation_required=True)
            logger.warning("%s injection requires reconciliation: %s", scenario, exc)
        except asyncio.CancelledError:
            self.evidence.update(injection_status="interrupted",
                                 physical_state_reconciliation_required=True)
            raise
        finally:
            agent = getattr(runtime, "agent", None)
            if agent is not None:
                agent.part_tracker = deepcopy(context.part_tracker)
            self.evidence["resource_values_after"] = context.snapshot()
            self.evidence["part_tracker_after"] = deepcopy(context.part_tracker)
            runtime.outcome["failure_evidence"] = deepcopy(self.evidence)
            runtime.queue_save()
        try:
            self.visual = await asyncio.to_thread(marker, self.marker_scene(), "show")
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            self.visual = {"status": "failed", "error": str(exc)}
        self.revision += 1
        if self.visual.get("status") != "completed":
            logger.warning("%s is latched but Gazebo marker failed: %s", scenario, self.visual)
        runtime.queue_save()
        return self.snapshot()

    async def after_acknowledgement(self) -> bool:
        """Latch before the product loop negotiates or dispatches another task."""
        if self.status == "armed" and self.checkpoint() is not None:
            await self.trigger()
            return True
        return False

    def not_reached(self) -> None:
        """Record an armed checkpoint that nominal execution could not establish."""
        if self.status == "armed" and not self.evidence.get("physical_state_reconciliation_required"):
            self.evidence = {"injection_status": "not_reached",
                             "reason": self.runtime.outcome.get("reason", "Checkpoint not reached")}
            self.runtime.outcome["failure_evidence"] = deepcopy(self.evidence)
            self.runtime.queue_save()


def reset_fault_scene(bridge, runtime) -> tuple[bool, str]:
    """Recreate the four-arm simulation before clearing its latched breakdown.

    The legacy in-place reset owns xarm6/ur5e controllers, not the recovery
    scene's four independent UR controllers. Recreating the owned scene also
    removes all attachments without commanding a hardware or legacy controller.
    """
    injection = runtime.conveyor_fault._injection_task
    if injection is not None and not injection.done():
        return False, "Failure injection is still recording physical effects; retry reset after it finishes."
    for name in ("recovery_rviz", "gazebo_dual"):
        error = bridge.ros2_stop(name)
        if error:
            return False, f"Recovery scene reset could not stop {name}: {error}"
    error = bridge.ros2_start("gazebo_dual")
    if error:
        return False, f"Recovery scene reset could not restart Gazebo: {error}"
    deadline = time.monotonic() + 120
    while True:
        ready, reason = bridge.simulation_start_ready()
        if ready:
            break
        if time.monotonic() >= deadline:
            return False, "Recovery scene reset is not ready: " + reason
        time.sleep(0.5)
    cleared = marker(runtime.conveyor_fault.marker_scene(), "clear")
    if cleared.get("status") != "completed":
        return (
            False,
            "Recovery scene restarted but breakdown marker absence could not be confirmed: "
            + str(cleared),
        )
    runtime.conveyor_fault.status = "reset"
    runtime.conveyor_fault.revision += 1
    runtime.conveyor_fault.visual = cleared
    runtime.conveyor_fault._robot_task = None
    runtime.conveyor_fault._robot_evidence = None
    runtime.conveyor_fault._robot_pickups = None
    if runtime.conveyor_fault._control_directory is not None:
        runtime.conveyor_fault._control_directory.cleanup()
        runtime.conveyor_fault._control_directory = None
    return (
        True,
        "Recovery Gazebo scene restarted; failure marker cleared. Start System creates a fresh run.",
    )
