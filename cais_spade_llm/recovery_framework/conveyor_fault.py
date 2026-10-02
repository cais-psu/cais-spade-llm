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
        from cais_spade_llm.recovery_framework.failure_checkpoints import CHECKPOINTS

        self.runtime = runtime
        self.configuration = deepcopy(setup.get("failure_scenario"))
        self.enabled = self.configuration is not None
        if self.enabled and (
            setup.get("execution_mode") != "simulation"
            or CHECKPOINTS.get(self.configuration.get("scenario")) != self.configuration.get("checkpoint")
        ):
            raise ValueError("Only configured observed failure checkpoints are integrated in simulation")
        self.status = "armed" if self.enabled else "disabled"
        self.revision = 0
        self.evidence: dict = {}
        self.visual: dict = {}
        self._machine_evidence: dict | None = None
        self._machine_task: dict | None = None
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
        return checkpoint(self.runtime.context, self.configuration)

    def holds_task(self, task: dict) -> bool:
        """Retain each selected part after pickup until both robots hold their parts."""
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

    async def retain_worker_interruption(self) -> None:
        """Retain a clock-bound fault even when Stop cancelled the worker reply."""
        if self._control_directory is None or self._machine_task is None or self.status == "triggered":
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
                    or (self.runtime.stopped and self._machine_evidence is None)
                    or (self.status != "armed" and self._machine_evidence is None)):
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
        if self.status == "armed":
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
    if runtime.conveyor_fault._control_directory is not None:
        runtime.conveyor_fault._control_directory.cleanup()
        runtime.conveyor_fault._control_directory = None
    return (
        True,
        "Recovery Gazebo scene restarted; failure marker cleared. Start System creates a fresh run.",
    )
