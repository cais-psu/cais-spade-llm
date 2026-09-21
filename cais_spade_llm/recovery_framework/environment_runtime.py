"""Runtime environmental matching behind the existing Start System surface."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from cais_spade_llm.agents.shared_information.environment_capabilities import (
    CapabilityReplyInbox,
    explore,
    message,
)
from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message
from cais_spade_llm.product.environment import EnvironmentProductContext, fingerprint
from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.resources.environment_models import (
    feasibility,
    process_json,
    project_transition,
)

logger = logging.getLogger(__name__)
RUN_DIRECTORY = ROOT / "cais_spade_llm/monitor/environment_runs"
_prepared: dict | None = None


def prepare_environment_start(setup: dict | None) -> None:
    """Capture saved matching inputs without searching or commanding a resource."""
    global _prepared
    if setup is None:
        _prepared = None
        return
    from cais_spade_llm.ui.recovery_setup import validate_setup

    inputs = validate_setup(setup)
    if "completion_conditions" in inputs["product_order"]:
        raise ValueError("Explicit delivery orders use their existing delivery runtime")
    # Validate the product contract before replacing any prepared startup input.
    context = EnvironmentProductContext(
        inputs["scene"], inputs["product_order"], inputs["geometry"], setup["permitted_resources"]
    )
    _prepared = {"setup": deepcopy(setup), "inputs": context.inputs}


def prepared_environment_start() -> dict | None:
    """Return the independent factory input for the requested Start System."""
    return deepcopy(_prepared)


class EnvironmentRuntime:
    """Connect ProductAgent discovery, CCA checks, task dispatch, and evidence."""

    def __init__(self, agent, prepared: dict, resources: list) -> None:
        self.agent = agent
        self.product_jid = str(agent.jid).split("/", 1)[0]
        self.context = EnvironmentProductContext(
            **prepared["inputs"], permitted_resources=prepared["setup"]["permitted_resources"]
        )
        self.jids = {
            resource.agent_name: str(resource.jid).split("/", 1)[0] for resource in resources
        }
        if set(self.jids) != set(self.context.models):
            raise ValueError("Environmental matching requires every configured ResourceAgent")
        self.replies: asyncio.Queue = asyncio.Queue()
        self.max_steps = 64
        self.max_search_states = 50_000
        self.stopped = False
        self.active_executions: set[asyncio.Task] = set()
        self.outcome = {"status": "prepared"}
        self.path = RUN_DIRECTORY / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + self.context.run_id[:8]
        )
        for resource in resources:
            resource.environment_runtime = self
            resource.environment_context = self.context.resources[resource.agent_name]
        agent.part_tracker = deepcopy(self.context.part_tracker)

    def snapshot(self) -> dict:
        """Expose read-only runtime models, environmental graph, and progress."""
        return {
            "models": {
                rid: resource.descriptor() for rid, resource in self.context.resources.items()
            },
            "environment_model": deepcopy(self.context.environment_model),
            "outcome": deepcopy(self.outcome),
            "product_states": deepcopy(self.context.part_tracker),
            "requirements": deepcopy(self.context.requirements),
            **({"processPlan": deepcopy(self.context.product_order["processPlan"])}
               if self.context.schema_version == 3 else {}),
            "revision": self.context.revision,
            "run_id": self.context.run_id,
        }

    def snapshot_revision(self) -> tuple:
        """Read display revisions without copying resource models or path graphs."""
        environment = self.context.environment_model
        return (
            self.context.run_id,
            self.context.revision,
            tuple(
                (
                    rid,
                    resource.revision,
                    resource.model.get("configuration_revision"),
                    fingerprint(resource.model.get("current_configuration", {})),
                    resource.evidence,
                    tuple(sorted(resource.executors)),
                )
                for rid, resource in self.context.resources.items()
            ),
            environment.get("request_id"),
            environment.get("status"),
            environment.get("closed"),
            *(
                len(environment.get(field, []))
                for field in ("nodes", "edges", "bids", "rejections")
            ),
            id(environment.get("selected_path")),
            id(self.outcome),
        )

    def save(self) -> None:
        """Save versioned evidence and process JSON from the actual runtime models."""
        self.path.mkdir(parents=True, exist_ok=True)
        report = {**self.context.report(), "outcome": deepcopy(self.outcome)}
        temporary = self.path / "run.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.path / "run.json")
        process_directory = self.path / "processes"
        process_directory.mkdir(exist_ok=True)
        names = {e["event_name"] for model in self.context.models.values() for e in model["events"]}
        for name in sorted(names):
            path = process_directory / f"{name}.json"
            content = json.dumps(process_json(self.context.models, name), indent=2) + "\n"
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(content, encoding="utf-8")
                temporary.replace(path)

    def stop(self, reason: str = "Stopped by operator") -> None:
        """Close discovery and suppress execution without altering acknowledged state."""
        self.stopped = True
        self.outcome = {"status": "stopped", "reason": reason}
        self.context.environment_model["closed"] = True
        for execution in tuple(self.active_executions):
            execution.get_loop().call_soon_threadsafe(execution.cancel)

    def install(self, agent) -> None:
        """Install matching and execution inboxes instead of the legacy order planner."""
        replies = Template(metadata={"type": "capability_reply"})
        agent.add_behaviour(CapabilityReplyInbox(), replies)
        execution = (
            Template(metadata={"type": "plan_safety_result"})
            | Template(metadata={"type": "ack"})
            | Template(metadata={"type": "replan_request"})
        )
        agent.add_behaviour(EnvironmentProductLoop(), execution)

    def set_plan(self, task: dict) -> None:
        """Build the existing DAG/FSA interface for one rechecked next task."""
        name = f"environment_{self.context.revision + 1}"
        actor = self.context.resources[task["resource_id"]]
        after, _ = project_transition(
            self.context.models,
            self.context.snapshot(),
            self.context.part_tracker,
            task,
            self.context.product_name,
            self.context.requirements,
        )
        node = {
            "id": name,
            "type": "task",
            "requirement_id": "environment",
            "function_name": task["event_name"],
            "params": deepcopy(task["parameters"]),
            "resource_jid": self.jids[task["resource_id"]],
            "predecessors": [],
            "successors": [],
            "sequence_index": 0,
            "status": "pending",
            "environment_task": deepcopy(task),
            "in_state": actor.valuation.get("resource_state", "idle"),
            "out_state": after[task["resource_id"]].get("resource_state", "idle"),
        }
        self.agent.process_planner.nodes = [node]
        self.agent.process_planner.compile_global_fsa()


class EnvironmentProductLoop(CyclicBehaviour):
    """Explore at runtime, then execute only supported and safety-checked tasks."""

    async def run(self) -> None:
        runtime = self.agent.environment_runtime
        signature = fingerprint(
            [
                runtime.context.revision,
                runtime.context.revisions(),
                runtime.context.permitted_resources,
                runtime.context.geometry,
            ]
        )
        if runtime.stopped or getattr(self, "_last_attempt", None) == signature:
            await asyncio.sleep(0.25)
            return
        self._last_attempt = signature
        try:
            await self.work(runtime)
        except (
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
        ) as exc:
            runtime.outcome = {"status": "blocked", "reason": str(exc)}
            logger.warning("Environmental run blocked: %s", exc)
        finally:
            self._last_attempt = fingerprint(
                [
                    runtime.context.revision,
                    runtime.context.revisions(),
                    runtime.context.permitted_resources,
                    runtime.context.geometry,
                ]
            )
            runtime.save()
            self.agent._set_kickoff_result(
                success=True,
                message=f"Environmental matching: {runtime.outcome['status']}. See Resources for offers and missing evidence.",
                retries_used=0,
                retries_max=0,
                violations=[],
            )

    async def work(self, runtime: EnvironmentRuntime) -> None:
        while not runtime.stopped:
            runtime.outcome = {"status": "exploring"}
            result = await explore(runtime, self)
            runtime.outcome = deepcopy(result)
            runtime.save()
            if result["status"] != "planned":
                return
            task = result["tasks"][0]
            if not result["executable"]:
                runtime.outcome.update(
                    status="execution_unavailable",
                    reason="The selected path requires controllers and completion validators listed in execution_unavailable",
                )
                return
            selected_revisions = runtime.context.revisions()
            runtime.set_plan(task)
            request_id = uuid4().hex
            payload = self.agent._build_plan_validation_payload(
                skip_revalidation=False, request_id=request_id
            )
            await send_agent_message(
                self, message(self.agent.cca_jid, "plan_safety_check", payload)
            )
            decision = await self.receive_from(
                self.agent.cca_jid, "plan_safety_result", timeout=60, request_id=request_id
            )
            if decision.get("ok") is not True:
                runtime.outcome = {
                    "status": "blocked",
                    "reason": "CCA rejected the selected transition",
                    "details": decision,
                }
                return
            if selected_revisions != runtime.context.revisions():
                continue
            pending = runtime.context.prepare(task)
            await send_agent_message(
                self, message(runtime.jids[task["resource_id"]], "task", pending)
            )
            while not runtime.stopped:
                ack = await self.receive_from(runtime.jids[task["resource_id"]], "ack", timeout=300)
                if ack.get("task_id") != pending["task_id"]:
                    continue
                if ack.get("status") in {"accepted", "running"}:
                    continue
                if ack.get("status") != "completed":
                    runtime.outcome = {
                        "status": "blocked",
                        "reason": "Resource did not complete the transition",
                        "details": ack,
                    }
                    return
                runtime.context.acknowledge(ack["acknowledgement"])
                self.agent.part_tracker = deepcopy(runtime.context.part_tracker)
                self.agent.process_planner.nodes[0]["status"] = "completed"
                runtime.save()
                break

    async def receive_from(
        self, sender: str, kind: str, *, timeout: float, request_id: str | None = None
    ) -> dict:
        """Authenticate control replies and honor CCA interruptions while waiting."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = await self.receive(timeout=min(1, deadline - time.monotonic()))
            if self.agent.environment_runtime.stopped:
                raise ValueError("Environmental run stopped")
            if msg is None:
                continue
            actual = str(msg.sender).split("/", 1)[0]
            if (
                msg.metadata.get("type") == "replan_request"
                and actual == str(self.agent.cca_jid).split("/", 1)[0]
            ):
                self.agent.environment_runtime.stop("CCA interrupted execution")
                raise ValueError("CCA interrupted execution")
            if msg.metadata.get("type") == kind and actual == str(sender).split("/", 1)[0]:
                payload = json.loads(msg.body)
                if request_id is None or payload.get("request_id") == request_id:
                    return payload
        raise TimeoutError(f"Timed out awaiting {kind}")


async def execute_environment_task(behaviour, msg, task: dict) -> None:
    """Execute a declared task only after resource checks and CCA permission."""
    agent = behaviour.agent
    runtime = getattr(agent, "environment_runtime", None)
    if runtime is None or str(msg.sender).split("/", 1)[0] != runtime.product_jid:
        return
    context, actor = runtime.context, agent.environment_context
    task_id = task.get("task_id")

    async def ack(status: str, **extra) -> None:
        await send_agent_message(
            behaviour,
            message(runtime.product_jid, "ack", {"task_id": task_id, "status": status, **extra}),
        )

    try:
        if task_id in actor.validated_completions:
            expected = {**task, "status": "completed"}
            if expected != actor.validated_completions[task_id]:
                raise ValueError("Conflicting repeated task")
            await ack("completed", acknowledgement=expected)
            return
        if runtime.stopped or context.pending != task or task["resource_id"] != actor.resource_id:
            raise ValueError("Task does not match current prepared execution")
        name = task["event_name"]
        if name not in actor.executors or task["resource_revisions"] != context.revisions():
            raise ValueError("Controller unavailable or resource configuration changed")
        part = task["parameters"].get("part_name") or context.outstanding()[0]
        status, reasons = feasibility(actor.model, task, context.geometry.get(part, {}))
        if status != "FEASIBLE":
            raise ValueError("; ".join(reasons))
        project_transition(
            context.models,
            context.snapshot(),
            context.part_tracker,
            task,
            context.product_name,
            context.requirements,
        )
        if (
            await actor.start_validators[name](
                deepcopy(task), context.snapshot(), context.geometry.get(part, {})
            )
            is not True
        ):
            raise ValueError("Resource start conditions have not been validated")
        await send_agent_message(
            behaviour,
            message(
                agent.cca_jid,
                "resource_event",
                {
                    "task_id": task_id,
                    "resource_jid": str(agent.jid),
                    "function_name": name,
                    "params": task["parameters"],
                    "status": "safety_check",
                },
            ),
        )
        decision = await asyncio.wait_for(agent._wait_for_safety_decision(task_id), 60)
        if (
            decision != "allow"
            or runtime.stopped
            or task["resource_revisions"] != context.revisions()
        ):
            raise ValueError("CCA permission missing or execution context changed")
        await ack("running")
        await send_agent_message(
            behaviour,
            message(
                agent.cca_jid,
                "resource_event",
                {
                    "task_id": task_id,
                    "resource_jid": str(agent.jid),
                    "function_name": name,
                    "params": task["parameters"],
                    "status": "running",
                },
            ),
        )
        execution = asyncio.create_task(actor.executors[name](deepcopy(task)))
        runtime.active_executions.add(execution)
        try:
            observations = await asyncio.wait_for(execution, agent.tool_timeout_s)
        finally:
            runtime.active_executions.discard(execution)
        if runtime.stopped or actor.completion_validators[name](task, observations) is not True:
            raise ValueError("Resource completion evidence did not validate")
        if task["resource_revisions"] != context.revisions():
            raise ValueError("Resource state changed during execution")
        completed = {**task, "status": "completed"}
        actor.validated_completions[task_id] = completed
        actor.completion_observations[task_id] = deepcopy(observations)
        await send_agent_message(
            behaviour,
            message(
                agent.cca_jid,
                "resource_event",
                {
                    "task_id": task_id,
                    "resource_jid": str(agent.jid),
                    "function_name": name,
                    "params": task["parameters"],
                    "status": "completed",
                },
            ),
        )
        await ack("completed", acknowledgement=completed)
    except (ValueError, KeyError, TypeError, TimeoutError, asyncio.TimeoutError) as exc:
        await ack("blocked", reason=str(exc))
