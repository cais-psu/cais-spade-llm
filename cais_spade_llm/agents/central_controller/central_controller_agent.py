"""Central Controller Agent (CCA) orchestrating safety checks and replanning."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Iterable

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent
from agents.central_controller.safety_logic import SafetyLogic
# Import the updated monitor
from agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
from cais_spade_llm.agents.central_controller.offline_safety_validator import OfflineSafetyValidator

class CentralControllerAgent(LlmAgent):
    """
    Central Controller Agent (CCA).
    Coordinates safety validation, online monitoring, and replanning signals.
    """

    agent_role = "controller"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        resource_agents: Optional[Iterable[Any]] = None,
        safety_file: str | None = None,
        **kw: Any,
    ) -> None:
        """Initialize controller state, safety logic, and monitoring scaffolding."""
        super().__init__(jid, password, name=name, agent_role="controller", **kw)

        self.agent_name = name
        self.safety_file = Path(safety_file) if safety_file else None
        self.resource_agents = list(resource_agents or [])
        
        base_safety_dir = Path("cais_spade_llm/safety")
        self.safety_logic_path = base_safety_dir / f"{name}_safety_logic.json"

        self.safety_logic: Optional[SafetyLogic] = None
        if self.safety_file:
            self.safety_logic = SafetyLogic(self, self.safety_file)

        self.safety_rules: list[dict[str, Any]] = []
        
        # We use the NEW OnlineSafetyMonitor
        self.safety_monitor: Optional[OnlineSafetyMonitor] = None

        # Runtime Plan FSA monitor
        self.plan_fsa_monitor: Optional[OnlineFsaMonitor] = None
        
        # NOTE: self.running_aps is removed; the monitor tracks it now.
        
        self.blocked_tasks: dict[str, dict[str, Any]] = {}

        self.logger.info(
            "CentralControllerAgent '%s' initialized. safety_file=%s",
            name, str(self.safety_file)
        )

    async def setup(self) -> None:
        """Attach startup, runtime monitor, and offline plan validation behaviours."""
        await super().setup()
        self.logger.info("[CCA] setup completed.")
        self.add_behaviour(self._InitCCA())
        t_resource = Template()
        t_resource.set_metadata("type", "resource_event")
        self.add_behaviour(self._Monitor(), t_resource)

        t_plan = Template()
        t_plan.set_metadata("type", "plan_safety_check")
        self.add_behaviour(self._PlanValidation(), t_plan)

    async def _send_replan_request(
        self,
        *,
        product_jid: str,
        reason: str,
        event: dict[str, Any],
        safety_info: Optional[dict[str, Any]],
    ) -> None:
        """Send a structured replanning request to a ProductAgent."""
        plan_ctx = {}
        if self.plan_fsa_monitor:
            plan_ctx = self.plan_fsa_monitor.build_replan_context(
                failure_event={
                    "failed_task_id": event.get("task_id"),
                    "task_id": event.get("task_id"),
                }
            )

        # Failure-focused safety context: keep only what helps replanning decisions.
        safety_ctx = {}
        if safety_info:
            safety_ctx = {
                "violated_rule_id": safety_info.get("violated_rule"),
                "violated_from": safety_info.get("violated_from"),
                "violated_to": safety_info.get("violated_to"),
                "candidate_aps": safety_info.get("candidate_aps") or [],
                "running_aps": safety_info.get("running_snapshot") or [],
            }

        # Debug log: capture the full context we are about to send for replanning.
        # Keep logs bounded to avoid flooding if the context grows large.
        self.logger.info(
            "[CCA] Replan request -> %s reason=%s task_id=%s plan_ctx=%s safety_ctx=%s",
            product_jid,
            reason,
            event.get("task_id"),
            json.dumps(plan_ctx, ensure_ascii=False)[:2000],
            json.dumps(safety_ctx, ensure_ascii=False)[:2000],
        )

        msg = Message(to=str(product_jid))
        msg.set_metadata("type", "replan_request")
        msg.body = json.dumps(
            {
                "reason": reason,
                "event": event,
                "plan_ctx": plan_ctx,
                "safety_ctx": safety_ctx,
            }
        )
        await self.send(msg)

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ #
    class _Monitor(CyclicBehaviour):
        """
        Listen for 'resource_event', delegate logic to OnlineSafetyMonitor,
        and send decisions back to resources.
        """

        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            # Safety guard: ensure monitor is loaded
            if not agent.safety_monitor:
                agent.logger.warning("[CCA] SafetyMonitor not loaded yet.")
                return

            # 1. DELEGATE PARSING to the Monitor
            #
            event = agent.safety_monitor.parse_resource_event(msg)
            if not event:
                return

            task_id = event["task_id"]
            status = event["status"]
            resource_jid = event["resource_jid"]
            function_name = event["function_name"]
            product_jid = (event.get("params") or {}).get("product_jid")

            # 2. HANDLE 'SAFETY_CHECK' (Start Event)
            if status == "safety_check":
                await self._handle_safety_check(
                    event=event,
                    task_id=task_id,
                    resource_jid=resource_jid,
                    product_jid=product_jid,
                )
                return

            await self._handle_runtime_event(
                event=event,
                task_id=task_id,
                status=status,
                resource_jid=resource_jid,
                function_name=function_name,
                product_jid=product_jid,
            )

        async def _handle_safety_check(
            self,
            *,
            event: dict[str, Any],
            task_id: str,
            resource_jid: str,
            product_jid: Optional[str],
        ) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            allowed, info = agent.safety_monitor.process_start_event(event)

            if not allowed:
                violated_rule = info.get("violated_rule")
                running_snapshot = info.get("running_snapshot", [])

                agent.logger.warning(
                    "[CCA] SAFETY VIOLATION: task=%s rule=%s (running=%s)",
                    task_id, violated_rule, running_snapshot
                )

                # Queue the task to retry later
                agent.blocked_tasks[task_id] = {
                    "event": event,
                    "violated_rule": violated_rule
                }

                agent.logger.info(
                    "[CCA] Queued task=%s as temporarily unsafe.", task_id
                )

                if product_jid:
                    await agent._send_replan_request(
                        product_jid=product_jid,
                        reason="safety_block",
                        event=event,
                        safety_info=info,
                    )
                # Do NOT send a reply yet; resource waits.
                return

            # If allowed
            agent.logger.info("[CCA] Safety OK: task=%s allowed.", task_id)
            await self._send_decision(resource_jid, task_id, "allow")

        async def _handle_runtime_event(
            self,
            *,
            event: dict[str, Any],
            task_id: str,
            status: str,
            resource_jid: str,
            function_name: str,
            product_jid: Optional[str],
        ) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            # ----- PLAN FSA TRACE: START EVENT ----- #
            if status == "running" and agent.plan_fsa_monitor:
                agent.plan_fsa_monitor.process_event(
                    event_type="start",
                    task_id=task_id,
                    function_name=function_name,
                    resource_jid=resource_jid,
                    status=status,
                )

            # 3. HANDLE 'FINISHED' / 'FAILED' (End Event)
            # Treat any "failed:*" status as a failed end event.
            is_failed = (status == "failed") or (isinstance(status, str) and status.startswith("failed"))
            is_completed = status in ("completed", "finished")
            if is_completed or is_failed:
                # ----- PLAN FSA TRACE: END EVENT ----- #
                if agent.plan_fsa_monitor:
                    agent.plan_fsa_monitor.process_event(
                        event_type=("fail" if is_failed else "done"),
                        task_id=task_id,
                        function_name=function_name,
                        resource_jid=resource_jid,
                        status=status,
                    )
                if is_failed:
                    agent.safety_monitor.process_fail_event(event)
                    agent.logger.info("[CCA] Task %s failed. State updated.", task_id)
                else:
                    agent.safety_monitor.process_finish_event(event)
                    agent.logger.info("[CCA] Task %s finished. State updated.", task_id)

                # Retry any blocked tasks now that state has changed
                await self._retry_blocked_tasks()

                # Deadlock detection:
                # no enabled starts, not in a marked state, and nothing currently running.
                if agent.plan_fsa_monitor and product_jid:
                    pm = agent.plan_fsa_monitor
                    cur_state = pm.current_state
                    if cur_state:
                        marked = set((pm.fsa or {}).get("A", {}).get("Xm") or [])
                        if cur_state not in marked:
                            resource_state = pm._parse_state(cur_state)
                            any_running = any(
                                info.get("status") == "running"
                                for info in resource_state.values()
                            )
                            if not any_running:
                                next_task_ids = pm._next_task_ids_from_state(cur_state)
                                if not next_task_ids:
                                    await agent._send_replan_request(
                                        product_jid=product_jid,
                                        reason="deadlock",
                                        event=event,
                                        safety_info=None,
                                    )

        async def _retry_blocked_tasks(self) -> None:
            """
            Iterate through blocked tasks and check if they are now allowed.
            """
            agent: "CentralControllerAgent" = self.agent # type: ignore
            
            if not agent.blocked_tasks or not agent.safety_monitor:
                return

            to_release = []

            # Check all blocked tasks against the NEW state
            for task_id, data in agent.blocked_tasks.items():
                event = data["event"]
                
                # Check again
                allowed, info = agent.safety_monitor.process_start_event(event)

                if allowed:
                    agent.logger.info(
                        "[CCA] Unblocking task=%s (rule %s no longer violated).",
                        task_id, data.get("violated_rule")
                    )
                    # Send allow decision
                    await self._send_decision(event["resource_jid"], task_id, "allow")
                    to_release.append(task_id)
                else:
                    # Still blocked, do nothing
                    pass

            # Cleanup
            for tid in to_release:
                agent.blocked_tasks.pop(tid, None)

        async def _send_decision(self, to_jid: str, task_id: str, decision: str):
            msg = Message(to=to_jid)
            msg.set_metadata("type", "safety_decision")
            msg.body = json.dumps({"task_id": task_id, "decision": decision})
            await self.send(msg)


    class _InitCCA(OneShotBehaviour):
        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            safety_logic = agent.safety_logic
            if not safety_logic:
                agent.logger.warning("[CCA] No SafetyPlanner configured.")
                return

            safety_text = safety_logic.load_nl_safety_text()
            if not safety_text:
                agent.logger.warning("[CCA] No NL safety text.")
                return

            await safety_logic.build_safety_rules_and_logic(safety_text)
            safety_logic.save(agent.safety_logic_path)

            agent.safety_rules = safety_logic.rules
            dfa_map = safety_logic.build_dfas_per_rule()

            # Initialize the NEW OnlineSafetyMonitor
            #
            agent.safety_monitor = OnlineSafetyMonitor(dfa_map, agent.safety_rules)

            agent.logger.info(
                "[CCA] _InitCCA completed. Monitor online with %d rules.",
                len(agent.safety_rules)
            )

    class _PlanValidation(CyclicBehaviour):
        """
        Behaviour that listens for 'plan_safety_check', uses OfflineFsaSafetyValidator
        to validate a compiled FSA plan offline, and replies with the result.
        """

        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                data = json.loads(msg.body or "{}")
                fsa = data.get("fsa")            # REQUIRED
                plan = data.get("plan")          # OPTIONAL (semantic AP mapping)
                product_jid = data.get("product_jid")
            except Exception:
                agent.logger.exception("[CCA] Malformed plan_safety_check.")
                return

            if not fsa:
                agent.logger.warning("[CCA] No FSA provided for offline validation.")
                return

            # Initialize plan FSA monitor for runtime tracing
            agent.plan_fsa_monitor = OnlineFsaMonitor(fsa)

            # Delegate to Offline FSA Validator
            if agent.safety_logic and agent.safety_logic.rule_dfas:
                validator = OfflineSafetyValidator(
                    rules=agent.safety_rules,
                    dfa_map=agent.safety_logic.rule_dfas
                )

                ok, violations = validator.validate_fsa_offline(
                    fsa=fsa,
                    plan=plan,
                    product_jid=product_jid
                )
            else:
                agent.logger.warning("[CCA] Safety logic not ready; skipping validation.")
                ok, violations = True, []

            # ---- NEW: log summary + details ----
            violated_rules = sorted({v.get("violated_rule_id") for v in violations if v.get("violated_rule_id")})
            agent.logger.info(
                "[CCA] Offline FSA Validation: %s (Violated rules: %d, Witnesses: %d) product=%s",
                "OK" if ok else "FAIL",
                len(violated_rules),
                len(violations),
                product_jid,
            )

            if not ok and violations:
                # Cap to avoid log spam
                max_witnesses = 3
                for i, v in enumerate(violations[:max_witnesses], start=1):
                    agent.logger.error(
                        "[CCA] VIOLATION #%d | rule=%s | %s | ltlf=%s",
                        i,
                        v.get("violated_rule_id"),
                        v.get("violation_text"),
                        v.get("violation_logic"),
                    )
                    agent.logger.error(
                        "[CCA]   witness_task_ids=%s witness_events=%s",
                        v.get("witness_task_ids"),
                        v.get("witness_events"),
                    )

                    # Per-transition debug (only present if you added _sigma/_q_from/_q_to in validator)
                    for t in v.get("witness_transitions", [])[:50]:
                        agent.logger.error(
                            "[CCA]     ↳ from=%s --%s--> %s | task=%s | sigma=%s | DFA:%s→%s",
                            t.get("from"),
                            t.get("event"),
                            t.get("to"),
                            t.get("task_id"),
                            t.get("_sigma"),   # may be None if you didn't patch validator
                            t.get("_q_from"),  # may be None if you didn't patch validator
                            t.get("_q_to"),    # may be None if you didn't patch validator
                        )

                if len(violations) > max_witnesses:
                    agent.logger.error(
                        "[CCA] ... %d more witness(es) suppressed",
                        len(violations) - max_witnesses
                    )
            # ---- END NEW ----

            # Reply
            try:
                reply = msg.make_reply()
                reply.set_metadata("type", "plan_safety_result")
                reply.body = json.dumps({
                    "ok": ok,
                    "violations": violations
                })
                await self.send(reply)
            except Exception:
                agent.logger.exception("Failed to send reply.")
