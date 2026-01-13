from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Iterable

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message

from agents.shared_information.llm_agent import LlmAgent
from agents.central_controller.safety_logic import SafetyLogic
# Import the updated monitor
from agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from agents.central_controller.offline_fsa_safety_validator import OfflineFsaSafetyValidator

class CentralControllerAgent(LlmAgent):
    """
    Central Controller Agent (CCA)
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
        
        # NOTE: self.running_aps is removed; the monitor tracks it now.
        
        self.blocked_tasks: dict[str, dict[str, Any]] = {}

        self.logger.info(
            "CentralControllerAgent '%s' initialized. safety_file=%s",
            name, str(self.safety_file)
        )

    async def setup(self) -> None:
        await super().setup()
        self.logger.info("[CCA] setup completed.")
        self.add_behaviour(self._InitCCA())
        self.add_behaviour(self._Monitor())
        self.add_behaviour(self._PlanValidation())

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

            if msg.metadata.get("type") != "resource_event":
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

            # 2. HANDLE 'SAFETY_CHECK' (Start Event)
            if status == "safety_check":
                #
                allowed, info = agent.safety_monitor.process_start_event(event)

                if not allowed:
                    violated_rule = info.get("violated_rule")
                    running_snapshot = info.get("running_snapshot", [])
                    
                    agent.logger.warning(
                        "[CCA] SAFETY VIOLATION: task=%s rule=%s (running=%s)",
                        task_id, violated_rule, running_snapshot
                    )

                    # Queue the task to retry later
                    if not hasattr(agent, "blocked_tasks"):
                        agent.blocked_tasks = {}
                    
                    agent.blocked_tasks[task_id] = {
                        "event": event,
                        "violated_rule": violated_rule
                    }
                    
                    agent.logger.info(
                        "[CCA] Queued task=%s as temporarily unsafe.", task_id
                    )
                    # Do NOT send a reply yet; resource waits.
                    return

                # If allowed
                agent.logger.info("[CCA] Safety OK: task=%s allowed.", task_id)
                await self._send_decision(resource_jid, task_id, "allow")
                return

            # 3. HANDLE 'FINISHED' / 'FAILED' (End Event)
            if status in ["finished", "failed"]:
                #
                agent.safety_monitor.process_finish_event(event)
                
                agent.logger.info("[CCA] Task %s finished. State updated.", task_id)

                # Retry any blocked tasks now that state has changed
                await self._retry_blocked_tasks()

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

            if msg.metadata.get("type") != "plan_safety_check":
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

            # Delegate to Offline FSA Validator
            if agent.safety_logic and agent.safety_logic.rule_dfas:
                validator = OfflineFsaSafetyValidator(
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
