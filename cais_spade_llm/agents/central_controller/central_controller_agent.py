# agents/central_controller/central_controller_agent.py

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional, Iterable

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from agents.shared_information.llm_agent import LlmAgent
from agents.central_controller.safety_logic import SafetyLogic
from agents.central_controller.safety_monitor import SafetyMonitor


class CentralControllerAgent(LlmAgent):
    """
    Central Controller Agent (CCA)

    - Loads NL safety requirements and builds structured rules + LTLf via SafetyLogic
    - Builds per-rule DFAs and wraps them in a SafetyMonitor
    - Runs a generic _Monitor behaviour that listens for 'resource_event' messages
      from ResourceAgents and calls monitor_run(...) to perform runtime safety checks.
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
        # Resource agents (used for grounding capability overviews in safety prompts)
        self.resource_agents = list(resource_agents or [])

        # Where the safety rules and logic will be saved (like ProductAgent plan.json)
        base_safety_dir = Path("cais_spade_llm/safety")
        self.safety_logic_path = base_safety_dir / f"{name}_safety_logic.json"

        # Safety logic scaffolding
        self.safety_logic: Optional[SafetyLogic] = None
        if self.safety_file:
            self.safety_logic = SafetyLogic(self, self.safety_file)

        # In-memory safety rules (for later DFA/LTLf integration)
        self.safety_rules: list[dict[str, Any]] = []

        # Runtime safety monitor (wraps all per-rule DFAs)
        self.safety_monitor: Optional[SafetyMonitor] = None

        # Track which APs are currently considered "running" across resources.
        # This lets the DFA see simultaneous actions: σ = running_aps ∪ candidate_aps.
        self.running_aps: set[str] = set()

        self.logger.info(
            "CentralControllerAgent '%s' initialized. safety_file=%s",
            name,
            str(self.safety_file) if self.safety_file else "(none)",
        )

    async def setup(self) -> None:
        await super().setup()
        self.logger.info("[CCA] setup completed.")

        # One-shot init behaviour (build safety rules once at startup)
        self.add_behaviour(self._InitCCA())

        # Cyclic monitoring behaviour
        self.add_behaviour(self._Monitor())

    # ------------------------------------------------------------------ #
    # Runtime monitoring entry point
    # ------------------------------------------------------------------ #
    async def monitor_run(self, msg) -> None:
        """
        Called by the _Monitor behaviour when a 'resource_event' arrives.

        We use a single message type with a 'status' field:

          - status == "running"  → task is starting (safety check + add APs)
          - otherwise            → task finished (remove APs)

        This avoids separate monitor_finish/resource_done functions.
        """
        if not self.safety_monitor:
            self.logger.warning("[CCA] safety_monitor not initialized.")
            return

        try:
            data = json.loads(msg.body or "{}")
        except Exception:
            self.logger.error("[CCA] Malformed resource_event body.")
            return

        task_id       = data.get("task_id")
        resource_jid  = data.get("resource_jid")
        function_name = data.get("function_name")
        params        = data.get("params") or {}
        status        = data.get("status") or "running"

        if not (resource_jid and function_name):
            self.logger.warning("[CCA] resource_event missing resource_jid or function_name.")
            return

        # Map task → AP labels using SafetyMonitor’s helper
        candidate_aps = self.safety_monitor._map_task_to_aps(
            resource_jid=resource_jid,
            function_name=function_name,
            params=params,
        )

        if not candidate_aps:
            self.logger.info(
                "[CCA] No AP mapping for task=%s (%s, %s, status=%s); skipping.",
                task_id,
                resource_jid,
                function_name,
                status,
            )
            return

        # -----------------------------
        # 1) START event → run safety
        # -----------------------------
        if status == "running":
            running_snapshot = list(self.running_aps)

            allowed, meta = self.safety_monitor.check(
                running_aps=running_snapshot,
                candidate_aps=candidate_aps,
            )

            if not allowed:
                self.logger.warning(
                    "[CCA] SAFETY VIOLATION: task=%s rule=%s aps=%s (running=%s)",
                    task_id,
                    meta.get("violated_rule"),
                    candidate_aps,
                    running_snapshot,
                )

                # -----------------------------
                # LLM REPLAN LOGIC
                # -----------------------------
                # DO NOT block the entire system.
                # Only send this violation info to the replan handler.
                try:
                    await self.handle_safety_violation_for_replan(
                        task_id=task_id,
                        resource_jid=resource_jid,
                        function_name=function_name,
                        params=params,
                        candidate_aps=candidate_aps,
                        running_aps=running_snapshot,
                        violated_rule=meta.get("violated_rule"),
                    )
                except Exception:
                    self.logger.exception("[CCA] Replan handler failed (ignored).")
                # -----------------------------                

                return

            # DFA accepted → mark these APs as running
            self.running_aps.update(candidate_aps)

            self.logger.info(
                "[CCA] Safety OK (start): task=%s aps=%s (now running=%s)",
                task_id,
                candidate_aps,
                sorted(self.running_aps),
            )
            return

        # --------------------------------
        # 2) FINISH event → remove APs
        # --------------------------------
        for ap in candidate_aps:
            if ap in self.running_aps:
                self.running_aps.discard(ap)

        self.logger.info(
            "[CCA] Task finished: task=%s status=%s removed_aps=%s (now running=%s)",
            task_id,
            status,
            candidate_aps,
            sorted(self.running_aps),
        )

    async def handle_safety_violation_for_replan(
        self,
        *,
        task_id: str,
        resource_jid: str,
        function_name: str,
        params: dict,
        candidate_aps: list,
        running_aps: list,
        violated_rule: str,
    ):
        """
        Placeholder for future LLM-driven replanning.

        This method receives:
        - the blocked task
        - its APs
        - the currently running APs
        - the violated rule
        and is responsible for triggering:
        - minimal structural repairs (e.g., add DAG edge TASK_1 -> TASK_6)
        - OR a full LLM replan request.
        
        NOTE:
        This runs OUTSIDE the hot path.
        Do NOT block the robots here — just queue/mark tasks as blocked
        and start async replanning in the background.
        """

        self.logger.info(
            "[CCA] (REPLAN PLACEHOLDER) Received safety violation for task=%s rule=%s",
            task_id,
            violated_rule,
        )

        # Example: mark this task locally so scheduler won't retry until repaired
        # You can extend this with your actual plan manager.
        # self.blocked_tasks.add(task_id)

        # Example structure for what you will send to LLM later
        violation_context = {
            "task": {
                "task_id": task_id,
                "resource_jid": resource_jid,
                "function_name": function_name,
                "params": params,
            },
            "safety": {
                "violated_rule": violated_rule,
                "running_aps": running_aps,
                "candidate_aps": candidate_aps,
            },
            "plan": await self._load_current_plan_if_available(),
        }

        # ---- Future use:
        # await self.llm_replan(violation_context)
        # ----

        # For now, just log the context size
        self.logger.debug(
            "[CCA] (REPLAN PLACEHOLDER) Violation context prepared: keys=%s",
            list(violation_context.keys()),
        )

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ #

    class _Monitor(CyclicBehaviour):
        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            if msg.metadata.get("type") != "resource_event":
                return

            await agent.monitor_run(msg)


    class _InitCCA(OneShotBehaviour):
        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent

            safety_logic = agent.safety_logic
            if not safety_logic:
                agent.logger.warning("[CCA] No SafetyPlanner configured.")
                return

            safety_text = safety_logic.load_nl_safety_text()
            if not safety_text:
                agent.logger.warning("[CCA] No NL safety text.")
                return

            # 1. Build structured rules + APs + LTLf
            await safety_logic.build_safety_rules_and_logic(safety_text)

            # 2. Save JSON
            safety_logic.save(agent.safety_logic_path)

            # 3. Store rules in memory
            agent.safety_rules = safety_logic.rules

            # 4. Build DFAs for runtime monitoring (one DOT per rule)
            dfa_map = safety_logic.build_dfas_per_rule()  # { "SAFE_1": dot_str, "SAFE_2": dot_str, ... }

            # 5. Create runtime SafetyMonitor on the agent
            agent.safety_monitor = SafetyMonitor(dfa_map, agent.safety_rules)

            agent.logger.info(
                "[CCA] _InitSafety completed. %d rules, %d DFAs.",
                len(agent.safety_rules),
                len(dfa_map),
            )
