# agents/central_controller/central_controller_agent.py

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from agents.shared_information.llm_agent import LlmAgent
from agents.central_controller.safety_planner import SafetyPlanner


class CentralControllerAgent(LlmAgent):
    """
    Central Controller Agent (CCA)

    Placeholder version:

      - Holds path to safety requirements file
      - Owns a SafetyPlanner (NL -> safety rules)
      - Has OneShot behaviour to build safety model at startup
      - Has Cyclic behaviour placeholder for future safety monitoring
    """

    agent_role = "controller"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        safety_file: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(jid, password, name=name, agent_role="controller", **kw)

        self.agent_name = name
        self.safety_file = Path(safety_file) if safety_file else None

        # Where the *structured* safety rules will be saved (like ProductAgent plan.json)
        base_safety_dir = Path("cais_spade_llm/safety")
        self.structured_safety_path = base_safety_dir / f"{name}_safety_requirements.json"

        # Safety planner scaffolding (similar to ProductAgent -> ProcessPlanner)
        self.safety_planner: Optional[SafetyPlanner] = None
        if self.safety_file:
            self.safety_planner = SafetyPlanner(self, self.safety_file)

        # In-memory safety rules (for later DFA/LTLf integration)
        self.safety_rules: list[dict[str, Any]] = []

        self.logger.info(
            "CentralControllerAgent '%s' initialized. safety_file=%s",
            name,
            str(self.safety_file) if self.safety_file else "(none)",
        )

    async def setup(self) -> None:
        await super().setup()
        self.logger.info("[CCA] setup completed.")

        # One-shot init behaviour (build safety rules once at startup)
        self.add_behaviour(self._InitSafety())

        # Cyclic monitoring behaviour (placeholder)
        self.add_behaviour(self._SafetyMonitor())

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ #

    class _InitSafety(OneShotBehaviour):
        """Build the safety rule model once at startup."""

        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore
            agent.logger.info("[CCA] _InitSafety starting.")

            planner = agent.safety_planner
            if not planner:
                agent.logger.warning(
                    "[CCA] No SafetyPlanner configured (missing safety_file)."
                )
                return

            safety_text = planner.load_nl_safety_text()
            if not safety_text:
                agent.logger.warning("[CCA] No NL safety text; _InitSafety aborted.")
                return

            # 1. NL → structured safety rules
            await planner.build_safety_rules(safety_text)

            # 2. Save structured safety (like ProductAgent does for requirements)
            planner.save(agent.structured_safety_path)

            # 3. Keep in memory for runtime use / UI
            agent.safety_rules = planner.rules

            agent.logger.info(
                "[CCA] _InitSafety completed with %d safety rule(s).",
                len(agent.safety_rules),
            )

    class _SafetyMonitor(CyclicBehaviour):
        """
        Placeholder: cyclic behaviour for safety monitoring.

        Later this will:
          - receive safety_query messages from ProductAgents
          - consult DFA/LTLf monitor
          - reply with allowed/blocked
        """

        async def run(self) -> None:
            agent: "CentralControllerAgent" = self.agent  # type: ignore

            # Placeholder: just sleep to keep loop alive
            # Later:
            #   msg = await self.receive(timeout=0.5)
            #   if msg: ... handle safety query ...
            await asyncio.sleep(0.5)
