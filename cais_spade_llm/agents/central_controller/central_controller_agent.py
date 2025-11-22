# agents/central_controller/central_controller_agent.py

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional, Iterable

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from agents.shared_information.llm_agent import LlmAgent
from agents.central_controller.safety_logic import SafetyLogic


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

            # 4. Build DFA for runtime monitoring
            dfa = safety_logic.build_dfa()
            agent.dfa = dfa

            agent.logger.info("[CCA] _InitSafety completed.")


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
