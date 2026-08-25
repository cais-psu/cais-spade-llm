"""Compose the shared ProductAgent behind the Phase 3.1 protocol."""

from __future__ import annotations

from typing import Any

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)

_PRODUCT_AGENT_JID = "spec2primitives_pa@localhost"
_PRODUCT_AGENT_NAME = "spec2primitives_pa"
_PRODUCT_AGENT_INSTRUCTIONS = (
    "For Spec2Primitives, follow the supplied structured needed_context prompt "
    "exactly. Do not plan, contact another agent, or execute robot behavior."
)


class _SharedProductAgentContextRuntime:
    """Expose only structured ProductAgent calls without its SPADE lifecycle."""

    def __init__(self) -> None:
        self._product_agent = ProductAgent(
            _PRODUCT_AGENT_JID,
            "",
            name=_PRODUCT_AGENT_NAME,
            instruction_override=_PRODUCT_AGENT_INSTRUCTIONS,
        )

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate one structured call with no tools or tool executor."""
        return await self._product_agent.ask_llm_structured(
            prompt,
            response_format=response_format,
        )


def create_product_agent_context_runtime() -> ProductAgentContextRuntime:
    """Create the read-only ProductAgent composition used by the PA UI."""
    return _SharedProductAgentContextRuntime()
