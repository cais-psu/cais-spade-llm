"""Compose the shared ProductAgent behind the Phase 3.1 protocol."""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.spec2primitives.agents.pa.context_interaction import (
    ProductAgentContextRuntime,
)

_PRODUCT_AGENT_JID = "spec2primitives_pa@localhost"
_PRODUCT_AGENT_NAME = "spec2primitives_pa"
_PRODUCT_AGENT_INSTRUCTIONS = (
    "For Spec2Primitives, investigate only through the supplied controlled tools "
    "and return the requested structured grounding result. Do not contact another "
    "agent or execute robot behavior."
)
_TOOL_TIMEOUT_SEC = 30.0


class _SharedProductAgentContextRuntime:
    """Expose only structured ProductAgent calls without its SPADE lifecycle."""

    def __init__(self, *, model: str) -> None:
        self._product_agent = ProductAgent(
            _PRODUCT_AGENT_JID,
            "",
            name=_PRODUCT_AGENT_NAME,
            instruction_override=_PRODUCT_AGENT_INSTRUCTIONS,
            model=model,
        )

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object]]
        ]
        | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        """Delegate one structured call with a bounded async tool bridge."""
        synchronous_executor = None
        if tool_executor is not None:
            event_loop = asyncio.get_running_loop()

            def synchronous_executor(
                tool_name: str,
                arguments: dict[str, Any],
            ) -> Mapping[str, object]:
                # The shared ProductAgent invokes tools from its worker thread;
                # evidence services remain owned by the UI event loop.
                future = asyncio.run_coroutine_threadsafe(
                    tool_executor(tool_name, arguments),
                    event_loop,
                )
                try:
                    return future.result(timeout=_TOOL_TIMEOUT_SEC)
                except FutureTimeoutError as exc:
                    future.cancel()
                    raise RuntimeError(
                        "Controlled evidence retrieval timed out."
                    ) from exc

        return await self._product_agent.ask_llm_structured(
            prompt,
            response_format=response_format,
            tools=tools,
            tool_executor=synchronous_executor,
            max_tool_rounds=max_tool_rounds,
        )


def create_product_agent_context_runtime(*, model: str) -> ProductAgentContextRuntime:
    """Create the read-only ProductAgent composition used by the PA UI."""
    return _SharedProductAgentContextRuntime(model=model)
