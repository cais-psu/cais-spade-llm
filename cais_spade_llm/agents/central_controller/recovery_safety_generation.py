"""Associate generated recovery with the existing CCA-owned safety requirements."""

from __future__ import annotations

from typing import Any

from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
    predefined_required, predefined_scope,
)


async def generate_recovery_safety_bundle(
    controller_agent: Any, payload: dict[str, Any],
) -> dict[str, Any]:
    """Reuse fixed requirements without asking an LLM to select or rewrite APs.

    Physical relevance and admission are established later from resource-owned
    primitive evidence. This association neither resets monitors nor grants motion.
    """
    if not predefined_required(controller_agent):
        raise ValueError("Recovery requires context-free predefined safety; recompile required")
    return predefined_scope(controller_agent, str(payload.get("recovery_safety_scope_id") or ""))
