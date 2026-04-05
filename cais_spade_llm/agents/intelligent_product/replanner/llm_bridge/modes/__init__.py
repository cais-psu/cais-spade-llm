"""Mode executors for active v4 bridge reasoning flows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn import (
    build_multi_turn_session_seed as _build_multi_turn_session_seed_legacy,
    execute_multi_turn_bridge as _execute_multi_turn_bridge_legacy,
    transition_multi_turn_phase,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_v2 import (
    build_multi_turn_session_seed as _build_multi_turn_session_seed_v2,
    execute_multi_turn_bridge as _execute_multi_turn_bridge_v2,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.single_shot import (
    build_single_shot_prompt_artifacts,
    execute_single_shot_bridge,
)


def _resolve_multi_turn_engine(
    prepared_bridge_request: dict[str, Any] | None,
) -> str:
    payload = dict(prepared_bridge_request or {})
    bridge_session = dict(payload.get("bridge_session") or {})
    requested = str(
        bridge_session.get("multi_turn_engine")
        or dict(payload.get("multi_turn_session_seed") or {}).get("multi_turn_engine")
        or "v2"
    ).strip().lower()
    if requested in ("v2", "legacy"):
        return requested
    return "v2"


def build_multi_turn_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(prepared_bridge_request or {})
    engine = _resolve_multi_turn_engine(payload)
    if engine == "v2":
        return _build_multi_turn_session_seed_v2(payload)
    seed = _build_multi_turn_session_seed_legacy(payload)
    seed["multi_turn_engine"] = engine
    return seed


async def execute_multi_turn_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload = prepared_bridge_request or {}
    engine = _resolve_multi_turn_engine(payload)
    if engine == "v2":
        return await _execute_multi_turn_bridge_v2(
            planner, payload, session_state=session_state,
        )
    return await _execute_multi_turn_bridge_legacy(planner, payload)

__all__ = [
    "build_multi_turn_session_seed",
    "build_single_shot_prompt_artifacts",
    "execute_multi_turn_bridge",
    "execute_single_shot_bridge",
    "transition_multi_turn_phase",
]
