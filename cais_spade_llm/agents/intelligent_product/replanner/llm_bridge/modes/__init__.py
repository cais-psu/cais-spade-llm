"""Mode executors for active v4 bridge reasoning flows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.multi_turn_v2 import (
    build_multi_turn_session_seed,
    execute_multi_turn_bridge,
    transition_multi_turn_phase,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.single_shot import (
    build_single_shot_prompt_artifacts,
    execute_single_shot_bridge,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.hybrid_des_v1 import (
    build_hybrid_session_seed as _build_hybrid_session_seed,
    execute_hybrid_des_bridge as _execute_hybrid_des_bridge,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes.procedural_des_v1 import (
    build_procedural_session_seed as _build_procedural_session_seed,
    execute_procedural_des_bridge as _execute_procedural_des_bridge,
)


def build_hybrid_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    return _build_hybrid_session_seed(deepcopy(prepared_bridge_request or {}))


def build_procedural_session_seed(
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any]:
    return _build_procedural_session_seed(deepcopy(prepared_bridge_request or {}))


async def execute_hybrid_des_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return await _execute_hybrid_des_bridge(
        planner,
        prepared_bridge_request or {},
        session_state=session_state,
    )


async def execute_procedural_des_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
    *,
    session_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return await _execute_procedural_des_bridge(
        planner,
        prepared_bridge_request or {},
        session_state=session_state,
    )


__all__ = [
    "build_hybrid_session_seed",
    "build_multi_turn_session_seed",
    "build_procedural_session_seed",
    "build_single_shot_prompt_artifacts",
    "execute_hybrid_des_bridge",
    "execute_multi_turn_bridge",
    "execute_procedural_des_bridge",
    "execute_single_shot_bridge",
    "transition_multi_turn_phase",
]
