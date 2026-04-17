"""Single-shot executor for the active v4 bridge."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.archive.prompts.single_shot import (
    build_single_shot_prompt_input,
    render_single_shot_prompt,
)


def build_single_shot_prompt_artifacts(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    reasoning_mode = planner._normalize_bridge_reasoning_mode(
        dict(prepared_bridge_request.get("bridge_session") or {}).get("reasoning_mode")
        or planner._resolve_bridge_reasoning_mode()
    )
    prompt_input = build_single_shot_prompt_input(
        reasoning_mode=reasoning_mode,
        llm_input=deepcopy(prepared_bridge_request.get("llm_input") or {}),
    )
    prompt_text = render_single_shot_prompt(prompt_input)
    return prompt_input, prompt_text


async def execute_single_shot_bridge(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(prepared_bridge_request.get("single_shot_prompt_input"), dict) or not str(
        prepared_bridge_request.get("single_shot_prompt_text") or ""
    ).strip():
        prompt_input, prompt_text = build_single_shot_prompt_artifacts(
            planner,
            prepared_bridge_request,
        )
        prepared_bridge_request["single_shot_prompt_input"] = deepcopy(prompt_input)
        prepared_bridge_request["single_shot_prompt_text"] = str(prompt_text or "")

    product_agent = getattr(planner, "product_agent", None)
    ask_llm = getattr(product_agent, "ask_llm", None)
    if not callable(ask_llm):
        raise RuntimeError("product_agent.ask_llm is required for single-shot bridge execution")

    bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
    reasoning_mode = str(bridge_session.get("reasoning_mode") or "single_shot").strip().lower()
    prompt_text = str(prepared_bridge_request.get("single_shot_prompt_text") or "")
    raw_response = await ask_llm(
        prompt=prompt_text,
        with_functions=False,
        temperature=0.0,
    )

    bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
    bridge_debug["status"] = "llm_output_recorded"
    bridge_debug["message"] = (
        "Single-shot prompt was sent to the LLM and the raw response was captured. "
        "Proposal parsing and validation are not implemented yet."
    )
    bridge_debug["single_shot_turn"] = {
        "reasoning_mode": reasoning_mode,
        "status": "llm_output_recorded",
        "prompt_input": deepcopy(
            prepared_bridge_request.get("single_shot_prompt_input") or {}
        ),
        "prompt_text": prompt_text,
        "raw_response": str(raw_response or ""),
    }
    prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
    if hasattr(planner, "_set_last_bridge_debug"):
        planner._set_last_bridge_debug(bridge_debug)
    return None
