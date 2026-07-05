"""Shared base class for all SPADE agents that communicate via an OpenAI-powered LLM."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI  # REST client for GPT models.
from spade.agent import Agent  # SPADE base class providing lifecycle hooks.

from cais_spade_llm.function_analyzer import (
    FunctionAnalyzer,  # Introspects agent methods for tool schemas.
)
from cais_spade_llm.prompts import BASE_INSTRUCTIONS, PROMPT_MAS_AGENT, ROLE_BLOCKS

load_dotenv()

_client = OpenAI()  # Single shared client so we reuse HTTP sessions and rate-limit buckets.


def _parse_structured_json_text(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        for start_idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, end_idx = decoder.raw_decode(text[start_idx:])
            except json.JSONDecodeError:
                continue
            trailing = text[start_idx + end_idx :].strip()
            if trailing:
                logging.getLogger(__name__).warning(
                    "Ignoring trailing text after structured JSON payload"
                )
            return parsed
        raise exc


def _default_model_name(*env_names: str, fallback: str) -> str:
    """Resolve a model slug from env vars, then fall back to a safe default."""
    for env_name in env_names:
        token = str(os.environ.get(env_name) or "").strip()
        if token:
            return token
    return str(fallback or "").strip()


_ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _default_reasoning_effort(*env_names: str, fallback: str) -> str:
    """Resolve a supported reasoning effort from env vars, then fall back."""
    token = _default_model_name(*env_names, fallback=fallback).strip().lower()
    if token in _ALLOWED_REASONING_EFFORTS:
        return token
    return str(fallback or "").strip().lower()


def _normalize_reasoning_effort_for_model(model_name: str, effort: str) -> str:
    """Normalize repo defaults to the target model family's supported effort set.

    GPT-5.4 family models currently reject ``minimal`` and accept ``none`` as the
    lowest-effort option. Keep this mapping narrow so we do not silently rewrite
    unrelated model families.
    """
    normalized_model = str(model_name or "").strip().lower()
    normalized_effort = str(effort or "").strip().lower()
    if normalized_model.startswith("gpt-5.4") and normalized_effort == "minimal":
        return "none"
    return normalized_effort


class LlmAgent(Agent):
    """Mixin-style agent that wires logging, tool analysis, and LLM access into SPADE agents."""

    # Shared tool catalogue cache so every agent has access to the same tool metadata.
    _TOOLS_CATALOG: list[dict[str, Any]] | None = None
    _TOOLS_BY_FUNC: dict[str, dict[str, Any]] | None = None
    _TOOLS_CATALOG_PATH: Path = Path("cais_spade_llm/initialization/tools.json")

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str | None = None,
        agent_role: str = "",
        model: str | None = None,
        non_function_model: str | None = None,
        instructions: str | None = None,
        function_names: list[str] | None = None,
    ) -> None:
        """Capture metadata that every LLM-aware agent needs (identity, prompts, and available tools)."""
        super().__init__(jid, password)

        self.agent_name = name or self.jid.localpart
        self.agent_role = (agent_role or "").lower()
        selected_model = str(model or "").strip() or _default_model_name(
            "CAIS_SPADE_LLM_MODEL",
            "OPENAI_MODEL",
            "CASE3_RECOVERY_MODEL",
            fallback="gpt-5.4",
        )
        selected_non_function_model = str(non_function_model or "").strip() or _default_model_name(
            "CAIS_SPADE_NON_FUNCTION_MODEL",
            "OPENAI_NON_FUNCTION_MODEL",
            "CAIS_SPADE_LLM_MODEL",
            "OPENAI_MODEL",
            "CASE3_RECOVERY_MODEL",
            fallback=selected_model,
        )
        selected_reasoning_effort = _default_reasoning_effort(
            "CAIS_SPADE_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            fallback="medium",
        )
        selected_non_function_reasoning_effort = _default_reasoning_effort(
            "CAIS_SPADE_NON_FUNCTION_REASONING_EFFORT",
            "OPENAI_NON_FUNCTION_REASONING_EFFORT",
            "CAIS_SPADE_REASONING_EFFORT",
            "OPENAI_REASONING_EFFORT",
            fallback=selected_reasoning_effort,
        )
        self.model = selected_model  # Main LLM that supports tool calling.
        self.non_function_model = (
            selected_non_function_model  # Cheaper model for plain generations.
        )
        self.reasoning_effort = _normalize_reasoning_effort_for_model(
            selected_model,
            selected_reasoning_effort,
        )
        self.non_function_reasoning_effort = _normalize_reasoning_effort_for_model(
            selected_non_function_model,
            selected_non_function_reasoning_effort,
        )

        # logging
        self.logger = logging.getLogger(f"agent:{self.agent_name}")
        if not self.logger.handlers:
            # Defer handler creation until the first instance with this name to avoid duplicate logs.
            self.logger.setLevel(logging.INFO)
            os.makedirs("cais_spade_llm/log", exist_ok=True)
            fh = logging.FileHandler(
                f"cais_spade_llm/log/{self.agent_name}_actions.log",
                mode="a",
                encoding="utf-8",
            )
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            # Mirror logs to disk (for audits) and stdout (for dev convenience).
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # tools registry
        self.function_analyzer = FunctionAnalyzer()
        self.executables: dict[str, Callable[..., Any]] = {}
        if function_names:
            # Only register methods that physically exist on the subclass to prevent runtime failures.
            for fn in function_names:
                if hasattr(self, fn) and callable(getattr(self, fn)):
                    self.executables[fn] = getattr(self, fn)

        # Cached JSON schema definitions consumed by OpenAI's tool-calling interface.
        self.function_info: list[dict[str, Any]] = []
        self._rebuild_tool_schemas()

        # System prompt includes base role instructions plus any user overrides.
        self.instructions = self._build_agent_instructions(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            overrides=instructions,
        )

    @staticmethod
    def _build_agent_instructions(
        *, agent_name: str, agent_role: str, overrides: str | None = None
    ) -> str:
        """Compose the system prompt: base instructions + role-specific block + optional overrides."""
        base = PROMPT_MAS_AGENT + "\n" + BASE_INSTRUCTIONS  # Shared prologue for every agent.
        role_block = ROLE_BLOCKS.get(agent_role.lower(), "")  # Role-specific reminders/live data.
        tail = f"\nCustom overrides:\n{overrides}\n" if overrides else ""  # User-provided tweaks.
        prompt = (
            base.replace("{agent_name}", agent_name)
            + ("\n" + role_block if role_block else "")
            + tail
        )
        return prompt.strip()

    # ------------------------------------------------------------------ #
    # Tool catalogue helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def configure_shared_tools_catalogue(cls, path: str | Path | None = None) -> str:
        """Point all agents at a specific tools catalogue and clear any cached snapshot."""
        resolved = (
            Path(path) if path is not None else Path("cais_spade_llm/initialization/tools.json")
        )
        LlmAgent._TOOLS_CATALOG_PATH = resolved.resolve()
        LlmAgent._TOOLS_CATALOG = None
        LlmAgent._TOOLS_BY_FUNC = None
        return str(LlmAgent._TOOLS_CATALOG_PATH)

    @classmethod
    def _load_shared_tools_catalogue(cls) -> None:
        """Lazy-load tools.json exactly once so every agent sees the same snapshot."""
        if LlmAgent._TOOLS_CATALOG is not None:
            return

        path = Path(LlmAgent._TOOLS_CATALOG_PATH)
        try:
            LlmAgent._TOOLS_CATALOG = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"tools.json missing at {path} – run FunctionAnalyzer.build_tools_catalogue() first."
            ) from exc

        LlmAgent._TOOLS_BY_FUNC = {
            row["function"]: row for row in (LlmAgent._TOOLS_CATALOG or []) if "function" in row
        }

    @property
    def tools_catalog(self) -> list[dict[str, Any]]:
        self.__class__._load_shared_tools_catalogue()
        return LlmAgent._TOOLS_CATALOG or []

    @property
    def tools_by_func(self) -> dict[str, dict[str, Any]]:
        self.__class__._load_shared_tools_catalogue()
        return LlmAgent._TOOLS_BY_FUNC or {}

    def _rebuild_tool_schemas(self) -> None:
        """Convert bound executable methods into JSON schema definitions for tool calling."""
        tools: list[dict[str, Any]] = []
        for fn_name, fn in self.executables.items():
            try:
                analyzed = self.function_analyzer.analyze_function(fn)
                name = analyzed.get("name", fn_name)
                description = analyzed.get("description", f"Tool: {name}")
                parameters = analyzed.get(
                    "parameters", {"type": "object", "properties": {}, "required": []}
                )
                # Follow the OpenAI tool calling format; SPADE higher layers only need these descriptions.
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": parameters,
                        },
                    }
                )
            except Exception as e:
                self.logger.exception(f"Function analysis failed for '{fn_name}': {e}")
        self.function_info = tools

    async def setup(self):
        """Log the configured LLM and functions so operators know the agent is ready."""
        self.logger.info(
            "[ready] %s (LLM=%s effort=%s, non_function=%s effort=%s, tools=%s)",
            self.jid,
            self.model,
            self.reasoning_effort,
            self.non_function_model,
            self.non_function_reasoning_effort,
            [t["function"]["name"] for t in self.function_info],
        )

    # ------------------------------------------------------------------ #
    # Resource capability helpers (shared by product/controller agents)
    # ------------------------------------------------------------------ #
    def _capability_catalogue(
        self, resource_agents: Iterable[Any] | None = None
    ) -> dict[str, set[str]]:
        """
        Build a map of capability -> set of values across resource agents.
        Each resource agent may expose `.static_capabilities` as dict[str, Iterable].
        """
        resources = (
            list(resource_agents)
            if resource_agents is not None
            else list(getattr(self, "resource_agents", []) or [])
        )

        caps: dict[str, set[str]] = {}
        for ra in resources:
            for key, val in getattr(ra, "static_capabilities", {}).items():
                iterable = val if isinstance(val, (list, tuple, set)) else [val]
                caps.setdefault(str(key).lower(), set()).update(map(str, iterable))
        return caps

    def _static_caps_overview(self, resource_agents: Iterable[Any] | None = None) -> str:
        """Human-friendly string summarizing capabilities for prompt grounding."""
        caps = self._capability_catalogue(resource_agents)
        if not caps:
            return "(no static capabilities registered)"
        return " | ".join(f"{k}: {', '.join(sorted(v))}" for k, v in caps.items())

    async def ask_llm(
        self,
        prompt: str | dict[str, Any],
        *,
        with_functions: bool = True,
        force_tool: bool = False,
        temperature: float = 0.0,
    ) -> dict[str, Any] | str:
        """Call the configured LLM, optionally exposing this agent's tool catalogue to force tool selection."""

        def _call():
            """Blocking helper executed in a thread so SPADE behaviours stay async friendly."""
            msgs: list[dict[str, Any]] = []
            if self.instructions:
                msgs.append({"role": "system", "content": self.instructions})
            # Allow callers to send either raw text or structured dicts (the latter is auto-serialized).
            user_content = (
                prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
            )
            msgs.append({"role": "user", "content": user_content})

            # Only expose tool schemas when requested; some flows prefer a pure text response for speed.
            tools = self.function_info if (with_functions and self.function_info) else None

            back = 1.0  # Initial retry delay (seconds) for exponential backoff.
            for _ in range(5):
                try:
                    # When tools are present we hit the multi-modal function model, otherwise a plain chat model.
                    if tools:
                        r = _client.chat.completions.create(
                            model=self.model,
                            messages=msgs,
                            tools=tools,
                            tool_choice=("required" if force_tool else "auto"),
                            reasoning_effort=self.reasoning_effort,
                            # temperature=temperature,
                        )
                    else:
                        r = _client.chat.completions.create(
                            model=self.non_function_model,
                            messages=msgs,
                            reasoning_effort=self.non_function_reasoning_effort,
                            # temperature=temperature,
                        )

                    choice = r.choices[0].message
                    # Tool call: return the function invocation payload so caller can dispatch it.
                    if getattr(choice, "tool_calls", None):
                        call = choice.tool_calls[0]
                        return {
                            "function_call": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            }
                        }
                    # Otherwise it's a plain completion.
                    return (choice.content or "").strip()
                except Exception as e:
                    # Simple exponential backoff to smooth out transient OpenAI errors or rate limits.
                    time.sleep(back)
                    back = min(back * 2, 8.0)
                    last_err = e
            raise RuntimeError(f"LLM call failed after retries: {type(last_err).__name__}")

        # Run the blocking OpenAI call off the event loop so SPADE behaviours stay responsive.
        return await asyncio.to_thread(_call)

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
    ) -> dict[str, Any]:
        """Call LLM with structured output + optional tool use (v3 bridge).

        Parameters
        ----------
        prompt:
            The user-role prompt to send.
        response_format:
            OpenAI ``response_format`` dict with ``type: "json_schema"``
            for constrained decoding.
        tools:
            Optional list of tool definitions the LLM may call mid-turn.
        tool_executor:
            Callback ``(tool_name, arguments_dict) -> result_dict`` invoked
            when the LLM emits a tool call.
        max_tool_rounds:
            Maximum number of tool-call rounds before raising.

        Returns
        -------
        dict:
            The parsed structured response from the LLM.
        """

        def _call() -> dict[str, Any]:
            msgs: list[dict[str, Any]] = []
            if self.instructions:
                msgs.append({"role": "system", "content": self.instructions})
            msgs.append({"role": "user", "content": prompt})

            for _ in range(max_tool_rounds + 1):
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": msgs,
                    "reasoning_effort": self.reasoning_effort,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": response_format,
                    },
                }
                if tools:
                    kwargs["tools"] = tools

                back = 1.0
                last_err: Exception | None = None
                for _ in range(5):
                    try:
                        r = _client.chat.completions.create(**kwargs)
                        break
                    except Exception as e:
                        time.sleep(back)
                        back = min(back * 2, 8.0)
                        last_err = e
                else:
                    raise RuntimeError(f"LLM call failed after retries: {type(last_err).__name__}")

                choice = r.choices[0].message

                # Handle tool calls if the LLM wants to use a tool.
                if getattr(choice, "tool_calls", None) and tool_executor:
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": choice.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in choice.tool_calls
                            ],
                        }
                    )
                    for tc in choice.tool_calls:
                        result = tool_executor(
                            tc.function.name,
                            json.loads(tc.function.arguments),
                        )
                        msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                    continue

                # No tool calls — return the structured response.
                return _parse_structured_json_text(choice.content or "{}")

            raise RuntimeError("Exceeded max tool rounds")

        return await asyncio.to_thread(_call)
