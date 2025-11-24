"""Shared base class for all SPADE agents that communicate via an OpenAI-powered LLM."""

from __future__ import annotations
import os, json, time, asyncio, logging
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Iterable

from openai import OpenAI  # REST client for GPT models.
from spade.agent import Agent  # SPADE base class providing lifecycle hooks.

from function_analyzer import FunctionAnalyzer  # Introspects agent methods for tool schemas.
from prompts import PROMPT_MAS_AGENT, BASE_INSTRUCTIONS, ROLE_BLOCKS
from dotenv import load_dotenv
load_dotenv()

_client = OpenAI()  # Single shared client so we reuse HTTP sessions and rate-limit buckets.


class LlmAgent(Agent):
    """Mixin-style agent that wires logging, tool analysis, and LLM access into SPADE agents."""

    # Shared tool catalogue cache so every agent has access to the same tool metadata.
    _TOOLS_CATALOG: List[Dict[str, Any]] | None = None
    _TOOLS_BY_FUNC: Dict[str, Dict[str, Any]] | None = None

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: Optional[str] = None,
        agent_role: str = "",
        model: str = "gpt-4o",
        non_function_model: str = "gpt-4o-mini",
        instructions: Optional[str] = None,
        function_names: Optional[List[str]] = None,
    ) -> None:
        """Capture metadata that every LLM-aware agent needs (identity, prompts, and available tools)."""
        super().__init__(jid, password)

        self.agent_name = name or self.jid.localpart
        self.agent_role = (agent_role or "").lower()
        self.model = model  # Main LLM that supports tool calling.
        self.non_function_model = non_function_model  # Cheaper model for plain generations.

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
        self.executables: Dict[str, Callable[..., Any]] = {}
        if function_names:
            # Only register methods that physically exist on the subclass to prevent runtime failures.
            for fn in function_names:
                if hasattr(self, fn) and callable(getattr(self, fn)):
                    self.executables[fn] = getattr(self, fn)

        # Cached JSON schema definitions consumed by OpenAI's tool-calling interface.
        self.function_info: List[Dict[str, Any]] = []
        self._rebuild_tool_schemas()

        # System prompt includes base role instructions plus any user overrides.
        self.instructions = self._build_agent_instructions(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            overrides=instructions,
        )

    @staticmethod
    def _build_agent_instructions(
        *, agent_name: str, agent_role: str, overrides: Optional[str] = None
    ) -> str:
        """Compose the system prompt: base instructions + role-specific block + optional overrides."""
        base = PROMPT_MAS_AGENT + "\n" + BASE_INSTRUCTIONS  # Shared prologue for every agent.
        role_block = ROLE_BLOCKS.get(agent_role.lower(), "")  # Role-specific reminders/live data.
        tail = f"\nCustom overrides:\n{overrides}\n" if overrides else ""  # User-provided tweaks.
        prompt = base.replace("{agent_name}", agent_name) + ("\n" + role_block if role_block else "") + tail
        return prompt.strip()

    # ------------------------------------------------------------------ #
    # Tool catalogue helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _load_shared_tools_catalogue(cls) -> None:
        """Lazy-load tools.json exactly once so every agent sees the same snapshot."""
        if LlmAgent._TOOLS_CATALOG is not None:
            return

        path = Path("cais_spade_llm/initialization/tools.json")
        try:
            LlmAgent._TOOLS_CATALOG = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                "tools.json missing – run FunctionAnalyzer.build_tools_catalogue() first."
            ) from exc

        LlmAgent._TOOLS_BY_FUNC = {
            row["function"]: row
            for row in (LlmAgent._TOOLS_CATALOG or [])
            if "function" in row
        }

    @property
    def tools_catalog(self) -> List[Dict[str, Any]]:
        self.__class__._load_shared_tools_catalogue()
        return LlmAgent._TOOLS_CATALOG or []

    @property
    def tools_by_func(self) -> Dict[str, Dict[str, Any]]:
        self.__class__._load_shared_tools_catalogue()
        return LlmAgent._TOOLS_BY_FUNC or {}

    def _rebuild_tool_schemas(self) -> None:
        """Convert bound executable methods into JSON schema definitions for tool calling."""
        tools: List[Dict[str, Any]] = []
        for fn_name, fn in self.executables.items():
            try:
                analyzed = self.function_analyzer.analyze_function(fn)
                name = analyzed.get("name", fn_name)
                description = analyzed.get("description", f"Tool: {name}")
                parameters = analyzed.get("parameters", {"type": "object", "properties": {}, "required": []})
                # Follow the OpenAI tool calling format; SPADE higher layers only need these descriptions.
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    }
                })
            except Exception as e:
                self.logger.exception(f"Function analysis failed for '{fn_name}': {e}")
        self.function_info = tools

    async def setup(self):
        """Log the configured LLM and functions so operators know the agent is ready."""
        self.logger.info(
            f"[ready] {self.jid} (LLM={self.model}, tools={[t['function']['name'] for t in self.function_info]})"
        )

    # ------------------------------------------------------------------ #
    # Resource capability helpers (shared by product/controller agents)
    # ------------------------------------------------------------------ #
    def _capability_catalogue(
        self, resource_agents: Optional[Iterable[Any]] = None
    ) -> dict[str, set[str]]:
        """
        Build a map of capability -> set of values across resource agents.
        Each resource agent may expose `.static_capabilities` as dict[str, Iterable].
        """
        resources = list(resource_agents) if resource_agents is not None else list(
            getattr(self, "resource_agents", []) or []
        )

        caps: dict[str, set[str]] = {}
        for ra in resources:
            for key, val in getattr(ra, "static_capabilities", {}).items():
                iterable = val if isinstance(val, (list, tuple, set)) else [val]
                caps.setdefault(str(key).lower(), set()).update(map(str, iterable))
        return caps

    def _static_caps_overview(
        self, resource_agents: Optional[Iterable[Any]] = None
    ) -> str:
        """Human-friendly string summarizing capabilities for prompt grounding."""
        caps = self._capability_catalogue(resource_agents)
        if not caps:
            return "(no static capabilities registered)"
        return " | ".join(
            f"{k}: {', '.join(sorted(v))}" for k, v in caps.items()
        )

    async def ask_llm(
        self,
        prompt: str | Dict[str, Any],
        *,
        with_functions: bool = True,
        force_tool: bool = False,
        temperature: float = 0.0,
    ) -> Dict[str, Any] | str:
        """Call the configured LLM, optionally exposing this agent's tool catalogue to force tool selection."""
        def _call():
            """Blocking helper executed in a thread so SPADE behaviours stay async friendly."""
            msgs: List[Dict[str, Any]] = []
            if self.instructions:
                msgs.append({"role": "system", "content": self.instructions})
            # Allow callers to send either raw text or structured dicts (the latter is auto-serialized).
            user_content = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
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
                            temperature=temperature,
                        )
                    else:
                        r = _client.chat.completions.create(
                            model=self.non_function_model,
                            messages=msgs,
                            temperature=temperature,
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
