# agents/shared_information/llm_agent.py
from __future__ import annotations
import os, json, time, asyncio, logging
from typing import Any, Callable, Optional, Dict, List

from openai import OpenAI  # v1.x SDK
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from function_analyzer import FunctionAnalyzer
from prompts import PROMPT_MAS_AGENT, BASE_INSTRUCTIONS, ROLE_BLOCKS

# Uses OPENAI_API_KEY from environment by default
_client = OpenAI()


class LlmAgent(Agent):
    """
    Parent for ProductAgent / ResourceAgent.
    - SPADE-native (async behaviours, XMPP messaging)
    - LLM helper (non-blocking)
    - Optional function/tool registry for model tool-calls or explicit RPC
    """

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: Optional[str] = None,
        agent_role: str = "",
        model: str = "gpt-4o",
        non_function_model: str = "gpt-4o",
        annotation: Optional[str] = None,
        instructions: Optional[str] = None,        # per-agent overrides from JSON
        function_names: Optional[List[str]] = None,
    ) -> None:
        # IMPORTANT: only pass jid/password to SPADE
        super().__init__(jid, password)

        # identity / metadata
        self.agent_name = name or self.jid.localpart
        self.agent_role = (agent_role or "").lower()
        self.annotation = annotation or ""
        self.model = model
        self.non_function_model = non_function_model

        # logging
        self.logger = logging.getLogger(f"agent:{self.agent_name}")
        if not self.logger.handlers:
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
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # functions / tools registry
        self.function_analyzer = FunctionAnalyzer()
        self.executables: Dict[str, Callable[..., Any]] = {}
        if function_names:
            for fn in function_names:
                if hasattr(self, fn) and callable(getattr(self, fn)):
                    self.executables[fn] = getattr(self, fn)

        # derived function schemas (legacy schema from analyzer) -> convert to v1 tools on call
        self.function_info = (
            [self.function_analyzer.analyze_function(f) for f in self.executables.values()]
            if self.executables
            else []
        )

        # system prompt
        self.instructions = self._build_agent_instructions(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            overrides=instructions,
        )

    # ---------- prompt builder ----------
    @staticmethod
    def _build_agent_instructions(
        *, agent_name: str, agent_role: str, overrides: Optional[str] = None
    ) -> str:
        base = PROMPT_MAS_AGENT + "\n" + BASE_INSTRUCTIONS
        role_block = ROLE_BLOCKS.get(agent_role.lower(), "")
        tail = f"\nCustom overrides:\n{overrides}\n" if overrides else ""
        prompt = base.replace("{agent_name}", agent_name) + ("\n" + role_block if role_block else "") + tail
        return prompt.strip()

    # ---------- SPADE lifecycle ----------
    async def setup(self):
        # 1) LLM query inbox: {type:"llm.query", body: {"prompt": "...", "with_functions": true/false}}
        t_llm = Template()
        t_llm.set_metadata("type", "llm.query")
        self.add_behaviour(self._LlmInbox(), t_llm)

        # 2) Tool RPC inbox: {type:"tool.call", body: {"name": "...", "args": {...}}}
        t_tool = Template()
        t_tool.set_metadata("type", "tool.call")
        self.add_behaviour(self._ToolInbox(), t_tool)

        self.logger.info(f"[ready] {self.jid} (LLM={self.model}, tools={list(self.executables)})")

    # ---------- Public helpers ----------
    async def ask_llm(
        self,
        prompt: str,
        *,
        with_functions: bool = True,
        temperature: float = 0.0,
    ) -> str:
        """Non-blocking LLM call; safe inside behaviours. OpenAI SDK v1.x."""
        def _call():
            msgs = []
            if self.instructions:
                msgs.append({"role": "system", "content": self.instructions})
            msgs.append({"role": "user", "content": prompt})

            # Convert legacy function schema → v1 tools only if requested
            tools = (
                [{"type": "function", "function": f} for f in self.function_info]
                if (with_functions and self.function_info)
                else None
            )

            back = 1.0
            for _ in range(5):
                try:
                    if tools:
                        r = _client.chat.completions.create(
                            model=self.model,
                            messages=msgs,
                            tools=tools,
                            tool_choice="auto",
                            temperature=temperature,
                        )
                    else:
                        r = _client.chat.completions.create(
                            model=self.non_function_model,
                            messages=msgs,
                            temperature=temperature,
                        )

                    msg = r.choices[0].message

                    # If the model called a tool, surface the first call as JSON
                    if getattr(msg, "tool_calls", None):
                        call = msg.tool_calls[0]
                        # call.function has .name and .arguments (JSON string)
                        return json.dumps(
                            {
                                "tool_call": {
                                    "id": call.id,
                                    "type": call.type,
                                    "function": {
                                        "name": call.function.name,
                                        "arguments": call.function.arguments,
                                    },
                                }
                            }
                        )

                    return (msg.content or "").strip()
                except Exception:
                    time.sleep(back)
                    back = min(back * 2, 8.0)
            raise RuntimeError("LLM call failed after retries.")

        return await asyncio.to_thread(_call)

    async def send_text(self, to_jid: str, text: str, *, mtype: str = "llm.reply") -> None:
        msg = Message(to=to_jid)
        msg.set_metadata("type", mtype)
        msg.body = text
        await self.send(msg)

    # ---------- Behaviours ----------
    class _LlmInbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.5)
            if not msg or msg.metadata.get("type") != "llm.query":
                return
            try:
                data = json.loads(msg.body or "{}")
                prompt = data.get("prompt", "")
                with_functions = bool(data.get("with_functions", True))
                ans = await self.agent.ask_llm(prompt, with_functions=with_functions)
                reply = Message(to=str(msg.sender))
                reply.set_metadata("type", "llm.reply")
                reply.body = json.dumps({"answer": ans, "model": self.agent.model})
                await self.send(reply)
            except Exception as e:
                err = Message(to=str(msg.sender))
                err.set_metadata("type", "error")
                err.body = json.dumps({"error": str(e)})
                await self.send(err)

    class _ToolInbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.5)
            if not msg or msg.metadata.get("type") != "tool.call":
                return
            try:
                data = json.loads(msg.body or "{}")
                fname, args = data.get("name"), data.get("args", {})
                fn = self.agent.executables.get(fname)
                if not fn:
                    raise ValueError(f"Unknown tool: {fname}")
                # sync or async tool
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = await asyncio.to_thread(fn, **args)
                reply = Message(to=str(msg.sender))
                reply.set_metadata("type", "tool.reply")
                reply.body = json.dumps({"success": True, "result": result})
                await self.send(reply)
            except Exception as e:
                err = Message(to=str(msg.sender))
                err.set_metadata("type", "tool.reply")
                err.body = json.dumps({"success": False, "error": str(e)})
                await self.send(err)
