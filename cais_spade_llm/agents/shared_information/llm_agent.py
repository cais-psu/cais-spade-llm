# agents/shared_information/llm_agent.py
from __future__ import annotations
import os, json, time, asyncio, logging
from typing import Any, Callable
import openai

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from function_analyzer import FunctionAnalyzer
from prompts import PROMPT_MAS_AGENT, BASE_INSTRUCTIONS

# --- OpenAI key (dev) ---
openai.api_key = os.getenv("OPENAI_API_KEY") or ""

class LlmAgent(Agent):
    """
    Parent for ProductAgent / ResourceAgent.
    - SPADE-native (async behaviours, XMPP messaging)
    - LLM helper (non-blocking)
    - Optional function/tool registry for model function-calls or explicit RPC
    """

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str | None = None,
        model: str = "gpt-4o",
        non_function_model: str = "gpt-4o",
        annotation: str | None = None,
        instructions: str | None = None,
        function_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(jid, password, **kwargs)

        # identity / metadata
        self.agent_name = name or self.jid.localpart
        self.annotation = annotation or ""
        self.model = model
        self.non_function_model = non_function_model

        # logging
        self.logger = logging.getLogger(f"agent:{self.agent_name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(f"manumas/log/{self.agent_name}_actions.log", mode="a")
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            ch = logging.StreamHandler(); ch.setFormatter(fmt)
            self.logger.addHandler(fh); self.logger.addHandler(ch)

        # functions / tools
        self.function_analyzer = FunctionAnalyzer()
        self.executables: dict[str, Callable[..., Any]] = {}
        if function_names:
            for fn in function_names:
                if hasattr(self, fn) and callable(getattr(self, fn)):
                    self.executables[fn] = getattr(self, fn)

        # system prompt
        base = PROMPT_MAS_AGENT + BASE_INSTRUCTIONS
        self.instructions = (base + (instructions or "")) if self.executables else (PROMPT_MAS_AGENT)

        # derived OpenAI function schemas (only if we have tools)
        self.function_info = [self.function_analyzer.analyze_function(f) for f in self.executables.values()] if self.executables else []

    # ---------- SPADE lifecycle ----------
    async def setup(self):
        # 1) LLM query inbox: {type:"llm.query", body: {"prompt": "...", "with_functions": true/false}}
        t_llm = Template(); t_llm.set_metadata("type", "llm.query")
        self.add_behaviour(self._LlmInbox(), t_llm)

        # 2) Tool RPC inbox: {type:"tool.call", body: {"name": "...", "args": {...}}}
        t_tool = Template(); t_tool.set_metadata("type", "tool.call")
        self.add_behaviour(self._ToolInbox(), t_tool)

        self.logger.info(f"[ready] {self.jid} (LLM={self.model}, tools={list(self.executables)})")

    # ---------- Public helpers ----------
    async def ask_llm(self, prompt: str, *, with_functions: bool = True, temperature: float = 0.0) -> str:
        """Non-blocking LLM call; safe inside behaviours."""
        def _call():
            msgs = []
            if self.instructions:
                msgs.append({"role": "system", "content": self.instructions})
            msgs.append({"role": "user", "content": prompt})

            # basic retry
            back = 1.0
            for _ in range(5):
                try:
                    if with_functions and self.function_info:
                        r = openai.ChatCompletion.create(
                            model=self.model,
                            messages=msgs,
                            functions=self.function_info,
                            function_call="auto",
                            temperature=temperature,
                        )
                    else:
                        r = openai.ChatCompletion.create(
                            model=self.non_function_model, messages=msgs, temperature=temperature
                        )
                    msg = r["choices"][0]["message"]

                    # If model decided to call a function, surface that as JSON text
                    if msg.get("function_call"):
                        return json.dumps({"function_call": msg["function_call"]})
                    return msg.get("content", "").strip()
                except Exception:
                    time.sleep(back); back = min(back * 2, 8.0)
            raise RuntimeError("LLM call failed after retries.")
        return await asyncio.to_thread(_call)

    async def send_text(self, to_jid: str, text: str, *, mtype: str = "llm.reply") -> None:
        msg = Message(to=to_jid); msg.set_metadata("type", mtype); msg.body = text
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
                err = Message(to=str(msg.sender)); err.set_metadata("type", "error")
                err.body = json.dumps({"error": str(e)}); await self.send(err)

    class _ToolInbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.5)
            if not msg or msg.metadata.get("type") != "tool.call":
                return
            try:
                data = json.loads(msg.body or "{}")
                fname, args = data.get("name"), data.get("args", {})  # {"name": "...", "args": {...}}
                fn = self.agent.executables.get(fname)
                if not fn:
                    raise ValueError(f"Unknown tool: {fname}")
                # sync or async tool
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = await asyncio.to_thread(fn, **args)
                reply = Message(to=str(msg.sender)); reply.set_metadata("type", "tool.reply")
                reply.body = json.dumps({"success": True, "result": result})
                await self.send(reply)
            except Exception as e:
                err = Message(to=str(msg.sender)); err.set_metadata("type", "tool.reply")
                err.body = json.dumps({"success": False, "error": str(e)})
                await self.send(err)
