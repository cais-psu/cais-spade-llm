# agents/resource_agent/resource_agent.py
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Iterable, Optional

from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent


class ResourceAgent(LlmAgent):
    """
    SPADE ResourceAgent
    - Receives tasks from ProductAgents (type=task).
    - Asks the LLM WITH tools enabled to choose a function.
    - Dispatches to a registered executable (async function) and returns an ACK.

    Assumptions about LlmAgent:
      - accepts function_names=... to expose tools to the LLM
      - has self.executables: Dict[str, Callable[..., Awaitable[Dict[str, Any]]]]
      - provides ask_llm(prompt: str|dict, with_functions: bool) -> dict|str
    """

    agent_role = "resource"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        function_names: Optional[Iterable[str]] = None,
        static_capabilities: Optional[Dict[str, Any]] = None,
        allowed_senders: Optional[Iterable[str]] = None,
        llm_timeout_s: int = 30,
        tool_timeout_s: int = 300,
        **kw,
    ) -> None:
        """
        :param function_names: Names of functions to expose to LLM (must be registered in `self.executables`).
        :param static_capabilities: Free-form dictionary you can expose/log for discovery.
        :param allowed_senders: Optional allowlist of JIDs that can send tasks.
        :param llm_timeout_s: Timeout for LLM tool-selection.
        :param tool_timeout_s: Timeout for the called tool execution.
        """
        super().__init__(
            jid,
            password,
            name=name,
            agent_role="resource",
            function_names=list(function_names or []),
            **kw,
        )
        self.static_capabilities = static_capabilities or {}
        self.allowed_senders = set(allowed_senders or [])
        self.llm_timeout_s = int(llm_timeout_s)
        self.tool_timeout_s = int(tool_timeout_s)

    # --------------------------------------------------------------------- #
    # SPADE lifecycle
    # --------------------------------------------------------------------- #

    async def setup(self):
        await super().setup()

        # Only receive messages that are type=task
        t_task = Template()
        t_task.set_metadata("type", "task")
        self.add_behaviour(self._TaskInbox(), t_task)

    # --------------------------------------------------------------------- #
    # Behaviours
    # --------------------------------------------------------------------- #

    class _TaskInbox(CyclicBehaviour):
        async def run(self):
            agent: "ResourceAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            # ----- trust boundary -----
            if agent.allowed_senders and str(msg.sender) not in agent.allowed_senders:
                agent.logger.warning(f"[Resource] Rejecting task from {msg.sender} (not allowed)")
                await self._ack(
                    msg,
                    task_id="?",
                    status="rejected:unauthorized",
                )
                return

            # ----- envelope -----
            protocol = msg.metadata.get("protocol", "")

            # ----- parse body -----
            try:
                data = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Resource] Malformed task body (not JSON).")
                await self._ack(msg, task_id="?", status="failed:bad_json")
                return

            task_id = data.get("task_id")
            instruction = data.get("instruction", "")
            phase_id = data.get("phase_id")  # optional

            if not task_id:
                agent.logger.warning("[Resource] Task without task_id.")
                await self._ack(msg, task_id="?", status="failed:missing_task_id")
                return

            agent.logger.info(
                f"[Resource] ← Task ({task_id}) from={msg.sender} "
                f"proto={protocol}"
            )

            # ----- EARLY ACK: accepted / queued -----
            # If you have local queuing, detect it and send "queued"; else "accepted".
            await self._ack(msg, task_id=task_id, status="accepted")

            # ----- LLM tool selection -----
            try:
                llm_resp = await asyncio.wait_for(
                    agent.ask_llm(instruction, with_functions=True, force_tool=True),  # <- requires your earlier change
                    timeout=agent.llm_timeout_s,
                )
            except asyncio.TimeoutError:
                await self._ack(msg, task_id=task_id, status="llm_timeout")
                return
            except Exception as e:
                agent.logger.exception("[Resource] LLM failure")
                await self._ack(msg, task_id=task_id, status=f"failed:llm:{type(e).__name__}")
                return

            fn_name, fn_args = _parse_function_call(llm_resp)
            if not fn_name:
                agent.logger.info(f"[Resource] ({task_id}) no_tool_match; responding.")
                await self._ack(msg, task_id=task_id, status="no_tool_match")
                return

            # ----- plumb routing/context -----
            fn_args.setdefault("sender_jid", str(msg.sender))
            fn_args.setdefault("task_id", task_id)
            if phase_id and "phase_id" not in fn_args:
                fn_args["phase_id"] = phase_id

            # ----- dispatch -----
            func = agent.executables.get(fn_name)
            if not func:
                agent.logger.warning(f"[Resource] Unknown tool '{fn_name}'")
                await self._ack(msg, task_id=task_id, status=f"failed:unknown_tool:{fn_name}")
                return

            agent.logger.info(f"[Resource] ({task_id}) Calling {fn_name}({fn_args})")
            try:
                result = await asyncio.wait_for(
                    func(**fn_args), timeout=agent.tool_timeout_s
                )

                # Tool should return a dict, at minimum {"status": "..."}
                status = (result or {}).get("status") or "completed"
            except asyncio.TimeoutError:
                status = "tool_timeout"
            except Exception as e:
                agent.logger.exception("[Resource] Tool execution failed")
                status = f"failed:tool:{type(e).__name__}"

            # ----- FINAL ACK -----
            await self._ack(msg, task_id=task_id, status=status)

        async def _ack(self, msg: Message, *, task_id: Optional[str], status: str):
            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", "ack")
            reply.body = json.dumps({"task_id": task_id, "status": status})
            await self.send(reply)

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _parse_function_call(resp: Any) -> tuple[str | None, Dict[str, Any]]:
    """
    Extract (name, args) from OpenAI tool call responses.
    Supports both:
      - legacy {"function_call": {"name": "...", "arguments": "..."}}
      - modern {"tool_calls": [{"type":"function","function":{"name","arguments"}}]}
    Returns ("name", dict_args) or (None, {}).
    """
    if not isinstance(resp, dict):
        return None, {}

    # Newer format: tool_calls is a list; take the first function call
    tool_calls = resp.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0] or {}
        if (first.get("type") == "function") and isinstance(first.get("function"), dict):
            f = first["function"]
            name = f.get("name")
            args_raw = f.get("arguments", {})
            if isinstance(args_raw, str):
                try:
                    return name, json.loads(args_raw)
                except json.JSONDecodeError:
                    return name, {}
            return name, (args_raw or {})
        # If tool_calls exists but is unusable, fall through to legacy parsing

    # Legacy format: single function_call object
    fc = resp.get("function_call")
    if isinstance(fc, dict):
        name = fc.get("name")
        args_raw = fc.get("arguments", {})
        if isinstance(args_raw, str):
            try:
                return name, json.loads(args_raw)
            except json.JSONDecodeError:
                return name, {}
        return name, (args_raw or {})

    return None, {}

