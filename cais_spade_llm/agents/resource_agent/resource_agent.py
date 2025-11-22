"""Resource agent that receives work orders, selects a tool via LLM, and executes it."""

# agents/resource_agent/resource_agent.py
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Iterable, Optional, Tuple

from spade.behaviour import CyclicBehaviour  # Behaviour base used for our inbox loop.
from spade.message import Message  # SPADE message objects (XMPP stanzas under the hood).
from spade.template import Template  # Filters incoming messages by metadata.

from agents.shared_information.llm_agent import LlmAgent


class ResourceAgent(LlmAgent):
    """
    SPADE ResourceAgent.

    - Receives tasks from ProductAgents (metadata[type] == "task").
    - Calls the LLM with tools enabled to select a function.
    - Dispatches to a registered executable (async function).
    - Sends ACK messages back to the ProductAgent with status.
    """

    agent_role = "resource"  # Used by LlmAgent to pick prompts and instructions for this class.

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
        **kw: Any,
    ) -> None:
        """
        :param function_names: Names of methods on this class to expose as tools.
        :param static_capabilities: Optional metadata for discovery/logging.
        :param allowed_senders: Optional allowlist of JIDs allowed to send tasks.
        :param llm_timeout_s: Timeout for LLM tool selection.
        :param tool_timeout_s: Timeout for tool execution.
        """
        super().__init__(
            jid,
            password,
            name=name,
            agent_role="resource",
            function_names=list(function_names or []),
            **kw,
        )
        # Optional metadata (payload limits, tool list, etc.) exposed to other agents or dashboards.
        self.static_capabilities: Dict[str, Any] = static_capabilities or {}
        # Optional sender allow-list: if populated, only those JIDs can submit work.
        self.allowed_senders = set(allowed_senders or [])
        # Separate timeouts keep LLM latency (planning) independent from tool runtime (execution).
        self.llm_timeout_s = int(llm_timeout_s)
        self.tool_timeout_s = int(tool_timeout_s)

    # ------------------------------------------------------------------ #
    # SPADE lifecycle
    # ------------------------------------------------------------------ #

    async def setup(self) -> None:
        """Register inbox behaviour filtered to 'task' messages."""
        await super().setup()

        # Only receive messages that are type="task"
        t_task = Template()
        t_task.set_metadata("type", "task")  # Ignore chat pings/acks/etc.; only react to tasks.
        self.add_behaviour(self._TaskInbox(), t_task)

    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ #

    class _TaskInbox(CyclicBehaviour):
        """Long-running behaviour that processes incoming tasks sequentially."""
        async def run(self) -> None:
            agent: "ResourceAgent" = self.agent  # type: ignore
            # Poll inbox frequently but yield control if nothing arrives to keep agent responsive.
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            # ----- trust boundary ----- #
            # Give operators a simple safety net: reject unexpected senders early.
            if agent.allowed_senders and str(msg.sender) not in agent.allowed_senders:
                agent.logger.warning(
                    f"[Resource] Rejecting task from {msg.sender} (not allowed)"
                )
                await self._ack(msg, task_id="?", status="rejected:unauthorized")
                return

            # ----- envelope ----- #
            # Protocol metadata lets us bump behaviours later (e.g., plan/exec distinctions).
            protocol = msg.metadata.get("protocol", "")

            # ----- parse body ----- #
            try:
                data = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Resource] Malformed task body (not JSON).")
                await self._ack(msg, task_id="?", status="failed:bad_json")
                return

            task_id = data.get("task_id")
            instruction = data.get("instruction", "")
            phase_id = data.get("phase_id")  # optional multi-phase flow identifier

            if not task_id:
                agent.logger.warning("[Resource] Task without task_id.")
                await self._ack(msg, task_id="?", status="failed:missing_task_id")
                return

            agent.logger.info(
                f"[Resource] ← Task ({task_id}) from={msg.sender} proto={protocol}"
            )

            # ----- EARLY ACK ----- #
            # Confirm receipt immediately so the ProductAgent can show progress even before execution.
            await self._ack(msg, task_id=task_id, status="accepted")

            # ----- LLM tool selection ----- #
            try:
                # Force the LLM to pick an explicit tool so we never free-text a task.
                llm_resp = await asyncio.wait_for(
                    agent.ask_llm(
                        instruction,
                        with_functions=True,
                        force_tool=True,
                    ),
                    timeout=agent.llm_timeout_s,
                )
            except asyncio.TimeoutError:
                await self._ack(msg, task_id=task_id, status="llm_timeout")
                return
            except Exception as e:
                agent.logger.exception("[Resource] LLM failure")
                await self._ack(
                    msg,
                    task_id=task_id,
                    status=f"failed:llm:{type(e).__name__}",
                )
                return

            fn_name, fn_args = _parse_function_call(llm_resp)
            if not fn_name:
                agent.logger.info(
                    f"[Resource] ({task_id}) no_tool_match; responding."
                )
                await self._ack(msg, task_id=task_id, status="no_tool_match")
                return

            # ----- plumb routing/context ----- #
            # Pass routing info into the tool implementation for downstream logging/rpc calls.
            fn_args.setdefault("sender_jid", str(msg.sender))
            fn_args.setdefault("task_id", task_id)
            if phase_id and "phase_id" not in fn_args:
                fn_args["phase_id"] = phase_id

            # ----- dispatch ----- #
            func = agent.executables.get(fn_name)
            if not func:
                agent.logger.warning(f"[Resource] Unknown tool '{fn_name}'")
                await self._ack(
                    msg,
                    task_id=task_id,
                    status=f"failed:unknown_tool:{fn_name}",
                )
                return

            agent.logger.info(f"[Resource] ({task_id}) Calling {fn_name}({fn_args})")
            try:
                result = await asyncio.wait_for(
                    func(**fn_args),
                    timeout=agent.tool_timeout_s,
                )
                # Tool implementations optionally return {"status": "..."}; default to completed.
                status = (result or {}).get("status") or "completed"
            except asyncio.TimeoutError:
                status = "tool_timeout"
            except Exception as e:
                agent.logger.exception("[Resource] Tool execution failed")
                status = f"failed:tool:{type(e).__name__}"

            # ----- FINAL ACK ----- #
            # Always reply back so the ProductAgent doesn't have to rely on timeouts.
            await self._ack(msg, task_id=task_id, status=status)

        async def _ack(
            self,
            msg: Message,
            *,
            task_id: Optional[str],
            status: str,
        ) -> None:
            """Send an acknowledgement/status update back to the originating ProductAgent."""
            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", "ack")
            reply.body = json.dumps({"task_id": task_id, "status": status})
            await self.send(reply)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _parse_function_call(resp):
    """Normalize the OpenAI response into (tool_name, args_dict)."""
    fc = resp.get("function_call")
    if not isinstance(fc, dict):
        return None, {}

    name = fc.get("name")
    args_raw = fc.get("arguments", {})

    if isinstance(args_raw, str):
        # OpenAI may return arguments as a JSON string; parse defensively.
        try:
            return name, json.loads(args_raw)
        except json.JSONDecodeError:
            return name, {}

    return name, args_raw
