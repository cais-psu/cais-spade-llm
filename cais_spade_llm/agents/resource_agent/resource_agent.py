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
        cca_jid: Optional[str] = None,   # <-- NEW
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

        self.cca_jid = cca_jid

        # Optional metadata (payload limits, tool list, etc.) exposed to other agents or dashboards.
        self.static_capabilities: Dict[str, Any] = static_capabilities or {}
        # Optional sender allow-list: if populated, only those JIDs can submit work.
        self.allowed_senders = set(allowed_senders or [])
        # Separate timeouts keep LLM latency (planning) independent from tool runtime (execution).
        self.llm_timeout_s = int(llm_timeout_s)
        self.tool_timeout_s = int(tool_timeout_s)

        self._safety_decisions: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # SPADE lifecycle
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        await super().setup()

        # Tasks from ProductAgent
        t_task = Template()
        t_task.set_metadata("type", "task")
        self.add_behaviour(self._TaskInbox(), t_task)

        # Safety decisions from CCA
        t_safety = Template()
        t_safety.set_metadata("type", "safety_decision")
        self.add_behaviour(self._SafetyDecisionInbox(), t_safety)

    async def _wait_for_safety_decision(self, task_id: str) -> Optional[str]:
        """
        Block until a safety_decision is available for this task_id.
        No timeout: waits indefinitely until CCA replies.
        """
        while True:
            decision = self._safety_decisions.pop(task_id, None)
            if decision is not None:
                return decision
            await asyncio.sleep(0.1)
    # ------------------------------------------------------------------ #
    # Behaviours
    # ------------------------------------------------------------------ ##
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
            fn_args.setdefault("product_jid", str(msg.sender))
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

            # ----- RESOURCE EVENT (FOR SAFETY) NOTIFICATION TO CCA ----- #
            try:
                # 1) Send request permission, not running
                resource_msg = Message(to=agent.cca_jid)
                resource_msg.set_metadata("type", "resource_event")
                resource_msg.body = json.dumps({
                    "task_id": task_id,
                    "resource_jid": str(agent.jid),
                    "function_name": fn_name,
                    "params": fn_args,
                    "status": "safety_check",   # <-- REQUEST permission
                })
                await self.send(resource_msg)
            except Exception:
                agent.logger.exception("[Resource] Failed to send resource_event to CCA (ignored).")

            # 2) Wait for CCA decision (no timeout; uses _SafetyDecisionInbox + dict)
            decision = await agent._wait_for_safety_decision(task_id)

            if decision == "block":
                await self._ack(msg, task_id=task_id, status="blocked")
                return

            # ---------------------------
            #  SAFETY PASSED → RUNNING
            # ---------------------------
            await self._ack(msg, task_id=task_id, status="running")

            running_msg = Message(to="cca@localhost")
            running_msg.set_metadata("type", "resource_event")
            running_msg.body = json.dumps({
                "task_id": task_id,
                "resource_jid": str(agent.jid),
                "function_name": fn_name,
                "params": fn_args,
                "status": "running",
            })
            await self.send(running_msg)

            # ---------------------------
            #  EXECUTE THE TOOL
            # ---------------------------
            try:
                result = await asyncio.wait_for(
                    func(**fn_args),
                    timeout=agent.tool_timeout_s,
                )
                final_status = (result or {}).get("status") or "completed"

                # ----- RESOURCE EVENT: TASK FINISHED (notify CCA) ----- #
                try:
                    done_msg = Message(to="cca@localhost")
                    done_msg.set_metadata("type", "resource_event")
                    done_msg.body = json.dumps({
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": final_status,  # e.g. "completed", "blocked", etc.
                    })
                    # fire-and-forget so we don't block on CCA
                    asyncio.create_task(self.send(done_msg))
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send final resource_event to CCA (ignored)."
                    )

            except asyncio.TimeoutError:
                agent.logger.exception("[Resource] Tool execution timeout")
                final_status = "tool_timeout"
            except Exception as e:
                agent.logger.exception("[Resource] Tool execution failed")
                final_status = f"failed:tool:{type(e).__name__}"

            # ---------------------------
            #  SEND FINAL ACK TO PA
            # ---------------------------
            await self._ack(msg, task_id=task_id, status=final_status)

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

    class _SafetyDecisionInbox(CyclicBehaviour):
        """Receives safety_decision messages from CCA and stores them on the agent."""
        async def run(self) -> None:
            agent: "ResourceAgent" = self.agent  # type: ignore

            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                data = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Resource] Malformed safety_decision body")
                return

            task_id = data.get("task_id")
            decision = data.get("decision")

            if not task_id or decision not in ("allow", "block"):
                agent.logger.warning(
                    "[Resource] Invalid safety_decision message: %s", data
                )
                return

            agent._safety_decisions[task_id] = decision
            agent.logger.info(
                "[Resource] Stored safety_decision=%s for task=%s",
                decision,
                task_id,
            )

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
