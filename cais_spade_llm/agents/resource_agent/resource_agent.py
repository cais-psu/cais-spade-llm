"""Resource agent that receives work orders, selects a tool via LLM, and executes it."""

# agents/resource_agent/resource_agent.py
from __future__ import annotations

import asyncio
import inspect
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

    def _snapshot_state(self) -> Dict[str, Any]:
        """
        Best-effort snapshot of resource state for failure context.
        Subclasses can override to provide richer state.
        """
        return {}

    def _build_failure_context(
        self,
        *,
        fn_name: str,
        fn_args: Dict[str, Any],
        result: Dict[str, Any] | None,
        final_status: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a normalized failure context payload for any failure type.

        Canonical schema:
          - failure_class
          - failure_mode
          - retryable
          - severity
          - affected_entities (optional)
          - observations (optional)
        """
        failure_context: Dict[str, Any] = {}

        if isinstance(result, dict) and isinstance(result.get("failure_context"), dict):
            failure_context.update(result.get("failure_context") or {})

        is_failed = isinstance(final_status, str) and final_status.startswith("failed")
        if not is_failed:
            return failure_context

        raw_mode = final_status.split(":", 1)[1] if ":" in final_status else final_status
        raw_mode = str(raw_mode).strip().lower()
        requested_mode = str(
            failure_context.get("failure_mode") or raw_mode
        ).strip().lower()

        # Canonical failure taxonomy used across all resources/tools.
        canonical_modes = {
            "slippage",
            "breakdown",
            "timeout",
            "collision",
            "unreachable",
            "safety_block",
            "unknown",
        }
        mode_aliases = {
            "tool_timeout": "timeout",
            "llm_timeout": "timeout",
            "resource_lost": "breakdown",
            "hardware_fault": "breakdown",
            "gripper_jam": "breakdown",
            "jam": "breakdown",
            "safety_violation": "safety_block",
            "blocked": "safety_block",
            "coordination_block": "safety_block",
            "path_blocked": "unreachable",
        }
        normalized_mode = mode_aliases.get(requested_mode, requested_mode)
        failure_mode = normalized_mode if normalized_mode in canonical_modes else "unknown"

        default_class_by_mode = {
            "breakdown": "resource_failure",
            "collision": "environment_failure",
            "unreachable": "environment_failure",
            "safety_block": "coordination_failure",
            "timeout": "execution_failure",
            "slippage": "execution_failure",
            "unknown": "execution_failure",
        }
        failure_class = str(
            failure_context.get("failure_class")
            or default_class_by_mode.get(failure_mode, "execution_failure")
        )

        non_retryable_modes = {"breakdown", "collision"}
        retryable = failure_context.get("retryable")
        if not isinstance(retryable, bool):
            retryable = failure_mode not in non_retryable_modes

        default_severity_by_mode = {
            "breakdown": "high",
            "collision": "high",
            "unreachable": "medium",
            "safety_block": "medium",
            "timeout": "medium",
            "slippage": "medium",
            "unknown": "medium",
        }
        severity = str(
            failure_context.get("severity")
            or default_severity_by_mode.get(failure_mode, "medium")
        )

        affected_entities = failure_context.get("affected_entities")
        if not isinstance(affected_entities, list):
            affected_entities = []
        if not affected_entities and fn_args.get("part_name"):
            affected_entities = [
                {
                    "entity_type": "part",
                    "entity_id": str(fn_args.get("part_name")),
                    "state": "unknown",
                }
            ]

        observations = failure_context.get("observations")
        if not isinstance(observations, dict):
            observations = {}
        observations.setdefault("function_name", fn_name)
        observations.setdefault("status", str(final_status))
        observations.setdefault("raw_failure_mode", requested_mode)
        if state_before:
            observations.setdefault("state_before", state_before)
        if state_after:
            observations.setdefault("state_after", state_after)

        canonical = {
            "failure_class": failure_class,
            "failure_mode": failure_mode,
            "retryable": retryable,
            "severity": severity,
            "affected_entities": affected_entities,
            "observations": observations,
        }

        # Preserve custom extension fields while keeping canonical keys stable.
        for k, v in failure_context.items():
            if k not in canonical:
                canonical[k] = v

        return canonical

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

            # ----- Tool selection ----- #
            # If the instruction already specifies a tool, honor it and skip the LLM.
            fn_name = None
            fn_args: Dict[str, Any] = {}
            if isinstance(instruction, dict):
                fn_name = instruction.get("function_name") or instruction.get("function")
                if isinstance(instruction.get("params"), dict):
                    fn_args = dict(instruction.get("params") or {})

            if not fn_name:
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

            running_msg = Message(to=agent.cca_jid)
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
            state_before = agent._snapshot_state()
            state_after = state_before
            result: Dict[str, Any] | None = None
            try:
                # Filter fn_args to only params the function accepts.
                # Functions that declare **kwargs receive everything;
                # others get only the params in their signature.
                sig = inspect.signature(func)
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if accepts_var_kw:
                    filtered_args = fn_args
                else:
                    accepted = set(sig.parameters.keys())
                    filtered_args = {k: v for k, v in fn_args.items() if k in accepted}

                result = await asyncio.wait_for(
                    func(**filtered_args),
                    timeout=agent.tool_timeout_s,
                )
                state_after = agent._snapshot_state()
                final_status = (result or {}).get("status") or "completed"

                # ----- RESOURCE EVENT: TASK FINISHED (notify CCA) ----- #
                try:
                    failure_context = agent._build_failure_context(
                        fn_name=fn_name,
                        fn_args=fn_args,
                        result=result if isinstance(result, dict) else None,
                        final_status=final_status,
                        state_before=state_before,
                        state_after=state_after,
                    )

                    done_msg = Message(to=agent.cca_jid)
                    done_msg.set_metadata("type", "resource_event")
                    done_msg.body = json.dumps({
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": final_status,  # e.g. "completed", "blocked", etc.
                        "failure_context": failure_context,
                        "current_state": state_after.get("current_state", "idle"),
                    })
                    # fire-and-forget so we don't block on CCA
                    asyncio.create_task(self.send(done_msg))
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send final resource_event to CCA (ignored)."
                    )

            except asyncio.TimeoutError:
                agent.logger.exception("[Resource] Tool execution timeout")
                final_status = "failed:tool_timeout"
            except Exception as e:
                agent.logger.exception("[Resource] Tool execution failed")
                final_status = f"failed:tool:{type(e).__name__}"

            # Notify CCA of failure so it can clean up running_aps / FSA state.
            if isinstance(final_status, str) and final_status.startswith("failed"):
                try:
                    state_after = agent._snapshot_state()
                    failure_context = agent._build_failure_context(
                        fn_name=fn_name,
                        fn_args=fn_args,
                        result=result if isinstance(result, dict) else None,
                        final_status=final_status,
                        state_before=state_before,
                        state_after=state_after,
                    )
                    fail_msg = Message(to=agent.cca_jid)
                    fail_msg.set_metadata("type", "resource_event")
                    fail_msg.body = json.dumps({
                        "task_id": task_id,
                        "resource_jid": str(agent.jid),
                        "function_name": fn_name,
                        "params": fn_args,
                        "status": final_status,
                        "failure_context": failure_context,
                        "current_state": state_after.get("current_state", "idle"),
                    })
                    await self.send(fail_msg)
                except Exception:
                    agent.logger.exception(
                        "[Resource] Failed to send fail resource_event to CCA."
                    )

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
