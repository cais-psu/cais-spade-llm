# agents/intelligent_product/product_agent.py
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable, Optional

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent


class ProductAgent(LlmAgent):
    """
    SPADE ProductAgent
    - Reads a product specification (optional) and sends a task to ResourceAgents.
    - Receives ACKs from Resource/Robot agents and logs status.
    - Keeps the PA simple: RA is the single broker that asks the LLM with tools.
    """

    agent_role = "product"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        resource_jids: Optional[Iterable[str]] = None,
        product_specification_file: Optional[str] = None,
        instruction_override: Optional[str] = None,
        broadcast: bool = False,
        **kw,
    ) -> None:
        """
        :param resource_jids: List of RA JIDs to target (first is used if broadcast=False).
        :param product_specification_file: Path to spec text (utf-8). Optional.
        :param instruction_override: If provided, this text is used instead of reading a file.
        :param broadcast: If True, send the task to all resource_jids; else only the first.
        """
        super().__init__(jid, password, name=name, agent_role="product", **kw)
        self.resource_jids = list(resource_jids or [])
        self.product_specification_file = product_specification_file
        self.instruction_override = instruction_override
        self.broadcast = broadcast

        # Simple in-memory map of task_id -> latest status string
        self.task_states: dict[str, str] = {}

        self.logger.info(f"ProductAgent '{name}' initialized.")

    # --------------------------------------------------------------------- #
    # SPADE lifecycle
    # --------------------------------------------------------------------- #

    async def setup(self):
        await super().setup()

        # Kickoff behaviour (runs once)
        self.add_behaviour(self._Kickoff())

        # ACK inbox with a template (only consume type=ack)
        t_ack = Template()
        t_ack.set_metadata("type", "ack")
        self.add_behaviour(self._AckInbox(), t_ack)

        self.logger.info(f"[Product] {self.jid} ready.")

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _read_spec_text(self) -> Optional[str]:
        """Return the instruction text: prefer override, else read file, else None."""
        if self.instruction_override:
            txt = self.instruction_override.strip()
            if txt:
                return txt

        if self.product_specification_file:
            try:
                p = Path(self.product_specification_file)
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
                self.logger.warning(f"[Product] Spec file is empty: {p}")
            except Exception as e:
                self.logger.exception(f"[Product] Failed to read spec: {e}")

        return None

    async def _send_task(
        self,
        *,
        to: str,
        task_id: str,
        instruction: str | dict,
        phase_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        protocol: str = "plan/1.0",
    ) -> None:
        """Compose and send a task message to a single RA."""
        body_payload = {
            "task_id": task_id,
            "instruction": instruction,
        }
        if phase_id:
            body_payload["phase_id"] = phase_id

        msg = Message(to=to)
        msg.set_metadata("type", "task")
        msg.set_metadata("protocol", protocol)
        if correlation_id:
            msg.set_metadata("correlation_id", correlation_id)

        msg.body = json.dumps(body_payload)
        await self.send(msg)

        self.logger.info(f"[Product] Sent task {task_id} → {to}")

    # --------------------------------------------------------------------- #
    # Behaviours
    # --------------------------------------------------------------------- #

    class _Kickoff(OneShotBehaviour):
        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore

            if not agent.resource_jids:
                agent.logger.warning("[Product] No resource_jids; kickoff aborted.")
                return

            instruction = agent._read_spec_text()
            if not instruction:
                agent.logger.warning("[Product] No instruction text; kickoff aborted.")
                return

            # If you prefer PA to pre-digest/condense, you could do:
            # instruction = await agent.ask_llm(instruction, with_functions=False)
            # But RA is the tool broker, so it's fine to forward raw spec.

            # IDs for traceability
            task_id = f"T-{uuid.uuid4().hex[:8].upper()}"
            phase_id = "P-001"
            correlation_id = f"C-{uuid.uuid4().hex[:10].upper()}"

            # Choose recipients
            targets = agent.resource_jids if agent.broadcast else [agent.resource_jids[0]]

            # Send to one or all RAs
            for to in targets:
                await agent._send_task(
                    to=to,
                    task_id=task_id,
                    instruction=instruction,
                    phase_id=phase_id,
                    correlation_id=correlation_id,
                )

            # Initialize local status
            agent.task_states[task_id] = "sent"

    class _AckInbox(CyclicBehaviour):
        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            # Only messages with type=ack reach here (Template)
            correlation = msg.metadata.get("correlation_id", "")
            try:
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed ACK body (not JSON).")
                return

            task_id = payload.get("task_id", "?")
            status = payload.get("status", "unknown")

            # Update local state
            agent.task_states[task_id] = status

            agent.logger.info(
                f"[Product] ACK ({task_id}) status='{status}' "
                f"from={msg.sender} corr={correlation}"
            )
