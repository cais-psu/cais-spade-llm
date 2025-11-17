"""High-level product agent that reads build instructions and instructs a resource agent."""

from __future__ import annotations

import json
import os, uuid
from pathlib import Path
from typing import Iterable, Optional

from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent
from agents.intelligent_product.process_planner import ProcessPlanner


class ProductAgent(LlmAgent):
    """
    SPADE ProductAgent
    - Reads a product specification (optional) and sends a task to ResourceAgents.
    - Receives ACKs from Resource/Robot agents and logs status.
    - Keeps the PA simple: RA is the single broker that asks the LLM with tools.
    """

    agent_role = "product"  # Registered role so the shared LLM base class can fetch the right prompts.

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        resource_jids: Optional[Iterable[str]] = None,
        product_specification_file: Optional[str] = None,
        instruction_override: Optional[str] = None,
        **kw,
    ) -> None:
        """
        :param resource_jids: List of RA JIDs to target (first is used).
        :param product_specification_file: Path to spec text (utf-8). Optional.
        :param instruction_override: If provided, this text is used instead of reading a file.
        """
        super().__init__(jid, password, name=name, agent_role="product", **kw)
        # Resource agents are the downstream executors; keep them in order and avoid mutating caller lists.
        self.resource_jids = list(resource_jids or [])
        self.product_specification_file = product_specification_file
        # Manual instruction text provided at runtime overrides any file read.
        self.instruction_override = instruction_override

        # Planner scaffolding (optional DAG building)
        self.plan_path = Path("cais_spade_llm/plan") / f"{name}_plan.json"
        self.process_planner = ProcessPlanner(self, [])

        # Simple in-memory map of task_id -> latest status string so UI/debug tooling can query progress.
        self.task_states: dict[str, str] = {}

        self.logger.info(f"ProductAgent '{name}' initialized.")

    # --------------------------------------------------------------------- #
    # SPADE lifecycle
    # --------------------------------------------------------------------- #

    async def setup(self):
        await super().setup()

        # Kickoff behaviour (runs once) to send the initial task to a resource agent.
        self.add_behaviour(self._Kickoff())

        # ACK inbox with a template (only consume type=ack)
        t_ack = Template()
        t_ack.set_metadata("type", "ack")
        # Register a cyclic behaviour to watch for acknowledgements from resource agents.
        self.add_behaviour(self._AckInbox(), t_ack)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _read_spec_text(self) -> Optional[str]:
        """Return the instruction text: prefer override, else read file, else None."""
        if self.instruction_override:
            # Honour explicit overrides so tests and manual runs can inject free-form prompts.
            txt = self.instruction_override.strip()
            if txt:
                return txt

        if self.product_specification_file:
            try:
                cwd = os.getcwd()
                self.logger.info(f"[Product] Current working directory: {cwd}")
                p = Path(self.product_specification_file)
                self.logger.info(f"[Product] Attempting to open: {p.resolve()}")
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
                self.logger.warning(f"[Product] Spec file is empty: {p}")
            except Exception as e:
                self.logger.exception(f"[Product] Failed to read spec: {e}")

        return None

    def _compose_task_msg(
        self,
        *,
        to: str,
        task_id: str,
        instruction: str | dict,
        phase_id: str | None = None,
        protocol: str = "plan/1.0",
    ) -> Message:
        """Create the structured SPADE Message that drives resource-agent task execution."""
        # Build a JSON payload with the instruction plus optional phase metadata.
        body = {"task_id": task_id, "instruction": instruction}
        if phase_id:
            body["phase_id"] = phase_id
        msg = Message(to=to)
        # Metadata drives the SPADE routing/filtering logic downstream.
        msg.set_metadata("type", "task")
        msg.set_metadata("protocol", protocol)
        msg.body = json.dumps(body)
        return msg

    def _extract_requirement_text(self) -> Optional[str]:
        """
        Load requirement snippets from either an override path or the default
        specification/products/requirements/<product>.txt file.
        """
        candidates = [Path(self.product_specification_file)]
        for req_path in candidates:
            try:
                if not req_path.exists():
                    continue
                txt = req_path.read_text(encoding="utf-8").strip()
                if txt:
                    self.logger.info(f"[Product] Using requirement file: {req_path.resolve()}")
                    return txt
                self.logger.warning(f"[Product] Requirement file empty: {req_path}")
            except Exception as exc:
                self.logger.exception(
                    "[Product] Failed to read requirements from %s: %s",
                    req_path,
                    exc,
                )
        return None

    def _build_high_level_plan(self, requirement_text: str) -> Optional[str]:
        """
        Build / save the high-level plan DAG using ProcessPlanner.
        """
        if not requirement_text:
            return None
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        message = self.process_planner.build_high_level(requirement_text)
        self.process_planner.save(self.plan_path)
        return message

    # --------------------------------------------------------------------- #
    # Behaviours
    # --------------------------------------------------------------------- #

    class _Kickoff(OneShotBehaviour):
        """Bootstrap behaviour that transforms the product spec into a single task message."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore

            # Without a resource to talk to, nothing can happen.
            if not agent.resource_jids:
                agent.logger.warning("[Product] No resource_jids; kickoff aborted.")
                return

            # Pull requirement text from the product requirement file.
            instruction = agent._extract_requirement_text()
            if not instruction:
                agent.logger.warning("[Product] No instruction text; kickoff aborted.")
                return

            plan_msg = agent._build_high_level_plan(instruction)
            if plan_msg:
                agent.logger.info(f"[Product] {plan_msg} (saved to {agent.plan_path})")

            # Short random identifiers keep logs readable across multiple runs.
            task_id = f"T-{uuid.uuid4().hex[:4].upper()}"
            phase_id = f"P-{uuid.uuid4().hex[:4].upper()}"

            # Choose exactly one target (no broadcast); first entry is the preferred RA.
            to = agent.resource_jids[0]

            msg = agent._compose_task_msg(
                to=to,
                task_id=task_id,
                instruction=instruction,
                phase_id=phase_id,
            )

            # Fire-and-forget task message; SPADE handles routing to the remote resource agent.
            await self.send(msg)
            agent.logger.info(f"[Product] Sent task {task_id} -> {to} with {msg}")

            # Track progress so future ACKs can update human-readable state.
            agent.task_states[task_id] = "sent"

    class _AckInbox(CyclicBehaviour):
        """Background behaviour that listens for acknowledgements from resource agents."""

        async def run(self):
            agent: "ProductAgent" = self.agent  # type: ignore
            # Poll SPADE inbox with a short timeout so other behaviours can interleave.
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            try:
                # ACKs reuse the JSON body format produced by resource agents.
                payload = json.loads(msg.body or "{}")
            except json.JSONDecodeError:
                agent.logger.warning("[Product] Malformed ACK body (not JSON).")
                return

            task_id = payload.get("task_id", "?")
            status = payload.get("status", "unknown")

            # Update local state (and allow higher-level UI hooks to inspect current task status).
            agent.task_states[task_id] = status

            agent.logger.info(
                f"[Product] ACK ({task_id}) status='{status}' from={msg.sender}"
            )
