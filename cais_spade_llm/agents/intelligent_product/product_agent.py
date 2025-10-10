# agents/intelligent_product/product_agent.py
from __future__ import annotations
import json
from pathlib import Path
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent


class ProductAgent(LlmAgent):
    agent_role = "product"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        resource_jids: list[str] | None = None,
        function_names: list[str] | None = None,
        annotation: str | None = None,
        instructions: str | None = None,
        product_specification_file: str | None = None,
        cad_files: list[str] | None = None,
        vector_dbs: dict | None = None,
    ) -> None:
        # Only pass SPADE-safe args up
        super().__init__(
            jid,
            password,
            name=name,
            agent_role="product",
            annotation=annotation,
            instructions=instructions,
            function_names=function_names,
        )

        # Product-level state
        self.resource_jids = resource_jids or []
        self.product_specification_file = product_specification_file
        self.cad_files = cad_files or []
        self.vector_dbs = vector_dbs or {}

        # Optional local state
        self.inbox: list = []
        self.task_states: dict = {}

        # Log initialization
        self.logger.info(f"ProductAgent '{name}' initialized with spec '{product_specification_file}'.")

    # ---------- SPADE lifecycle ----------
    async def setup(self):
        await super().setup()

        # Receive ACKs from resources
        t_ack = Template()
        t_ack.set_metadata("type", "ack")
        self.add_behaviour(self._AckInbox(), t_ack)

        # Optional one-shot kickoff (driven by assembly_instructions.txt)
        self.add_behaviour(self._Kickoff())

        self.logger.info(f"[Product] {self.jid} ready and waiting for assembly instructions.")

    # ---------- Behaviours ----------
    class _Kickoff(OneShotBehaviour):
        async def run(self):
            if not self.agent.resource_jids:
                self.agent.logger.warning("[Product] No resource_jids configured; kickoff aborted.")
                return

            p = self.agent.product_specification_file
            if not p:
                self.agent.logger.warning("[Product] No product_specification_file set; kickoff aborted.")
                return

            try:
                # Load assembly_instructions.txt verbatim; no prompt text here
                spec_text = Path(p).read_text(encoding="utf-8").strip()
            except Exception as e:
                self.agent.logger.error(f"[Product] Failed to read spec file '{p}': {e}")
                return

            if not spec_text:
                self.agent.logger.warning(f"[Product] Spec file '{p}' is empty; kickoff aborted.")
                return

            # Ask LLM using only system prompt (from prompts.py) + user text (the file contents)
            instruction = await self.agent.ask_llm(spec_text, with_functions=False)

            payload = {"task_id": "T-001", "instruction": instruction}

            # Send to the first resource for now (routing policy can evolve later)
            to = self.agent.resource_jids[0]
            msg = Message(to=to)
            msg.set_metadata("type", "task")
            msg.body = json.dumps(payload)
            await self.send(msg)

            self.agent.logger.info(f"[Product] sent {payload['task_id']} → {to}")

    class _AckInbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.5)
            if not msg or msg.metadata.get("type") != "ack":
                return
            try:
                data = json.loads(msg.body or "{}")
            except Exception:
                data = {"raw": msg.body}
            self.agent.logger.info(f"[Product] ACK: {data}")
