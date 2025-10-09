# agents/intelligent_product/product_agent.py
from __future__ import annotations
import json, asyncio
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template
from agents.shared_information.llm_agent import LlmAgent

class ProductAgent(LlmAgent):
    agent_role = "product"

    def __init__(
        self, jid: str, password: str, *,
        name: str,
        resource_jids: list[str] | None = None,
        function_names: list[str] | None = None,
        annotation: str | None = None,
        instructions: str | None = None,
        **kw,
    ):
        super().__init__(
            jid, password,
            name=name,
            annotation=annotation,
            instructions=instructions,
            function_names=function_names,
            **kw,
        )
        self.resource_jids = resource_jids or []

    async def setup(self):
        await super().setup()

        # receive acks from resources
        t_ack = Template(); t_ack.set_metadata("type", "ack")
        self.add_behaviour(self._AckInbox(), t_ack)

        # optional one-shot kickoff
        self.add_behaviour(self._Kickoff())

    class _Kickoff(OneShotBehaviour):
        async def run(self):
            if not self.agent.resource_jids:
                return
            # ask LLM to draft an instruction (or build your own)
            text = await self.agent.ask_llm(
                "The part MCP needs to be instructed to be assembled from prusa-mk4-2 to placed in assembly station.",
                with_functions=False,
            )

            payload = {"task_id": "T-001", "instruction": text}
            # send to the first resource for demo
            to = self.agent.resource_jids[0]
            msg = Message(to=to); msg.set_metadata("type", "task")
            msg.body = json.dumps(payload)
            await self.send(msg)
            self.agent.logger.info(f"[Product] sent {payload['task_id']} → {to} :: {text}")

    class _AckInbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.5)
            if not msg or msg.metadata.get("type") != "ack":
                return
            data = json.loads(msg.body or "{}")
            self.agent.logger.info(f"[Product] ACK: {data}")

