# agents/resource_agent/resource_agent.py
from __future__ import annotations
import json, asyncio
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from agents.shared_information.llm_agent import LlmAgent


class ResourceAgent(LlmAgent):
    agent_role = "resource"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        function_names: list[str] | None = None,
        static_capabilities: dict | None = None,
        vector_dbs: dict | None = None,
        annotation: str | None = None,
        instructions: str | None = None,
    ) -> None:
        # Only pass SPADE-safe args up; include the role
        super().__init__(
            jid,
            password,
            name=name,
            agent_role="resource",
            annotation=annotation,
            instructions=instructions,   # per-agent overrides from JSON (optional)
            function_names=function_names,
        )
        # Resource-level state
        self.static_capabilities = static_capabilities or {}
        self.vector_dbs = vector_dbs or {}

    async def setup(self):
        await super().setup()

        # Task inbox
        t_task = Template()
        t_task.set_metadata("type", "task")
        self.add_behaviour(self._TaskInbox(), t_task)

    class _TaskInbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.5)
            if not msg or msg.metadata.get("type") != "task":
                return

            try:
                data = json.loads(msg.body or "{}")
            except Exception:
                data = {}

            task_id = data.get("task_id")
            instruction = data.get("instruction", "").strip()

            # If your ROLE_BLOCKS['resource'] tells the agent to validate/clarify
            # and output one executable step, we can just pass the instruction text.
            # No inline prompt strings here.
            plan = instruction
            if instruction:
                plan = await self.agent.ask_llm(instruction, with_functions=False)

            self.agent.logger.info(f"[Resource] executing ({task_id}): {plan}")

            # Simulate execution (replace with real handler)
            await asyncio.sleep(1.0)

            # Send ACK
            reply = Message(to=str(msg.sender))
            reply.set_metadata("type", "ack")
            reply.body = json.dumps({"task_id": task_id, "status": "completed"})
            await self.send(reply)
