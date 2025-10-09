# agents/resource_agent/robot_agent.py
from __future__ import annotations
import json, asyncio
from typing import Any, Dict, List
from agents.resource_agent.resource_agent import ResourceAgent
#from agents.shared_information.history import History
from spade.message import Message

class RobotAgent(ResourceAgent):
    """Robot-specific resource (UR5e, xArm) that executes assembly tasks."""
    agent_role = "robot"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        annotation: str | None = None,
        instructions: str | None = None,
        function_names: List[str] | None = None,   # include "assembly"
        static_capabilities: Dict[str, Any] | None = None,
        vector_dbs: Dict[str, Any] | None = None,
        **kw,
    ):
        super().__init__(
            jid, password,
            name=name,
            annotation=annotation,
            instructions=instructions,
            function_names=function_names,
            static_capabilities=static_capabilities,
            vector_dbs=vector_dbs,
            **kw,
        )
        self.history = History([f"manumas/configs/{name}.json"])
        self._busy = False
        self._queue: list[Dict[str, Any]] = []
        self.logger.info(f"RobotAgent '{name}' initialized.")

    # keep the ResourceAgent setup (it already adds _TaskInbox etc.)
    async def setup(self):
        await super().setup()
        self.logger.info(f"[Robot] {self.jid} ready and listening for tasks.")

    # ---------------------------------------------------------------------- #
    # Robot-specific tool – exposed as function for LLM or called by inbox
    # ---------------------------------------------------------------------- #
    async def assembly(
        self,
        product_name: str,
        part_name_list: list[str],
        origin_resource_location: str,
        sender_jid: str | None = None,
        phase_id: str | None = None,
        task_id: str | None = None,
    ) -> dict:
        """Execute assembly sequence asynchronously (capacity = 1)."""
        job = dict(
            product_name=product_name,
            part_name_list=part_name_list,
            origin_resource_location=origin_resource_location,
            sender_jid=sender_jid,
            phase_id=phase_id,
            task_id=task_id,
        )

        if self._busy:
            self._queue.append(job)
            self.logger.info(f"[Robot] Busy – queued job {task_id}")
            return {"status": "queued", "content": f"Job {task_id} queued."}

        await self._run_assembly(job)
        return {"status": "in_progress", "content": f"Started job {task_id}."}

    async def _run_assembly(self, job: Dict[str, Any]):
        """Simulate performing assembly."""
        self._busy = True
        product = job["product_name"]
        parts   = job["part_name_list"]
        sender  = job.get("sender_jid")
        task_id = job.get("task_id")

        self.logger.info(f"[START] Assembling {product} on {self.agent_name}")
        for idx, part in enumerate(parts, start=1):
            self.logger.info(f"  • Working on {part} ({idx}/{len(parts)})")
            await asyncio.sleep(1.0)  # simulate action
        self.logger.info(f"[DONE] {product} assembly complete")

        if sender:
            await self._send_ack(to=sender, task_id=task_id, status="completed")

        self._busy = False
        if self._queue:
            next_job = self._queue.pop(0)
            await self._run_assembly(next_job)

    async def _send_ack(self, *, to: str, task_id: str | None, status: str):
        """Sends SPADE ack message."""
        msg = Message(to=to)
        msg.set_metadata("type", "ack")
        msg.body = json.dumps({"task_id": task_id, "status": status})
        await self.send(msg)
