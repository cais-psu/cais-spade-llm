# agents/resource_agent/robot_agent.py
from __future__ import annotations
import json, asyncio
from typing import Any, Dict, List, Optional

from spade.message import Message
from agents.resource_agent.resource_agent import ResourceAgent

class RobotAgent(ResourceAgent):
    """Robot-specific resource (UR5e, xArm) that executes assembly tasks."""
    agent_role = "robot"

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        annotation: Optional[str] = None,
        instructions: Optional[str] = None,
        function_names: Optional[List[str]] = None,   # include "assembly" if you want LLM to call it
        static_capabilities: Optional[Dict[str, Any]] = None,
        vector_dbs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            jid,
            password,
            name=name,
            annotation=annotation,
            instructions=instructions,
            function_names=function_names,
            static_capabilities=static_capabilities,
            vector_dbs=vector_dbs,
        )
        self._busy: bool = False
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
        part_name_list: List[str],
        origin_resource_location: str,
        sender_jid: Optional[str] = None,
        phase_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        verb:            [assembly]
        object_type:     [product]
        phase:           assembly
        in_state:        printed
        out_state:       assembled
        freeze_resource: false
        params:
          product_name:             string
          part_name_list:           list[string]
          origin_resource_location: string  # where parts are picked
          sender_jid:               string? # (auto-filled) requester
          phase_id:                 string? # (auto-filled)
          task_id:                  string? # (auto-filled)
        description: Executes an assembly process for parts within a product using this robot agent.
        ---
        :param product_name: Name/ID of the product to assemble.
        :param part_name_list: List of parts to assemble in order.
        :param origin_resource_location: Identifier of the source resource (e.g., printer) for pickup.
        :param sender_jid: Product agent JID that requested the job (optional).
        :param phase_id: Phase identifier inside a plan (optional).
        :param task_id: Task identifier inside the phase (optional).
        :returns: Dict with status and message about queueing or start.
        """
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

    async def _run_assembly(self, job: Dict[str, Any]) -> None:
        """Simulate performing assembly of all parts in `job`."""
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

    async def _send_ack(self, *, to: str, task_id: Optional[str], status: str) -> None:
        """Sends SPADE ack message."""
        msg = Message(to=to)
        msg.set_metadata("type", "ack")
        msg.body = json.dumps({"task_id": task_id, "status": status})
        await self.send(msg)
