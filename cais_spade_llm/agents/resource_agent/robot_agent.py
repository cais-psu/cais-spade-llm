# agents/resource_agent/robot_agent.py
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from spade.message import Message

from agents.resource_agent.resource_agent import ResourceAgent


class RobotAgent(ResourceAgent):
    """
    Robot resource (e.g., UR5e, xArm) capable of running an assembly sequence.
    Inherits LLM tool-broker plumbing from ResourceAgent; exposes `assembly` as a tool.
    """

    agent_role = "robot"

    def __init__(self, jid: str, password: str, *, name: str, **kw) -> None:
        # Expose the 'assembly' function to the LLM
        kw.setdefault("function_names", ["assembly"])
        super().__init__(jid, password, name=name, **kw)

        self._busy: bool = False
        self._queue: list[dict] = []
        self.agent_name: str = name

        # Defensive registration in case LlmAgent doesn't auto-bind by name
        if getattr(self, "executables", None) is not None:
            self.executables.setdefault("assembly", self.assembly)  # type: ignore[attr-defined]

        self.logger.info(f"RobotAgent '{name}' initialized. tools={list(self.executables.keys())}")

    # ------------------------------------------------------------------ #
    # Robot-specific tool – exposed to LLM (function_call) or RA inbox
    # ------------------------------------------------------------------ #
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
        return {"status": "started", "content": f"Started job {task_id}."}

    # ------------------------------------------------------------------ #
    # Internal execution & ACKs
    # ------------------------------------------------------------------ #
    async def _run_assembly(self, job: Dict[str, Any]) -> None:
        """Simulate performing assembly of all parts in `job`."""
        self._busy = True

        product = job["product_name"]
        parts = job["part_name_list"]
        sender = job.get("sender_jid")
        task_id = job.get("task_id")

        self.logger.info(f"[START] Assembling {product} on {self.agent_name} from {job['origin_resource_location']}")
        try:
            # Simple simulation of a pick→place loop per part
            for idx, part in enumerate(parts, start=1):
                self.logger.info(f"  • ({idx}/{len(parts)}) move→pick→move→place: {part}")
                await asyncio.sleep(1.0)  # simulate action time
            self.logger.info(f"[DONE] {product} assembly complete")
            if sender:
                await self._send_ack(to=sender, task_id=task_id, status="completed")
        except Exception as e:
            self.logger.exception("[Robot] Assembly failed")
            if sender:
                await self._send_ack(to=sender, task_id=task_id, status=f"failed:{type(e).__name__}")
        finally:
            self._busy = False
            # Drain queue if any
            if self._queue:
                next_job = self._queue.pop(0)
                await self._run_assembly(next_job)

    async def _send_ack(self, *, to: str, task_id: Optional[str], status: str) -> None:
        """Sends a SPADE ACK message back to the requester."""
        msg = Message(to=to)
        msg.set_metadata("type", "ack")
        msg.body = json.dumps({"task_id": task_id, "status": status})
        await self.send(msg)
