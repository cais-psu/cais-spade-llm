"""Robot resource agent exposing pick/move/place primitives for assembly."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from agents.resource_agent.resource_agent import ResourceAgent


class RobotAgent(ResourceAgent):
    """Robot resource (UR5e, xArm, etc.) with granular motion tools."""

    agent_role = "robot"

    def __init__(self, jid: str, password: str, *, name: str, **kw: Any) -> None:
        kw.setdefault(
            "function_names",
            [
                "pick_part",
                "move_to_pick_location",
                "move_loaded_to_destination",
                "move_home",
                "place_part",
            ],
        )
        super().__init__(jid, password, name=name, **kw)

        self.agent_name = name
        self._held_part: Optional[str] = None
        self.logger.info(
            "RobotAgent '%s' initialized. tools=%s",
            name,
            list(self.executables.keys()),
        )

    async def pick_part(
        self,
        part_name: str,
        origin_resource_location: str,
        *,
        gripper: Optional[str] = None,
        sender_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        phase:           assembly
        in_state:        printed
        out_state:       picked
        params:
          part_name:               string
          origin_resource_location:string
          gripper:                string
          sender_jid:             string
          task_id:                string
        description: Move to origin_resource_location and pick the specified part using the requested gripper.
        ---
        """
        if self._held_part:
            msg = f"Already holding {self._held_part}; place it before picking a new part."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        await self._simulate_action(
            f"Picking {part_name} from {origin_resource_location} "
            f"(gripper={gripper or 'default'})"
        )
        self._held_part = part_name
        return {"status": "completed", "content": f"Picked {part_name}."}

    async def move_to_pick_location(
        self,
        origin_resource_location: str,
        *,
        speed: Optional[float] = None,
        sender_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        phase:           assembly
        in_state:        idle
        out_state:       at_pick
        params:
          origin_resource_location:string
          speed:                   float
          sender_jid:              string
          task_id:                 string
        description: Move empty gripper to the origin location in preparation for picking.
        ---
        """
        if self._held_part:
            msg = "Cannot move-to-pick while already holding a part."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        await self._simulate_action(
            f"Travel empty to pick location {origin_resource_location} "
            f"(speed={speed or 'default'})"
        )
        return {
            "status": "completed",
            "content": f"Arrived at {origin_resource_location} ready to pick.",
        }

    async def move_loaded_to_destination(
        self,
        destination_location: str,
        *,
        speed: Optional[float] = None,
        sender_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        phase:           assembly
        in_state:        picked
        out_state:       positioned
        params:
          destination_location:    string
          speed:                   float
          sender_jid:              string
          task_id:                 string
        description: Move while carrying the picked part to its destination pose.
        ---
        """
        if not self._held_part:
            msg = "Cannot move-loaded without holding a part."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        await self._simulate_action(
            f"Move loaded part {self._held_part} to {destination_location} "
            f"(speed={speed or 'default'})"
        )
        return {
            "status": "completed",
            "content": f"Reached {destination_location} with {self._held_part}.",
        }

    async def move_home(
        self,
        *,
        sender_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        phase:           assembly
        in_state:        placed
        out_state:       idle
        params:
          sender_jid:              string
          task_id:                 string
        description: Return the robot arm to a predefined home position.
        ---
        """
        if self._held_part:
            msg = "Cannot move home while still holding a part; place it first."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        await self._simulate_action("Moving arm to home position")
        return {"status": "completed", "content": "At home position."}

    async def place_part(
        self,
        destination_location: str,
        *,
        orientation: Optional[str] = None,
        sender_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        phase:           assembly
        in_state:        positioned
        out_state:       placed
        params:
          destination_location:    string
          orientation:             string
          sender_jid:              string
          task_id:                 string
        description: Place the currently held part at the destination with the requested orientation.
        ---
        """
        if not self._held_part:
            msg = "No part currently held; run pick_part first."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        await self._simulate_action(
            f"Placing {self._held_part} at {destination_location} "
            f"(orientation={orientation or 'default'})"
        )
        placed = self._held_part
        self._held_part = None
        return {"status": "completed", "content": f"Placed {placed}."}

    async def _simulate_action(self, description: str, *, duration: float = 1.0):
        self.logger.info("[Robot] %s", description)
        await asyncio.sleep(duration)




    '''
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
        phase:           assembly
        in_state:        printed
        out_state:       assembled
        params:
          product_name:             string
          part_name_list:           list[string]
          origin_resource_location: string  # where parts are picked
          sender_jid:               string # (auto-filled) requester
          phase_id:                 string # (auto-filled)
          task_id:                  string # (auto-filled)
        description: Executes an assembly process for parts within a product using this robot agent.
        ---
        :param product_name: Name/ID of the product to assemble.
        :param part_name_list: List of parts to assemble in order.
        :param origin_resource_location: Identifier of the source resource (e.g., printer) for pickup.
        :param sender_jid: Product agent JID that requested the job.
        :param phase_id: Phase identifier inside a plan.
        :param task_id: Task identifier inside the phase.
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
        # IMPORTANT: let ResourceAgent send the final ACK using this return value
        return {"status": "completed", "content": f"Completed job {task_id}."}

    # ------------------------------------------------------------------ #
    # Internal execution (no messaging here)
    # ------------------------------------------------------------------ #
    async def _run_assembly(self, job: Dict[str, Any]) -> None:
        """Simulate performing assembly of all parts in `job`."""
        self._busy = True

        product = job["product_name"]
        parts = job["part_name_list"]

        self.logger.info(f"[START] Assembling {product} on {self.agent_name} from {job['origin_resource_location']}")
        try:
            # Simple simulation of a move→pick→move→place loop per part
            for idx, part in enumerate(parts, start=1):
                self.logger.info(f"  • ({idx}/{len(parts)}) move→pick→move→place: {part}")
                await asyncio.sleep(1.0)  # simulate action time
            self.logger.info(f"[DONE] {product} assembly complete")
        finally:
            self._busy = False
            # Drain queue if any
            if self._queue:
                next_job = self._queue.pop(0)
                await self._run_assembly(next_job)
    '''
