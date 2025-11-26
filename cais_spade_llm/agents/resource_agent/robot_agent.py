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


    async def move_to_pick_location(
        self,
        origin_resource_location: str,
        *,
        speed: Optional[float] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: idle
        out_state: at_pick

        required_context_keys: [origin]

        params:
          origin_resource_location:
            type: string
            description: Target origin location to approach for picking.
          speed:
            type: number
            description: Optional motion speed.
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Move empty gripper to the part's origin location.
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

    async def pick_part(
        self,
        part_name: str,
        origin_resource_location: str,
        *,
        gripper: Optional[str] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: printed
        out_state: picked

        required_context_keys: [origin]

        params:
          part_name:
            type: string
            description: Name of the part to pick.
          origin_resource_location:
            type: string
            description: Origin location of the part (printer or fixture).
          gripper:
            type: string
            description: Optional gripper configuration.
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Pick a printed part from an origin location.
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

    async def move_loaded_to_destination(
        self,
        destination_location: str,
        *,
        speed: Optional[float] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: picked
        out_state: positioned

        required_context_keys: [destination]

        params:
          destination_location:
            type: string
            description: Destination location to carry the loaded part.
          speed:
            type: number
            description: Optional motion speed while loaded.
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Move the loaded part to its destination location.
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

    async def place_part(
        self,
        destination_location: str,
        *,
        orientation: Optional[str] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: positioned
        out_state: placed

        required_context_keys: [destination]

        params:
          destination_location:
            type: string
            description: Target placement location.
          orientation:
            type: string
            description: Optional placement orientation.
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Place the currently held part at a destination.
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

    async def move_home(
        self,
        *,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: placed
        out_state: idle

        context: []

        params:
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Return robot arm to its home position.
        ---
        """

        if self._held_part:
            msg = "Cannot move home while still holding a part; place it first."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        await self._simulate_action("Moving arm to home position")
        return {"status": "completed", "content": "At home position."}

    async def _simulate_action(self, description: str, *, duration: float = 300.0):
        """
        Simulate a long-running robot action while printing progress every 5 seconds,
        including robot name for clarity when multiple robots run in parallel.
        """
        robot = self.agent_name

        self.logger.info("[%s] %s (estimated %.1f sec)", robot, description, duration)

        interval = 5.0   # print every 5 seconds
        elapsed = 0.0

        while elapsed < duration:
            await asyncio.sleep(interval)
            elapsed += interval
            self.logger.info(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        self.logger.info("[%s] Finished: %s", robot, description)
