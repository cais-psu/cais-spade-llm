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
          part_name:
            type: string
            description: Name of the part to pick up.
          origin_resource_location:
            type: string
            description: Resource or printer identifier the part is located at.
          gripper:
            type: string
            description: Optional gripper program or pose configuration to use.
          sender_jid:
            type: string
            description: Product agent JID that issued the request.
          task_id:
            type: string
            description: Planner task identifier supplied by the product agent.
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
          origin_resource_location:
            type: string
            description: Location identifier to approach for picking.
          speed:
            type: number
            description: Optional motion speed override.
          sender_jid:
            type: string
            description: Product agent JID that issued the request.
          task_id:
            type: string
            description: Planner task identifier supplied by the product agent.
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
          destination_location:
            type: string
            description: Target pose or waypoint for the carried part.
          speed:
            type: number
            description: Optional motion speed override while loaded.
          sender_jid:
            type: string
            description: Product agent JID that issued the request.
          task_id:
            type: string
            description: Planner task identifier supplied by the product agent.
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
          sender_jid:
            type: string
            description: Product agent JID that issued the request.
          task_id:
            type: string
            description: Planner task identifier supplied by the product agent.
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
          destination_location:
            type: string
            description: Target placement location identifier.
          orientation:
            type: string
            description: Optional placement orientation override.
          sender_jid:
            type: string
            description: Product agent JID that issued the request.
          task_id:
            type: string
            description: Planner task identifier supplied by the product agent.
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
