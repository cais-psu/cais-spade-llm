"""Robot resource agent exposing pick/move/place primitives for assembly."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from agents.resource_agent.resource_agent import ResourceAgent
from xarmlib.wrapper import XArmAPI
import urx
from robot_UR_patch import apply_urx_patches
apply_urx_patches()
import math
import socket
import time



class RobotAgent(ResourceAgent):
    """Robot resource (UR5e, xArm, etc.) with granular motion tools."""

    agent_role = "robot"

    def __init__(self, jid: str, password: str, *, name: str, robotIP: str, robotType: str, ur5e_Params: dict = None, **kw: Any) -> None:
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
        self.robot_type = robotType
        self._held_part: Optional[str] = None
        self.arm = None
        self.params = {}

        if robotType == "xarm":
            self.setup_arm_xarm(ip = robotIP)

        if robotType == "ur5e":
            self.setup_arm_ur5e(ip = robotIP, ur5e_Params = ur5e_Params)

        self.logger.info(
            "RobotAgent '%s' initialized. tools=%s",
            name,
            list(self.executables.keys()),
        )

    def setup_arm_ur5e(self, ip: str, ur5e_Params: Optional[dict] = None):
        """Initialize connection to the UR5e robot arm hardware."""
        self.logger.info("[Robot] Setting up UR5e robot arm connection...")
        print("Setting up UR5e robot arm connection...")
        self.arm = urx.Robot(ip)
        self.params = ur5e_Params 
        time.sleep(1)
        print("UR5e robot arm setup complete.")
        print("Moving UR5e to home position...")
        asyncio.run(self._perform_action("Moving UR5e to home position", 
                             move={"x": -200, "y": -136, "z": -270, "roll": 0, "pitch": 0.0, "yaw": 0.0}, duration=10.0))
        pass

    def setup_arm_xarm(self, ip: str):
        """Initialize connection to the robot arm hardware."""

        self.logger.info("[Robot] Setting up robot arm connection...")
        print("Setting up robot arm connection...")

        self.arm = XArmAPI(ip, baud_checkset=False)

        self.params = {
            'grip_speed': 800,
            'radius': -1,
            'auto_enable': True,
            'wait': False,
            'speed': 20,
            'acc': 10000,
            'angle_speed': 20,
            'angle_acc': 500,
            'quit': False,
        }

        # Move the arm to the initial position
        print("Moving arm to initial position...")
        asyncio.run(self.move_home())

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_gripper_position(300)

        print("Robot arm setup complete.")

        pass

    async def move_to_pick_location(
        self,
        origin_resource_location: str,
        part_name: str,
        *,
        position: Optional[Dict[str, float]] = {"x": 250, "y": -150, "z": 400, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
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
          part_name:
            type: string
            description: Name of the part intended to be picked (for tracking).
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

        await self._perform_action(
            f"Travel empty to pick location {origin_resource_location} for {part_name} "
            f"(speed={speed or 'default'})", 
            move=position, duration=10.0
        )
        return {
            "status": "completed",
            "content": f"Arrived at {origin_resource_location} ready to pick {part_name}.",
        }

    async def pick_part(
        self,
        part_name: str,
        origin_resource_location: str,
        *,
        gripper: Optional[str] = None,
        position: Optional[Dict[str, float]] = None,
        speed: Optional[float] = None,
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

        if position is not None:
            await self._perform_action(
                f"Travel empty to pick point {origin_resource_location} for {part_name} "
                f"(speed={speed or 'default'})", 
                move=position, duration=10.0
            )

        await self._perform_gripper_action(
            f"Picking {part_name} from {origin_resource_location} "
            f"(gripper={gripper or 'default'})", openFactor=200
        )
        self._held_part = part_name
        return {"status": "completed", "content": f"Picked {part_name}."}

    async def move_loaded_to_destination(
        self,
        destination_location: str,
        part_name: str,
        *,
        position: Optional[Dict[str, float]] = {"x": 250, "y": -150, "z": 400, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
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
          part_name:
            type: string
            description: Name of the part being moved.
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
        
        # Consistency check: Ensure we are moving the part we think we are moving
        if part_name and self._held_part != part_name:
            self.logger.warning(
                "[Robot] Requested to move '%s' but currently holding '%s'. Proceeding with held part.",
                part_name, self._held_part
            )

        await self._perform_action_action(
            f"Move loaded part {self._held_part} to {destination_location} "
            f"(speed={speed or 'default'})",
            move=position, duration=10.0
        )
        return {
            "status": "completed",
            "content": f"Reached {destination_location} with {self._held_part}.",
        }

    async def place_part(
        self,
        destination_location: str,
        part_name: str,
        *,
        orientation: Optional[str] = None,
        position: Optional[Dict[str, float]] = None,
        speed: Optional[float] = None,
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
          part_name:
            type: string
            description: Name of the part being placed.
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

        # Consistency check
        if part_name and self._held_part != part_name:
            self.logger.warning(
                "[Robot] Requested to place '%s' but currently holding '%s'. Placing held part.",
                part_name, self._held_part
            )

        if position is not None:
            await self._perform_action(
                f"Travel full to place point {destination_location} for {part_name} "
                f"(speed={speed or 'default'})", 
                move=position, duration=10.0
            )

        await self._perform_gripper_action(
            f"Placing {self._held_part} at {destination_location} "
            f"(orientation={orientation or 'default'})", openFactor=300
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

        in_state: any
        out_state: any

        context: []

        params:
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Robot arm move to its home position.
        ---
        """
        
        if self._held_part:
            msg = "Cannot move home while still holding a part; place it first."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}
        

        await self._perform_action("Moving arm to home position", 
                                   move={"x": 250, "y": 0, "z": 450, "roll": 180.0, "pitch": 0.0, "yaw": 0.0}, duration=10.0)
        return {"status": "completed", "content": "At home position."}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    async def _simulate_action(self, description: str, *, duration: float = 5.0):
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
            print(f"[{robot}] ... {description} ({elapsed:.1f} / {duration:.1f} sec)")
            self.logger.info(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        self.logger.info("[%s] Finished: %s", robot, description)

    async def _perform_gripper_action(self, description: str, *, duration: float = 5.0, openFactor:int = 300):
        """
        Simulate a gripper action (pick/place) while printing progress every 5 seconds,
        including robot name for clarity when multiple robots run in parallel.
        """
        robot = self.agent_name

        self.logger.info("[%s] %s (estimated %.1f sec)", robot, description, duration)

        # Start the gripper action
        self.arm.set_gripper_position(openFactor, wait=False)

        interval = 1.0   # print every 5 seconds
        elapsed = 0.0

        while elapsed < duration:
            await asyncio.sleep(interval)
            elapsed += interval
            print(f"[{robot}] ... {description} ({elapsed:.1f} / {duration:.1f} sec)")
            self.logger.info(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        self.logger.info("[%s] Finished: %s", robot, description)

    async def _perform_action(self, description: str, *, move:Dict[str, float], duration: float = 5.0, speed: Optional[float] = None, other_params: Optional[Dict[str, Any]] = None):
        """
        Simulate a long-running robot action while printing progress every 5 seconds,
        including robot name for clarity when multiple robots run in parallel.
        """
        robot = self.agent_name
        
        self.logger.info("[%s] %s (estimated %.1f sec)", robot, description, duration)

        if self.robot_type == "xarm":
            useSpeed = self.params['speed']
            if speed is not None:
                useSpeed = speed

            ##start the actual robot action here using the move and other_params
            self.arm.set_position(move['x'], move['y'], move['z'], move['roll'], move['pitch'], move['yaw'],
                                    speed= useSpeed, wait= False)
        elif self.robot_type == "ur5e":
            coords_mm_deg = list(move.values())
            pos_m = [c / 1000.0 for c in coords_mm_deg[:3]]
            orient_rad = [math.radians(a) for a in coords_mm_deg[3:]]
            pose = pos_m + orient_rad
            print("POSE TYPE:", type(pose))
            print("POSE VALUE:", pose)
            self.arm.movel(pose, wait=False)

        interval = 1.0  
        elapsed = 0.0

        def check_is_moving():
            if self.robot_type == "xarm":
                return self.arm.get_is_moving()
            elif self.robot_type == "ur5e":
                return self.arm.is_program_running()  

        while check_is_moving():
            await asyncio.sleep(interval)
            elapsed += interval
            print(f"[{robot}] ... {description} ({elapsed:.1f} sec elapsed)")
            self.logger.info(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        ##Track the overall time it takes to perform the action and print progress every 5 seconds

        self.logger.info("[%s] Finished: %s", robot, description)

    def gripper_set(self, width, force):
        tool_index = 0
        body = f"""
        local rg = rpc_factory("xmlrpc","http://localhost:41414")
        local ret = rg.rg_grip({tool_index}, {float(width)}, {float(force)})
        textmsg("rg_grip returned: ", ret)
        """
        self._rtde_c.sendCustomScriptFunction("rg2_cmd", body)
