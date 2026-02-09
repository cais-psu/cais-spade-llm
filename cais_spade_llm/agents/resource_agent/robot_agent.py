"""Robot resource agent exposing pick/move/place primitives for assembly."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from agents.resource_agent.resource_agent import ResourceAgent


class RobotAgent(ResourceAgent):
    """Robot resource (UR5e, xArm, etc.) with granular motion tools."""

    agent_role = "robot"

    def __init__(self, jid: str, password: str, *, name: str, **kw: Any) -> None:
        # Failure-injection controls for SG placement tests.
        # - always: fail every qualifying SG place
        # - once: fail first qualifying SG place, then allow
        # - off: never inject SG slippage failures
        sg_slippage_mode = str(kw.pop("sg_slippage_mode", "always")).lower()
        if sg_slippage_mode not in {"always", "once", "off"}:
            sg_slippage_mode = "always"
        self.sg_slippage_mode = sg_slippage_mode
        # Scope to one robot by name ("xarm6"), or "any".
        self.sg_slippage_scope = str(kw.pop("sg_slippage_scope", "xarm6")).lower()
        self._sg_slippage_triggered = False

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

        # Runtime state tracking for replanning context
        self._current_state: str = "idle"  # idle, at_pick, picked, positioned, placed
        self._position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}  # Simulated position
        self._gripper_state: str = "open"  # open, closed

        self.logger.info(
            "RobotAgent '%s' initialized. tools=%s sg_slippage_mode=%s sg_slippage_scope=%s",
            name,
            list(self.executables.keys()),
            self.sg_slippage_mode,
            self.sg_slippage_scope,
        )

    def _robot_scope_name(self) -> str:
        """Lower-cased stable robot identifier used for scoped fault injection."""
        return str(self.agent_name or "").split("@", 1)[0].lower()

    def _jid_domain(self) -> str:
        """Best-effort XMPP domain from this robot JID."""
        jid_text = str(getattr(self, "jid", "") or "")
        if "@" in jid_text:
            return jid_text.split("@", 1)[1]
        return "localhost"

    def _agent_ref_to_jid(self, agent_ref: str) -> str:
        """Normalize agent references from config into full JIDs."""
        ref = str(agent_ref or "").strip()
        if not ref:
            return ""
        if "@" in ref:
            return ref
        return f"{ref}@{self._jid_domain()}"

    def _select_slippage_drop_site(self) -> Dict[str, Any]:
        """
        Pick a data-driven drop site from static capabilities.
        Prefer shared staging sites (accessible by multiple robots).
        """
        caps = self.static_capabilities or {}
        staging_areas = caps.get("staging_areas") if isinstance(caps, dict) else {}
        if not isinstance(staging_areas, dict):
            staging_areas = {}

        my_name = self._robot_scope_name()
        shared_candidates: list[tuple[str, Dict[str, Any]]] = []
        fallback_candidates: list[tuple[str, Dict[str, Any]]] = []

        for site_name, raw_site in staging_areas.items():
            if not isinstance(raw_site, dict):
                continue
            site = dict(raw_site)
            accessible_by_raw = site.get("accessible_by")
            accessible_by = [
                str(x).strip().lower()
                for x in (accessible_by_raw or [])
                if str(x).strip()
            ]
            site["accessible_by"] = accessible_by
            fallback_candidates.append((str(site_name), site))
            if len(set(accessible_by)) >= 2 or any(a != my_name for a in accessible_by):
                shared_candidates.append((str(site_name), site))

        if shared_candidates:
            site_name, site = shared_candidates[0]
        elif fallback_candidates:
            site_name, site = fallback_candidates[0]
        else:
            # Last-resort defaults are derived from current runtime state.
            return {
                "site_name": "unknown_site",
                "drop_region": "unknown_region",
                "drop_location": "unknown_location",
                "position": dict(self._position),
                "recoverable_by": [str(self.jid)],
            }

        recoverable_by = [
            self._agent_ref_to_jid(name) for name in site.get("accessible_by", [])
        ]
        recoverable_by = [jid for jid in recoverable_by if jid] or [str(self.jid)]

        x = site.get("x")
        y = site.get("y")
        z = site.get("z")
        if all(isinstance(v, (int, float)) for v in (x, y, z)):
            position = {"x": float(x), "y": float(y), "z": float(z)}
        else:
            position = dict(self._position)

        return {
            "site_name": site_name,
            "drop_region": f"{site_name}_region",
            "drop_location": site_name,
            "position": position,
            "recoverable_by": recoverable_by,
        }

    def _should_inject_sg_slippage(self, target_part_name: str) -> bool:
        """Return True if SG slippage should be injected for this placement."""
        if str(target_part_name) != "SG":
            return False
        if self.sg_slippage_mode == "off":
            return False

        if self.sg_slippage_scope not in ("any", self._robot_scope_name()):
            return False

        if self.sg_slippage_mode == "once" and self._sg_slippage_triggered:
            return False

        self._sg_slippage_triggered = True
        return True

    def _build_generic_failure_context(
        self,
        *,
        failure_mode: str,
        affected_entities: Optional[list[dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build a generic failure context payload usable across modes
        (e.g. slippage, breakdown, timeout).
        """
        context: Dict[str, Any] = {
            "failure_class": "execution_failure",
            "failure_mode": str(failure_mode),
        }
        if affected_entities:
            context["affected_entities"] = affected_entities
        return context

    async def move_to_pick_location(
        self,
        origin_resource_location: str,
        part_name: str,
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

        await self._simulate_action(
            f"Travel empty to pick location {origin_resource_location} for {part_name} "
            f"(speed={speed or 'default'})"
        )
        self._current_state = "at_pick"
        # Simulated position update (in real system, would query robot controller)
        self._position = {"x": 0.0, "y": 0.0, "z": 300.0}
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
        self._current_state = "picked"
        self._gripper_state = "closed"
        return {"status": "completed", "content": f"Picked {part_name}."}

    async def move_loaded_to_destination(
        self,
        destination_location: str,
        part_name: str,
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

        await self._simulate_action(
            f"Move loaded part {self._held_part} to {destination_location} "
            f"(speed={speed or 'default'})"
        )
        self._current_state = "positioned"
        # Simulated position update
        self._position = {"x": 400.0, "y": -200.0, "z": 200.0}
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

        # Simulate slippage for SG placement (configurable test injection).
        placed_target = part_name or self._held_part
        if self._should_inject_sg_slippage(placed_target):
            msg = "Simulated slippage: SG failed to seat during placement."
            self.logger.error("[Robot] %s", msg)
            drop_site = self._select_slippage_drop_site()

            # Part is no longer held after slippage.
            self._held_part = None
            self._current_state = "recovery_required"
            self._gripper_state = "open"
            self._position = dict(drop_site.get("position") or self._position)

            affected = []
            if placed_target:
                affected.append(
                    {
                        "entity_type": "part",
                        "entity_id": str(placed_target),
                        "state": "unplaced",
                    }
                )

            return {
                "status": "failed:slippage",
                "content": msg,
                "failure_context": self._build_generic_failure_context(
                    failure_mode="slippage",
                    affected_entities=affected,
                ),
            }

        await self._simulate_action(
            f"Placing {self._held_part} at {destination_location} "
            f"(orientation={orientation or 'default'})"
        )
        placed = self._held_part
        self._held_part = None
        self._current_state = "placed"
        self._gripper_state = "open"
        return {
            "status": "completed",
            "content": f"Placed {placed}.",
            "placed_location": destination_location,  # For part tracking
        }

    '''
    async def place_part(
        self,
        destination_location: str,
        part_name: str,
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

        # Simulate slippage for MCP placement (hard-coded).
        # Keep holding the part to reflect a failed place action.
        if (part_name or self._held_part) == "MCP":
            msg = "Simulated slippage: MCP failed to seat during placement."
            self.logger.error("[Robot] %s", msg)
            # State remains "positioned" with gripper still closed
            return {"status": "failed:slippage", "content": msg}

        await self._simulate_action(
            f"Placing {self._held_part} at {destination_location} "
            f"(orientation={orientation or 'default'})"
        )
        placed = self._held_part
        self._held_part = None
        self._current_state = "placed"
        self._gripper_state = "open"
        return {
            "status": "completed",
            "content": f"Placed {placed}.",
            "placed_location": destination_location,  # For part tracking
        }
    '''
    
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

        await self._simulate_action("Moving arm to home position")
        self._current_state = "idle"
        self._position = {"x": 0.0, "y": 0.0, "z": 445.0}  # Home position
        return {"status": "completed", "content": "At home position."}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _snapshot_state(self) -> Dict[str, Any]:
        """Robot-specific state snapshot (override)."""
        return {
            "held_part": self._held_part,
            "current_state": self._current_state,
            "position": self._position.copy(),
            "gripper_state": self._gripper_state,
        }

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
            self.logger.info(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        self.logger.info("[%s] Finished: %s", robot, description)
