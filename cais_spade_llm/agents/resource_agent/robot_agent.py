"""Robot resource agent exposing pick/move/place primitives for assembly."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

from agents.resource_agent.resource_agent import ResourceAgent
from resources.robot import UR5eController, XArm6Controller


class RobotAgent(ResourceAgent):
    """Robot resource (UR5e, xArm, etc.) with granular motion tools.

    Execution behavior is selected by `execution_mode`:
    - `dry_run`: keep pure asyncio simulation via `_simulate_action`
    - `simulation`: call ROS2 controller phases (Gazebo)
    - `physical`: call ROS2 controller phases against real hardware stack
    """

    agent_role = "robot"
    _DEFAULT_PREWARM_TIMEOUT_S = 60.0

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

        # Controller config from the environment-specific robot JSON block.
        controller_config = kw.pop("controller_config", {})
        self.controller_config = controller_config
        self.named_positions = kw.pop("named_positions", {}) or {}
        self.motion_config = controller_config.get("motion", {})
        self.parts_tuning = controller_config.get("parts_tuning", {})
        execution_mode = str(kw.pop("execution_mode", "dry_run")).strip().lower()
        if execution_mode not in {"dry_run", "simulation", "physical"}:
            execution_mode = "dry_run"
        self.execution_mode = execution_mode
        self.controller_prewarm_timeout_s = float(
            kw.pop("controller_prewarm_timeout_s", self._DEFAULT_PREWARM_TIMEOUT_S)
        )
        self.enable_controller_prewarm = str(
            kw.pop(
                "enable_controller_prewarm",
                os.environ.get("ENABLE_ROBOT_AGENT_PREWARM", "0"),
            )
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Pop before super().__init__ to avoid unexpected kwarg error.
        self._injected_controller = kw.pop("prewarmed_controller", None)

        kw.setdefault(
            "function_names",
            [
                "pick_approach",
                "pick_grasp",
                "place_approach",
                "move_home",
                "place_insert",
            ],
        )
        super().__init__(jid, password, name=name, **kw)

        self.agent_name = name
        self._held_part: Optional[str] = None

        # Runtime state tracking for replanning context
        self._current_state: str = "idle"  # idle, at_pick, picked, positioned, placed
        self._position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}  # Simulated position
        self._gripper_state: str = "open"  # open, closed
        # Use pre-initialized controller (from Gazebo prewarm) if available,
        # to avoid paying the ROS2 init cost again on first task.
        if self._injected_controller is not None:
            self._controller = self._injected_controller
            self._controller_prewarm_done = True
            self.logger.info("[Robot] Using prewarmed controller for %s", name)
        else:
            self._controller = self._build_controller()
            self._controller_prewarm_done = self.execution_mode == "dry_run"
        self._injected_controller = None  # release reference
        self._controller_prewarm_attempted = False
        self._controller_prewarm_lock = asyncio.Lock()
        self._controller_prewarm_task: asyncio.Task | None = None

        self.logger.info(
            (
                "RobotAgent '%s' initialized. mode=%s tools=%s "
                "sg_slippage_mode=%s sg_slippage_scope=%s"
            ),
            name,
            self.execution_mode,
            list(self.executables.keys()),
            self.sg_slippage_mode,
            self.sg_slippage_scope,
        )

    async def teardown(self) -> None:
        """Clean up controller and prewarm task when the agent stops."""
        # Cancel any in-progress prewarm task.
        if self._controller_prewarm_task is not None and not self._controller_prewarm_task.done():
            self._controller_prewarm_task.cancel()
            try:
                await self._controller_prewarm_task
            except (asyncio.CancelledError, Exception):
                pass
            self._controller_prewarm_task = None

        # Shut down the ROS2 controller (kills spin thread, destroys node).
        if self._controller is not None:
            try:
                self._controller.shutdown()
                self.logger.info("[Robot] Controller shutdown complete for %s", self.agent_name)
            except Exception:
                self.logger.exception("[Robot] Controller shutdown failed for %s", self.agent_name)
            self._controller = None

    async def setup(self) -> None:
        await super().setup()
        # Run prewarm in background so startup/ready signal is not blocked.
        if (
            self.enable_controller_prewarm
            and
            self.execution_mode != "dry_run"
            and self._controller is not None
            and not self._controller_prewarm_done
        ):
            if self._controller_prewarm_task is None or self._controller_prewarm_task.done():
                self.logger.info(
                    "[Robot] Controller prewarm queued in background for %s",
                    self.agent_name,
                )
                self._controller_prewarm_task = asyncio.create_task(
                    self._ensure_controller_prewarmed()
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

    def _build_controller(self):
        """
        Build the low-level robot controller when execution_mode requires hardware/ROS2.

        dry_run mode intentionally keeps controller as None and uses _simulate_action.
        """
        if self.execution_mode == "dry_run":
            return None

        robot_scope = self._robot_scope_name()
        try:
            if robot_scope.startswith("ur5e"):
                return UR5eController(
                    controller_config=self.controller_config,
                    named_positions=self.named_positions,
                    execution_mode=self.execution_mode,
                )
            if robot_scope.startswith("xarm6"):
                return XArm6Controller(
                    controller_config=self.controller_config,
                    named_positions=self.named_positions,
                    execution_mode=self.execution_mode,
                )

            self.logger.error(
                "[Robot] Unknown robot '%s' for controller selection; "
                "falling back to dry_run mode.",
                self.agent_name,
            )
            self.execution_mode = "dry_run"
            return None
        except Exception as exc:
            self.logger.exception("[Robot] Failed to build controller: %s", exc)
            self.execution_mode = "dry_run"
            return None

    async def _ensure_controller_prewarmed(self) -> None:
        if not self.enable_controller_prewarm:
            return
        if self.execution_mode == "dry_run" or self._controller is None:
            self._controller_prewarm_done = True
            return
        if self._controller_prewarm_done:
            return

        async with self._controller_prewarm_lock:
            if self._controller_prewarm_done:
                return
            if self._controller_prewarm_attempted:
                return
            self._controller_prewarm_attempted = True
            ok, elapsed = await self._wait_for_services(self._controller)
            if ok:
                self._controller_prewarm_done = True
                self.logger.info(
                    "[Robot] Controller prewarm ready for %s in %.2fs",
                    self.agent_name,
                    elapsed,
                )
            else:
                self.logger.warning(
                    "[Robot] Controller prewarm failed for %s; will defer to first task.",
                    self.agent_name,
                )

    async def _wait_for_services(self, controller: Any) -> tuple[bool, float]:
        start = time.monotonic()
        try:
            ok = await asyncio.to_thread(
                controller.wait_for_services,
                self.controller_prewarm_timeout_s,
            )
        except Exception as exc:
            self.logger.exception("[Robot] Controller prewarm exception: %s", exc)
            return False, time.monotonic() - start
        return bool(ok), time.monotonic() - start

    async def _run_phase_or_simulate(
        self,
        *,
        phase_name: str,
        simulate_description: str,
        simulate_duration: float = 5.0,
        **phase_kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a controller phase in simulation/physical mode, otherwise use dry_run."""
        if self.execution_mode == "dry_run":
            await self._simulate_action(simulate_description, duration=simulate_duration)
            return {"success": True, "message": f"Simulated: {simulate_description}"}

        await self._ensure_controller_prewarmed()

        if self._controller is None:
            return {"success": False, "message": "controller is not initialized"}

        method = getattr(self._controller, phase_name, None)
        if not callable(method):
            return {
                "success": False,
                "message": f"controller missing phase method '{phase_name}'",
            }

        try:
            result = await asyncio.to_thread(method, **phase_kwargs)
            if isinstance(result, dict):
                return result
            if isinstance(result, bool):
                return {
                    "success": result,
                    "message": f"{phase_name} {'ok' if result else 'failed'}",
                }
            return {"success": False, "message": f"{phase_name} returned invalid result"}
        except Exception as exc:
            self.logger.exception("[Robot] %s execution failed", phase_name)
            return {"success": False, "message": f"{phase_name} exception: {type(exc).__name__}"}

    @staticmethod
    def _normalize_phase_result(
        phase_result: Dict[str, Any],
        *,
        on_success: str,
        on_failure: str,
    ) -> tuple[bool, Dict[str, Any]]:
        """Convert controller/sim result to ResourceAgent ACK payload shape."""
        ok = bool((phase_result or {}).get("success"))
        msg = str((phase_result or {}).get("message") or (on_success if ok else on_failure))
        payload: Dict[str, Any] = {
            "status": "completed" if ok else "failed",
            "content": msg,
        }
        if isinstance((phase_result or {}).get("observations"), dict):
            payload["observations"] = dict(phase_result["observations"])
        return ok, payload

    async def pick_approach(
        self,
        origin_resource_location: str,
        part_name: str,
        *,
        speed: Optional[float] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
        product_geometry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: idle
        out_state: at_pick

        required_context_keys: [origin]
        context_mapping:
          location_param: origin_resource_location
          location_type: part_location

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
          product_geometry:
            type: object
            description: Product geometry payload containing part poses in world frame.
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Approach the part's origin location with empty gripper.
        ---
        """

        if self._held_part:
            msg = "Cannot move-to-pick while already holding a part."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        phase = await self._run_phase_or_simulate(
            phase_name="pick_approach",
            simulate_description=(
                f"Travel empty to pick location {origin_resource_location} for {part_name} "
                f"(speed={speed or 'default'})"
            ),
            origin_resource_location=origin_resource_location,
            part_name=part_name,
            product_geometry=product_geometry,
            speed=speed,
        )
        ok, payload = self._normalize_phase_result(
            phase,
            on_success=f"Arrived at {origin_resource_location} ready to pick {part_name}.",
            on_failure=(
                f"Failed to approach pick location {origin_resource_location} "
                f"for {part_name}."
            ),
        )
        if not ok:
            return payload

        self._current_state = "at_pick"
        # Simulated position update (in real system, would query robot controller)
        self._position = {"x": 0.0, "y": 0.0, "z": 300.0}
        return payload

    async def pick_grasp(
        self,
        part_name: str,
        origin_resource_location: str,
        *,
        gripper: Optional[str] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
        product_geometry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: at_pick
        out_state: picked
        part_in_state: ready

        required_context_keys: [origin]
        context_mapping:
          location_param: origin_resource_location
          location_type: current_location

        part_transition:
          completed:
            state: in_gripper
            location_template: "{resource_jid}_gripper"

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
          product_geometry:
            type: object
            description: Product geometry payload containing part poses in world frame.
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Pick a ready part from an origin location.
        ---
        """

        if self._held_part:
            msg = f"Already holding {self._held_part}; assemble it before picking a new part."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        phase = await self._run_phase_or_simulate(
            phase_name="pick_grasp",
            simulate_description=(
                f"Picking {part_name} from {origin_resource_location} "
                f"(gripper={gripper or 'default'})"
            ),
            part_name=part_name,
            origin_resource_location=origin_resource_location,
            product_geometry=product_geometry,
            gripper=gripper,
        )
        ok, payload = self._normalize_phase_result(
            phase,
            on_success=f"Picked {part_name}.",
            on_failure=f"Failed to pick {part_name} from {origin_resource_location}.",
        )
        if not ok:
            return payload

        self._held_part = part_name
        self._current_state = "picked"
        self._gripper_state = "closed"
        return payload

    async def place_approach(
        self,
        destination_location: str,
        part_name: str,
        *,
        speed: Optional[float] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
        product_geometry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: picked
        out_state: positioned
        part_in_state: in_gripper

        required_context_keys: [destination]
        context_mapping:
          location_param: destination_location
          location_type: reachable_location

        part_transition:
          completed:
            state: in_transit
            location_template: "{resource_jid}_gripper"

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
          product_geometry:
            type: object
            description: Product geometry payload containing target placement poses.
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

        phase = await self._run_phase_or_simulate(
            phase_name="place_approach",
            simulate_description=(
                f"Move loaded part {self._held_part} to {destination_location} "
                f"(speed={speed or 'default'})"
            ),
            destination_location=destination_location,
            part_name=part_name or self._held_part,
            product_geometry=product_geometry,
            speed=speed,
        )
        ok, payload = self._normalize_phase_result(
            phase,
            on_success=f"Reached {destination_location} with {self._held_part}.",
            on_failure=f"Failed to move loaded part to {destination_location}.",
        )
        if not ok:
            return payload

        self._current_state = "positioned"
        # Simulated position update
        self._position = {"x": 400.0, "y": -200.0, "z": 200.0}
        return payload

    async def place_insert(
        self,
        destination_location: str,
        part_name: str,
        *,
        orientation: Optional[str] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
        product_geometry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ---
        process: assembly
        resource_type: robot

        in_state: positioned
        out_state: idle
        part_in_state: in_transit

        required_context_keys: [destination]
        context_mapping:
          location_param: destination_location
          location_type: current_location

        part_transition:
          completed:
            state: assembled
            verify_camera: true
            location_param: destination_location

        params:
          destination_location:
            type: string
            description: Final assembly location for the part.
          part_name:
            type: string
            description: Name of the part being assembled.
          orientation:
            type: string
            description: Optional placement orientation.
          product_geometry:
            type: object
            description: Product geometry payload containing insertion target pose.
          product_jid:
            type: string
            description: JID of the ProductAgent that owns this task.
          task_id:
            type: string

        description: Assemble the currently held part at its final destination.
        ---
        """

        if not self._held_part:
            msg = "No part currently held; run pick_grasp first."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        placed_target = part_name or self._held_part
        if self._should_inject_sg_slippage(placed_target):
            self.logger.error("[Robot] Assembly verification failed for %s.", placed_target)

            self._held_part = None
            self._current_state = "recovery_required"
            self._gripper_state = "open"

            return {
                "status": "failed",
                "content": "Assembly verification failed.",
                "observations": {
                    "gripper_force": 0.0,
                    "camera_detection": {
                        "object_found": True,
                        "zone": f"{self._robot_scope_name()}_workspace",
                        "shape_match_confidence": 0.85,
                        "orientation": "upright",
                        "visible_damage": False,
                    },
                    "last_commanded_location": destination_location,
                },
            }

        phase = await self._run_phase_or_simulate(
            phase_name="place_insert",
            simulate_description=(
                f"Assembling {self._held_part} at {destination_location} "
                f"(orientation={orientation or 'default'})"
            ),
            destination_location=destination_location,
            part_name=part_name or self._held_part,
            product_geometry=product_geometry,
            orientation=orientation,
        )
        ok, payload = self._normalize_phase_result(
            phase,
            on_success=f"Assembled {self._held_part} at {destination_location}.",
            on_failure=f"Failed to assemble {self._held_part} at {destination_location}.",
        )
        if not ok:
            self._current_state = "recovery_required"
            return payload

        placed = self._held_part
        self._held_part = None
        self._current_state = "idle"
        self._gripper_state = "open"
        payload["placed_location"] = destination_location
        if not payload.get("content"):
            payload["content"] = f"Assembled {placed} at {destination_location}."
        return payload

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
            msg = "Cannot move home while still holding a part; assemble it first."
            self.logger.warning("[Robot] %s", msg)
            return {"status": "blocked", "content": msg}

        phase = await self._run_phase_or_simulate(
            phase_name="move_home",
            simulate_description="Moving arm to home position",
            simulate_duration=3.0,
        )
        ok, payload = self._normalize_phase_result(
            phase,
            on_success="At home position.",
            on_failure="Failed to move to home position.",
        )
        if not ok:
            return payload

        self._current_state = "idle"
        self._position = {"x": 0.0, "y": 0.0, "z": 445.0}  # Home position
        return payload

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _snapshot_state(self) -> Dict[str, Any]:
        """Robot-specific state snapshot (override)."""
        return {
            "execution_mode": self.execution_mode,
            "controller_ready": self._controller is not None if self.execution_mode != "dry_run" else True,
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

        interval = 2.0   # print every 5 seconds
        elapsed = 0.0

        while elapsed < duration:
            await asyncio.sleep(interval)
            elapsed += interval
            self.logger.debug(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        self.logger.info("[%s] Finished: %s", robot, description)
