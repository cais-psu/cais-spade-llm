"""Robot resource agent exposing pick/move/place primitives for assembly."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.resources.robot import UR5eController, XArm6Controller

_UR5E_GAZEBO_ARM_TRAJECTORY_TOPIC = "/ur5e_joint_trajectory_controller/joint_trajectory"


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
        # Failure-injection controls for LG placement tests.
        # Accept legacy lcp_* keys as aliases so older fixtures still load.
        # - always: fail every qualifying LG place
        # - once: fail first qualifying LG place, then allow
        # - off: never inject LG slippage failures
        lg_slippage_mode = str(
            kw.pop("lg_slippage_mode", kw.pop("lcp_slippage_mode", "always"))
        ).lower()
        if lg_slippage_mode not in {"always", "once", "off"}:
            lg_slippage_mode = "always"
        self.lg_slippage_mode = lg_slippage_mode
        # Scope to one robot by name ("xarm6"), or "any".
        self.lg_slippage_scope = str(
            kw.pop("lg_slippage_scope", kw.pop("lcp_slippage_scope", "xarm6"))
        ).lower()
        self._lg_slippage_triggered = False

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

        # Register bridge-only method in executables for runtime dispatch,
        # but NOT in function_names / ALLOWED_FUNCS / tools.json.
        self.executables["execute_recovery_macro"] = self.execute_recovery_macro

        # Runtime state tracking for replanning context
        self._current_state: str = "idle"  # idle, at_pick, picked, positioned, placed (placed = at destination, part released)
        self._position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}  # Simulated position
        self._gripper_state: str = "open"
        self._bridge_pose_ref: Optional[str] = None
        # Phased context threading through pick→grasp→approach→insert.
        # Replaces the controller's _active_ctx; owned by the agent.
        self._pick_ctx: Dict[str, Any] = {}  # open, closed
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
                "lg_slippage_mode=%s lg_slippage_scope=%s"
            ),
            name,
            self.execution_mode,
            list(self.executables.keys()),
            self.lg_slippage_mode,
            self.lg_slippage_scope,
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

    def _should_inject_lg_slippage(self, target_part_name: str) -> bool:
        """Return True if LG slippage should be injected for this placement."""
        if str(target_part_name) != "LG":
            return False
        if self.lg_slippage_mode == "off":
            return False

        if self.lg_slippage_scope not in ("any", self._robot_scope_name()):
            return False

        if self.lg_slippage_mode == "once" and self._lg_slippage_triggered:
            return False

        self._lg_slippage_triggered = True
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
                trajectory_topic = (
                    _UR5E_GAZEBO_ARM_TRAJECTORY_TOPIC
                    if str(self.execution_mode or "").strip().lower() == "simulation"
                    else None
                )
                return UR5eController(
                    trajectory_topic=trajectory_topic or "/scaled_joint_trajectory_controller/joint_trajectory",
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

    def _classify_failure_mode(self, message: str) -> str:
        text = str(message or "").strip().lower()
        if not text:
            return "unknown"
        if "timed out" in text or "timeout" in text:
            return "timeout"
        if any(
            marker in text
            for marker in (
                "no parts detected",
                "not detected",
                "cannot read current ee pose",
                "planning fraction",
                "trajectory goal rejected",
                "failed to move",
                "failed to descend",
                "failed to lift",
                "unreachable",
            )
        ):
            return "unreachable"
        if "safety" in text and "block" in text:
            return "safety_block"
        return "unknown"

    def _task_failure(
        self,
        message: str,
        *,
        step: str,
        observations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        detail = str(message or "task failed")
        self.logger.error("[Robot] %s failed: %s", step, detail)
        failure_observations: Dict[str, Any] = {"step": step}
        if isinstance(observations, dict):
            failure_observations.update(observations)
        return {
            "status": "failed",
            "content": detail,
            "failure_context": {
                "failure_mode": self._classify_failure_mode(detail),
                "observations": failure_observations,
            },
        }

    async def _execute_controller_helper(
        self,
        helper_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a controller-only helper that is not exposed as a bridge primitive."""
        if self.execution_mode == "dry_run":
            await self._simulate_action(f"helper:{helper_name}({params})", duration=1.0)
            return {"success": True, "message": f"Simulated helper: {helper_name}"}

        await self._ensure_controller_prewarmed()

        if self._controller is None:
            return {"success": False, "message": "controller is not initialized"}

        method = getattr(self._controller, helper_name, None)
        if not callable(method):
            return {
                "success": False,
                "message": f"controller missing helper '{helper_name}'",
            }

        try:
            result = await asyncio.to_thread(method, **params)
            last_failure = str(getattr(self._controller, "_last_failure_message", "") or "").strip()
            if isinstance(result, bool):
                message = f"{helper_name} {'ok' if result else 'failed'}"
                if not result and last_failure:
                    message = f"{message}: {last_failure}"
                return {"success": result, "message": message}
            if isinstance(result, dict):
                normalized = dict(result)
                if (
                    not normalized.get("success")
                    and not str(normalized.get("message") or "").strip()
                    and last_failure
                ):
                    normalized["message"] = last_failure
                return normalized
            return {"success": False, "message": f"{helper_name} returned unexpected type"}
        except Exception as exc:
            self.logger.exception("[Robot] Helper '%s' execution failed", helper_name)
            return {
                "success": False,
                "message": f"{helper_name} exception: {type(exc).__name__}: {exc}",
            }

    def _log_step(self, step: str, message: str, **fields: Any) -> None:
        details = ", ".join(f"{key}={value}" for key, value in fields.items())
        suffix = f" ({details})" if details else ""
        self.logger.info("[Robot] %s: %s%s", step, message, suffix)

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

        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"Travel empty to pick location {origin_resource_location} for {part_name} "
                f"(speed={speed or 'default'})",
                duration=5.0,
            )
            self._pick_ctx = {
                "part_name": part_name,
                "model_name": "",
                "tx": 0.0, "ty": 0.0, "tz": 0.0,
                "pick_z": 0.0, "travel_z": 1.2,
                "part_height": 0.08, "tcp_offset_z": -0.17,
                "pick_tcp_z": 0.0,
                "start_x": 0.0, "start_y": 0.0, "start_z": 0.0,
            }
            self._current_state = "at_pick"
            self._position = {"x": 0.0, "y": 0.0, "z": 300.0}
            self._bridge_pose_ref = None
            return {
                "status": "completed",
                "content": f"Arrived at {origin_resource_location} ready to pick {part_name}.",
            }

        # Simulation / physical: geometry helper + primitives.
        targets = await asyncio.to_thread(
            self._controller.compute_pick_targets,
            part_name,
            product_geometry,
        )
        if not targets.get("success"):
            return self._task_failure(
                str(targets.get("message") or "failed to compute pick targets"),
                step="pick_approach.compute_pick_targets",
                observations={"part_name": part_name},
            )
        self._log_step(
            "pick_approach",
            "computed pick targets",
            part=targets.get("part_name"),
            x=f"{targets.get('tx', 0.0):.3f}",
            y=f"{targets.get('ty', 0.0):.3f}",
            pick_z=f"{targets.get('pick_z', 0.0):.3f}",
            travel_z=f"{targets.get('travel_z', 0.0):.3f}",
        )

        # Open gripper.
        self._log_step("pick_approach", "opening gripper")
        r = await self._execute_primitive("open_gripper", {})
        if not r.get("success"):
            return self._task_failure(
                str(r.get("message") or "failed to open gripper before pick approach"),
                step="pick_approach.open_gripper",
            )

        # Move above part at travel height.
        self._log_step("pick_approach", "moving above part", z=f"{targets['travel_z']:.3f}")
        r = await self._execute_controller_helper(
            "_move_xy_at_z",
            {
                "x": targets["tx"],
                "y": targets["ty"],
                "z": targets["travel_z"],
                "label": "Move above part",
                "speed": speed,
            },
        )
        if not r.get("success"):
            return self._task_failure(
                str(r.get("message") or "failed to move above part"),
                step="pick_approach.move_above_part",
                observations={"part_name": targets.get("part_name")},
            )

        # Descend to pick height.
        self._log_step("pick_approach", "descending to pick pose", z=f"{targets['pick_z']:.3f}")
        r = await self._execute_controller_helper(
            "_move_pose_direct",
            {
                "x": targets["tx"],
                "y": targets["ty"],
                "z": targets["pick_z"],
                "label": (
                    f"Descend to pick (EE z={targets['pick_z']:.3f}, "
                    f"TCP z={targets['pick_tcp_z']:.3f})"
                ),
            },
        )
        if not r.get("success"):
            return self._task_failure(
                str(r.get("message") or "failed to descend to pick position"),
                step="pick_approach.descend",
                observations={"part_name": targets.get("part_name")},
            )

        self._pick_ctx = {
            "part_name": targets["part_name"],
            "model_name": targets["model_name"],
            "tx": targets["tx"],
            "ty": targets["ty"],
            "tz": targets["tz"],
            "pick_z": targets["pick_z"],
            "travel_z": targets["travel_z"],
            "part_height": targets["part_height"],
            "tcp_offset_z": targets["tcp_offset_z"],
            "pick_tcp_z": targets["pick_tcp_z"],
            "start_x": targets["start_x"],
            "start_y": targets["start_y"],
            "start_z": targets["start_z"],
        }
        self._current_state = "at_pick"
        self._position = {"x": targets["tx"], "y": targets["ty"], "z": targets["pick_z"]}
        self._bridge_pose_ref = None
        return {
            "status": "completed",
            "content": f"Arrived at {origin_resource_location} ready to pick {part_name}.",
        }

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

        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"Picking {part_name} from {origin_resource_location} "
                f"(gripper={gripper or 'default'})",
                duration=5.0,
            )
        else:
            # Close gripper to grasp.
            self._log_step("pick_grasp", "closing gripper", part=part_name)
            r = await self._execute_primitive("close_gripper", {})
            if not r.get("success"):
                return self._task_failure(
                    str(r.get("message") or f"failed to close gripper to grasp {part_name}"),
                    step="pick_grasp.close_gripper",
                    observations={"part_name": part_name},
                )

            # Attach part in simulation (Gazebo link attacher).
            model_name = self._pick_ctx.get("model_name", "")
            if model_name:
                self._log_step("pick_grasp", "attaching part", model=model_name)
                r = await self._execute_primitive("attach_part", {"model_name": model_name})
                if not r.get("success"):
                    return self._task_failure(
                        str(r.get("message") or f"failed to attach {model_name}"),
                        step="pick_grasp.attach_part",
                        observations={"part_name": part_name, "model_name": model_name},
                    )

            # Lift part to travel height after grasping.
            travel_z = self._pick_ctx.get("travel_z", 1.2)
            tx = self._pick_ctx.get("tx", 0.0)
            ty = self._pick_ctx.get("ty", 0.0)
            self._log_step("pick_grasp", "lifting part", z=f"{travel_z:.3f}")
            r = await self._execute_controller_helper(
                "_move_pose_direct",
                {"x": tx, "y": ty, "z": travel_z, "label": "Lift after grasp"},
            )
            if not r.get("success"):
                return self._task_failure(
                    str(r.get("message") or "failed to lift after grasp"),
                    step="pick_grasp.lift",
                    observations={"part_name": part_name},
                )

        self._held_part = part_name
        self._current_state = "picked"
        self._gripper_state = "closed"
        return {"status": "completed", "content": f"Picked {part_name}."}

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

        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"Move loaded part {self._held_part} to {destination_location} "
                f"(speed={speed or 'default'})",
                duration=5.0,
            )
            self._pick_ctx.update({
                "slot_x": 0.0, "slot_y": 0.0,
                "board_top_z": 1.025, "place_z": 1.1,
                "destination_location": destination_location,
            })
            self._current_state = "positioned"
            self._position = {"x": 400.0, "y": -200.0, "z": 200.0}
            self._bridge_pose_ref = None
            return {
                "status": "completed",
                "content": f"Reached {destination_location} with {self._held_part}.",
            }

        # Simulation / physical: geometry helper + primitives.
        place = await asyncio.to_thread(
            self._controller.compute_place_targets,
            self._pick_ctx,
            product_geometry,
        )
        if not place.get("success"):
            return self._task_failure(
                str(place.get("message") or "failed to compute place targets"),
                step="place_approach.compute_place_targets",
                observations={"part_name": self._held_part},
            )
        travel_z = self._pick_ctx.get("travel_z", 1.2)
        tx = self._pick_ctx.get("tx", 0.0)
        ty = self._pick_ctx.get("ty", 0.0)
        self._log_step(
            "place_approach",
            "computed place targets",
            part=self._held_part,
            slot_x=f"{place.get('slot_x', 0.0):.3f}",
            slot_y=f"{place.get('slot_y', 0.0):.3f}",
            place_z=f"{place.get('place_z', 0.0):.3f}",
            travel_z=f"{travel_z:.3f}",
        )

        # Move laterally above destination (already at travel_z from pick_grasp lift).
        self._log_step("place_approach", "moving above destination")
        r = await self._execute_controller_helper(
            "_move_xy_at_z",
            {
                "x": place["slot_x"],
                "y": place["slot_y"],
                "z": travel_z,
                "label": "Move above destination",
                "speed": speed,
            },
        )
        if not r.get("success"):
            return self._task_failure(
                str(r.get("message") or "failed to move above destination"),
                step="place_approach.move_above_destination",
                observations={"part_name": self._held_part},
            )

        # Descend to place height.
        self._log_step("place_approach", "descending to place pose", z=f"{place['place_z']:.3f}")
        r = await self._execute_controller_helper(
            "_move_pose_direct",
            {
                "x": place["slot_x"],
                "y": place["slot_y"],
                "z": place["place_z"],
                "label": (
                    f"Descend to place (EE z={place['place_z']:.3f}, "
                    f"TCP z={place.get('place_tcp_z', 0.0):.3f})"
                ),
                "speed": getattr(self._controller, "release_descend_time_scale", None),
            },
        )
        if not r.get("success"):
            return self._task_failure(
                str(r.get("message") or "failed to descend to place position"),
                step="place_approach.descend",
                observations={"part_name": self._held_part},
            )

        self._pick_ctx.update({
            "slot_x": place["slot_x"],
            "slot_y": place["slot_y"],
            "board_top_z": place["board_top_z"],
            "place_z": place["place_z"],
            "part_height": place["part_height"],
            "destination_location": destination_location,
        })
        if place.get("model_name"):
            self._pick_ctx["model_name"] = place["model_name"]

        self._current_state = "positioned"
        self._position = {
            "x": place["slot_x"], "y": place["slot_y"], "z": place["place_z"],
        }
        self._bridge_pose_ref = None
        return {
            "status": "completed",
            "content": f"Reached {destination_location} with {self._held_part}.",
        }

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
        out_state: placed
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
        if self._should_inject_lg_slippage(placed_target):
            self.logger.error("[Robot] Assembly verification failed for %s.", placed_target)

            # Simulate failed insertion by dropping LG into a UR5e-reachable
            # recovery lane beside the assembly board.
            drop_x, drop_y, drop_z = 0.18, 0.10, 1.035
            if self.execution_mode != "dry_run":
                model_name = self._pick_ctx.get("model_name", "")
                if model_name and self._controller is not None:
                    await asyncio.to_thread(
                        self._controller.detach_part, model_name
                    )
                    # Place on its side in a board-adjacent recovery lane.
                    # LG recovery uses a flat top-down pickup, so keep the part upright.
                    await asyncio.to_thread(
                        self._controller.set_entity_pose,
                        model_name,
                        x=drop_x, y=drop_y, z=drop_z,
                        qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                    )
                    self.logger.warning(
                        "[Robot] LG slippage: %s rolled to the board-adjacent "
                        "recovery lane (%.2f, %.2f, %.2f)",
                        model_name, drop_x, drop_y, drop_z,
                    )
                    await asyncio.to_thread(self._controller.open_gripper)

            self._held_part = None
            self._current_state = "recovery_required"
            self._gripper_state = "open"
            self._pick_ctx = {}

            return {
                "status": "failed",
                "content": "Assembly verification failed.",
                "observations": {
                    "gripper_force": 0.0,
                    "camera_detection": {
                        "object_found": True,
                        "zone": "assembly_board_v1",
                        "shape_match_confidence": 0.85,
                        "orientation": "flat",
                        "visible_damage": False,
                    },
                    "last_commanded_location": destination_location,
                    "dropped_location": {
                        "x": drop_x, "y": drop_y, "z": drop_z,
                        "region": "assembly_board_edge",
                        "near": "ur5e_recovery_lane",
                        "description": "Rolled off assembly board into the UR5e recovery lane",
                    },
                },
            }

        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"Assembling {self._held_part} at {destination_location} "
                f"(orientation={orientation or 'default'})",
                duration=5.0,
            )
            placed = self._held_part
            self._held_part = None
            self._current_state = "placed"
            self._gripper_state = "open"
            self._pick_ctx = {}
            return {
                "status": "completed",
                "content": f"Assembled {placed} at {destination_location}.",
                "placed_location": destination_location,
            }

        # Simulation / physical: open gripper + detach + snap + lift.
        model_name = self._pick_ctx.get("model_name", "")
        slot_x = self._pick_ctx.get("slot_x", 0.0)
        slot_y = self._pick_ctx.get("slot_y", 0.0)
        board_top_z = self._pick_ctx.get("board_top_z", 1.025)
        part_height = self._pick_ctx.get("part_height", 0.08)
        place_z = self._pick_ctx.get("place_z", board_top_z + part_height)
        travel_z = self._pick_ctx.get("travel_z", 1.2)
        self._log_step(
            "place_insert",
            "releasing part",
            part=self._held_part,
            model=model_name or "(none)",
            slot_x=f"{slot_x:.3f}",
            slot_y=f"{slot_y:.3f}",
        )
        release = await self._execute_controller_helper(
            "_release_part_sequence",
            {
                "model_name": model_name,
                "slot_x": slot_x,
                "slot_y": slot_y,
                "part_height": part_height,
                "board_top_z": board_top_z,
                "place_z": place_z,
                "travel_z": travel_z,
            },
        )
        released_ok = bool(release.get("success"))

        placed = self._held_part
        self._held_part = None
        self._current_state = "placed"
        self._gripper_state = "open"
        self._pick_ctx = {}

        if not released_ok:
            self._current_state = "recovery_required"
            return self._task_failure(
                str(release.get("message") or f"failed to assemble {placed} at {destination_location}"),
                step="place_insert.release",
                observations={"part_name": placed, "destination_location": destination_location},
            )

        return {
            "status": "completed",
            "content": f"Assembled {placed} at {destination_location}.",
            "placed_location": destination_location,
        }

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
        out_state: idle

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

        if self.execution_mode == "dry_run":
            await self._simulate_action("Moving arm to home position", duration=3.0)
            self._current_state = "idle"
            self._position = {"x": 0.0, "y": 0.0, "z": 445.0}
            self._bridge_pose_ref = "home"
            self._pick_ctx = {}
            return {"status": "completed", "content": "At home position."}

        self._log_step("move_home", "returning robot to home pose")
        r = await self._execute_controller_helper("move_home", {})
        if r.get("success"):
            self._current_state = "idle"
            message = str(r.get("message") or "")
            if "remembered start pose" in message:
                pose = await self._execute_primitive("get_current_pose", {})
                pose_data = pose.get("pose") if isinstance(pose, dict) else None
                if isinstance(pose_data, dict):
                    self._position = {
                        "x": float(pose_data.get("x", 0.0)),
                        "y": float(pose_data.get("y", 0.0)),
                        "z": float(pose_data.get("z", 0.0)),
                    }
                else:
                    start_x = self._pick_ctx.get("start_x")
                    start_y = self._pick_ctx.get("start_y")
                    start_z = self._pick_ctx.get("start_z")
                    self._position = {
                        "x": float(start_x or 0.0),
                        "y": float(start_y or 0.0),
                        "z": float(start_z or 0.0),
                    }
                self._bridge_pose_ref = None
            else:
                self._position = {"x": 0.0, "y": 0.0, "z": 445.0}
                self._bridge_pose_ref = "home"
            self._pick_ctx = {}
            return {"status": "completed", "content": "At home position."}

        return self._task_failure(
            str(r.get("message") or "failed to move to home position"),
            step="move_home.controller",
        )

    # ------------------------------------------------------------------ #
    # Bridge-only recovery macro executor
    # ------------------------------------------------------------------ #

    # Controller primitives available for bridge macro steps.
    _BRIDGE_PRIMITIVES = frozenset({
        "move_cartesian",
        "move_pose",
        "move_relative",
        "move_to_named_pose",
        "rotate_wrist",
        "open_gripper",
        "close_gripper",
        "detect_parts",
        "compute_pick_targets",
        "compute_place_targets",
        "attach_part",
        "detach_part",
        "get_current_pose",
    })

    async def execute_recovery_macro(
        self,
        macro_name: str,
        primitive_steps: list,
        *,
        expected_start_state: Optional[str] = None,
        expected_snapshot: Optional[Dict[str, Any]] = None,
        product_jid: Optional[str] = None,
        task_id: Optional[str] = None,
        in_state: Optional[str] = None,
        out_state: Optional[str] = None,
        **context: Any,
    ) -> Dict[str, Any]:
        """Execute a bridge-generated recovery macro as an ordered primitive sequence.

        This method is registered in self.executables for runtime dispatch
        but is excluded from function_names and the shared tools catalog.
        It is only callable through bridge-approved recovery macro tasks.
        """
        from cais_spade_llm.agents.intelligent_product.replanner.primitive_semantics import (
            apply_effects_to_snapshot,
            build_primitive_catalog,
            extract_step_output,
            get_robot_bridge_snapshot,
            resolve_param_refs,
            snapshot_matches_expected,
            sync_agent_from_bridge_snapshot,
            validate_and_project_steps,
        )

        self.logger.info(
            "[Robot] execute_recovery_macro '%s' (%d steps) start_state=%s expected=%s",
            macro_name,
            len(primitive_steps or []),
            self._current_state,
            expected_start_state,
        )

        # Validate start state if specified.
        if expected_start_state and self._current_state != expected_start_state:
            msg = (
                f"Recovery macro '{macro_name}' expected start state "
                f"'{expected_start_state}' but robot is in '{self._current_state}'"
            )
            self.logger.error("[Robot] %s", msg)
            return {
                "status": "failed",
                "content": msg,
                "observations": {
                    "macro_name": macro_name,
                    "expected_start_state": expected_start_state,
                    "actual_state": self._current_state,
                    "step_index": -1,
                },
            }

        runtime_snapshot = get_robot_bridge_snapshot(self)
        if expected_snapshot:
            matches, mismatch_message = snapshot_matches_expected(runtime_snapshot, expected_snapshot)
            if not matches:
                msg = (
                    f"Recovery macro '{macro_name}' expected snapshot mismatch: "
                    f"{mismatch_message}"
                )
                self.logger.error("[Robot] %s", msg)
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "expected_snapshot": expected_snapshot,
                        "actual_snapshot": runtime_snapshot,
                        "step_index": -1,
                    },
                }

        if not primitive_steps:
            return {
                "status": "failed",
                "content": f"Recovery macro '{macro_name}' has no primitive steps",
            }

        primitive_catalog = build_primitive_catalog(self)
        primitive_meta_by_name = {
            str(entry.get("name", "")).strip(): entry
            for entry in primitive_catalog
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        }
        semantic_ok, _projected_runtime_snapshot, semantic_error = validate_and_project_steps(
            primitive_steps,
            primitive_catalog,
            runtime_snapshot,
            grounding_context={},
        )
        if not semantic_ok:
            msg = (
                f"Recovery macro '{macro_name}' failed runtime semantic validation: "
                f"{semantic_error}"
            )
            self.logger.error("[Robot] %s", msg)
            return {
                "status": "failed",
                "content": msg,
                "observations": {
                    "macro_name": macro_name,
                    "expected_snapshot": expected_snapshot,
                    "actual_snapshot": runtime_snapshot,
                    "semantic_error": semantic_error,
                    "step_index": -1,
                },
            }

        # Execute each primitive step sequentially.
        results: list[Dict[str, Any]] = []
        step_outputs: dict[str, Any] = {}
        for step_idx, step in enumerate(primitive_steps):
            primitive = step.get("primitive", "") if isinstance(step, dict) else ""
            raw_params = step.get("params", {}) if isinstance(step, dict) else {}
            store_as = str(step.get("store_as", "")).strip() if isinstance(step, dict) else ""

            if primitive not in self._BRIDGE_PRIMITIVES:
                msg = (
                    f"Unknown primitive '{primitive}' at step {step_idx} "
                    f"in macro '{macro_name}'"
                )
                self.logger.error("[Robot] %s", msg)
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "completed_steps": len(results),
                    },
                }

            try:
                params = resolve_param_refs(
                    raw_params,
                    {},
                    step_outputs=step_outputs,
                )
            except Exception as exc:
                msg = (
                    f"Macro '{macro_name}' could not resolve params at step {step_idx} "
                    f"({primitive}): {exc}"
                )
                self.logger.error("[Robot] %s", msg)
                return {
                    "status": "failed",
                    "content": msg,
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "raw_params": raw_params,
                        "step_outputs": deepcopy(step_outputs),
                    },
                }

            # Dispatch to controller primitive.
            step_result = await self._execute_primitive(primitive, params)

            self.logger.info(
                "[Robot] macro '%s' step %d/%d: %s -> %s",
                macro_name,
                step_idx + 1,
                len(primitive_steps),
                primitive,
                "ok" if step_result.get("success") else "failed",
            )
            results.append({"primitive": primitive, "result": step_result})

            if not step_result.get("success", False):
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' failed at step {step_idx} "
                        f"({primitive}): {step_result.get('message', '')}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "primitive_result": step_result,
                        "completed_steps": len(results) - 1,
                        "total_steps": len(primitive_steps),
                    },
                }

            primitive_meta = primitive_meta_by_name.get(primitive)
            if primitive_meta is not None:
                runtime_snapshot = apply_effects_to_snapshot(
                    {**dict(step), "params": params},
                    primitive_meta,
                    runtime_snapshot,
                )
                sync_agent_from_bridge_snapshot(self, runtime_snapshot)

            if store_as:
                step_output, output_error = extract_step_output(
                    primitive=primitive,
                    params=params,
                    step_result=step_result,
                )
                if output_error:
                    return {
                        "status": "failed",
                        "content": (
                            f"Macro '{macro_name}' could not store output at step {step_idx} "
                            f"({primitive}): {output_error}"
                        ),
                        "observations": {
                            "macro_name": macro_name,
                            "step_index": step_idx,
                            "primitive": primitive,
                            "primitive_result": step_result,
                            "store_as": store_as,
                            "completed_steps": len(results),
                        },
                    }
                step_outputs[store_as] = step_output

        # All steps succeeded. Update logical state if out_state specified.
        if out_state:
            runtime_snapshot["current_state"] = out_state
            sync_agent_from_bridge_snapshot(self, runtime_snapshot)
        self.logger.info(
            "[Robot] Recovery macro '%s' completed (%d steps). state=%s",
            macro_name,
            len(primitive_steps),
            self._current_state,
        )

        return {
            "status": "completed",
            "content": f"Recovery macro '{macro_name}' completed successfully",
            "observations": {
                "macro_name": macro_name,
                "completed_steps": len(results),
                "total_steps": len(primitive_steps),
            },
        }

    async def _execute_primitive(
        self, primitive: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single controller primitive, handling dry_run and simulation modes."""
        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"primitive:{primitive}({params})", duration=1.0
            )
            return {"success": True, "message": f"Simulated: {primitive}"}

        await self._ensure_controller_prewarmed()

        if self._controller is None:
            return {"success": False, "message": "controller is not initialized"}

        method = getattr(self._controller, primitive, None)
        if not callable(method):
            return {
                "success": False,
                "message": f"controller missing primitive '{primitive}'",
            }

        try:
            result = await asyncio.to_thread(method, **params)
            last_failure = str(getattr(self._controller, "_last_failure_message", "") or "").strip()
            # Normalize: bool-returning primitives (open_gripper, close_gripper)
            if isinstance(result, bool):
                message = f"{primitive} {'ok' if result else 'failed'}"
                if not result and last_failure:
                    message = f"{message}: {last_failure}"
                return {
                    "success": result,
                    "message": message,
                }
            # List-returning primitives (detect_parts)
            if isinstance(result, list):
                return {
                    "success": True,
                    "message": f"{primitive} returned {len(result)} items",
                    "data": result,
                }
            # Dict-returning primitives (move_cartesian, etc.)
            if isinstance(result, dict):
                normalized = dict(result)
                if (
                    not normalized.get("success")
                    and not str(normalized.get("message") or "").strip()
                    and last_failure
                ):
                    normalized["message"] = last_failure
                return normalized
            return {"success": False, "message": f"{primitive} returned unexpected type"}
        except Exception as exc:
            self.logger.exception("[Robot] Primitive '%s' execution failed", primitive)
            return {
                "success": False,
                "message": f"{primitive} exception: {type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_bridge_snapshot(self) -> Dict[str, Any]:
        """Return the current primitive-level bridge snapshot for this robot."""
        from cais_spade_llm.agents.intelligent_product.replanner.primitive_semantics import (
            get_robot_bridge_snapshot,
        )

        return get_robot_bridge_snapshot(self)

    def _snapshot_state(self) -> Dict[str, Any]:
        """Robot-specific state snapshot (override)."""
        controller_ready = True
        if self.execution_mode != "dry_run":
            controller_ready = bool(
                self._controller is not None
                and getattr(self._controller, "is_usable", lambda: False)()
            )
        return {
            "execution_mode": self.execution_mode,
            "controller_ready": controller_ready,
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
