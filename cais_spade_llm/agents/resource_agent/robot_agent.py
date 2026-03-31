"""Robot resource agent exposing pick/move/place primitives for assembly."""

from __future__ import annotations

import asyncio
import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional

from cais_spade_llm.resources.robot.robot_profile import ROBOT_PROFILE
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.resources.robot import UR5eController, XArm6Controller
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (
    build_failure_event,
    failure_context_from_scenario_config,
    load_failure_scenario_config,
)

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
    _RESOURCE_PROFILE = ROBOT_PROFILE

    def __init__(self, jid: str, password: str, *, name: str, **kw: Any) -> None:
        raw_failure_scenarios = kw.pop("failure_scenarios", None)
        self.failure_scenarios = self._normalize_failure_scenario_bindings(raw_failure_scenarios)
        self._triggered_failure_scenarios: set[str] = set()

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
        self._current_state: str = "idle"  # idle, at_pick, picked, positioned, placed (placed = at destination, part released)
        self._position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}  # Simulated position
        self._gripper_state: str = "open"
        self._bridge_pose_ref: Optional[str] = None
        # Shared task execution context threaded across task-level functions.
        # `_pick_ctx` remains as a temporary compatibility alias.
        self._task_ctx: Dict[str, Any] = {}
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
        self._primitive_catalog_cache: list | None = None

        self.logger.info(
            (
                "RobotAgent '%s' initialized. mode=%s tools=%s "
                "failure_scenarios=%s"
            ),
            name,
            self.execution_mode,
            list(self.executables.keys()),
            [
                {
                    "scenario_id": binding.get("scenario_id"),
                    "mode": binding.get("mode"),
                    "scope": binding.get("scope"),
                }
                for binding in self.failure_scenarios
            ],
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

    @property
    def _pick_ctx(self) -> Dict[str, Any]:
        """Compatibility alias for older code paths that still reference `_pick_ctx`."""
        return self._task_ctx

    @_pick_ctx.setter
    def _pick_ctx(self, value: Dict[str, Any]) -> None:
        self._task_ctx = value if isinstance(value, dict) else {}

    @staticmethod
    def _normalize_failure_scenario_bindings(raw_bindings: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for raw_binding in raw_bindings or []:
            if isinstance(raw_binding, str):
                binding = {"scenario_id": raw_binding}
            elif isinstance(raw_binding, dict):
                binding = deepcopy(raw_binding)
            else:
                continue
            scenario_id = str(binding.get("scenario_id") or "").strip()
            if not scenario_id:
                continue
            mode = str(binding.get("mode") or "always").strip().lower()
            if mode not in {"always", "once", "off"}:
                mode = "always"
            scope = str(binding.get("scope") or "any").strip().lower() or "any"
            binding["scenario_id"] = scenario_id
            binding["mode"] = mode
            binding["scope"] = scope
            normalized.append(binding)
        return normalized

    @staticmethod
    def _normalize_optional_selector(
        raw_value: Any,
        *,
        lowercase: bool = False,
    ) -> set[str] | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_items = [raw_value]
        elif isinstance(raw_value, (list, tuple, set)):
            raw_items = list(raw_value)
        else:
            return set()

        values: set[str] = set()
        for raw_item in raw_items:
            token = str(raw_item or "").strip()
            if not token:
                continue
            values.add(token.lower() if lowercase else token)
        if not values:
            return set()
        return values

    def _scenario_selector_values(
        self,
        *,
        trigger: dict[str, Any],
        scenario_id: str,
        plural_key: str,
        singular_key: str | None = None,
        lowercase: bool = False,
    ) -> set[str] | None:
        if plural_key in trigger:
            raw_value = trigger.get(plural_key)
        elif singular_key and singular_key in trigger:
            raw_value = trigger.get(singular_key)
        else:
            return None
        values = self._normalize_optional_selector(raw_value, lowercase=lowercase)
        if values == set():
            self.logger.warning(
                "[Robot] Failure scenario '%s' has invalid selector '%s'; skipping scenario.",
                scenario_id,
                plural_key,
            )
        return values

    def _match_failure_scenario(
        self,
        *,
        function_name: str,
        checkpoint: str,
        part_name: str = "",
        call_args: Dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        function_token = str(function_name or "").strip().lower()
        checkpoint_token = str(checkpoint or "").strip().lower()
        part_token = str(part_name or "").strip()
        for binding in self.failure_scenarios:
            scenario_id = str(binding.get("scenario_id") or "").strip()
            if not scenario_id:
                continue
            mode = str(binding.get("mode") or "always").strip().lower()
            scope = str(binding.get("scope") or "any").strip().lower() or "any"
            if mode == "off":
                continue
            if scope not in {"any", self._robot_scope_name()}:
                continue
            trigger_key = str(binding.get("trigger_key") or scenario_id).strip() or scenario_id
            if mode == "once" and trigger_key in self._triggered_failure_scenarios:
                continue

            try:
                scenario_config = load_failure_scenario_config(scenario_id)
            except Exception:
                self.logger.exception(
                    "[Robot] Failed to load failure scenario config '%s'.",
                    scenario_id,
                )
                continue

            trigger = dict(scenario_config.get("trigger") or {})
            function_names = self._scenario_selector_values(
                trigger=trigger,
                scenario_id=scenario_id,
                plural_key="function_names",
                singular_key="function_name",
                lowercase=True,
            )
            if function_names == set():
                continue
            if function_names is not None and function_token not in function_names:
                continue

            checkpoints = self._scenario_selector_values(
                trigger=trigger,
                scenario_id=scenario_id,
                plural_key="checkpoints",
                lowercase=True,
            )
            if checkpoints == set():
                continue
            if checkpoints is not None and checkpoint_token not in checkpoints:
                continue

            part_names = self._scenario_selector_values(
                trigger=trigger,
                scenario_id=scenario_id,
                plural_key="part_names",
                singular_key="part_name",
            )
            if part_names == set():
                continue
            if part_names is not None and part_token not in part_names:
                continue

            execution_modes = self._scenario_selector_values(
                trigger=trigger,
                scenario_id=scenario_id,
                plural_key="execution_modes",
                lowercase=True,
            )
            if execution_modes == set():
                continue
            if execution_modes is None:
                execution_modes = self._scenario_selector_values(
                    trigger=dict(scenario_config.get("injection") or {}),
                    scenario_id=scenario_id,
                    plural_key="enabled_execution_modes",
                    lowercase=True,
                )
                if execution_modes == set():
                    continue
            if execution_modes is not None and self.execution_mode not in execution_modes:
                continue

            effects = deepcopy(scenario_config.get("effects") or [])
            if not isinstance(effects, list) or not effects:
                self.logger.warning(
                    "[Robot] Failure scenario '%s' has no effects; skipping scenario.",
                    scenario_id,
                )
                continue

            return {
                "scenario_id": scenario_id,
                "trigger_key": trigger_key,
                "binding": deepcopy(binding),
                "scenario_config": deepcopy(scenario_config),
                "effects": effects,
                "call_args": deepcopy(call_args or {}),
                "function_name": str(function_name or "").strip(),
                "checkpoint": str(checkpoint or "").strip(),
                "matched_part_name": part_token,
                "base_failure_context": failure_context_from_scenario_config(scenario_config),
            }
        return None

    @staticmethod
    def _resolve_effect_ref(ref: str, effect_context: Dict[str, Any]) -> Any:
        path = [segment for segment in str(ref or "").split(".") if segment]
        if not path:
            raise ValueError("effect ref is empty")
        sentinel = object()
        current: Any = effect_context.get(path[0], sentinel)
        if current is sentinel:
            raise KeyError(path[0])
        for segment in path[1:]:
            if isinstance(current, dict):
                if segment not in current:
                    raise KeyError(ref)
                current = current[segment]
            else:
                if not hasattr(current, segment):
                    raise KeyError(ref)
                current = getattr(current, segment)
        return deepcopy(current)

    def _resolve_effect_value(self, value: Any, effect_context: Dict[str, Any]) -> Any:
        if isinstance(value, dict):
            if set(value.keys()) == {"ref"}:
                return self._resolve_effect_ref(str(value.get("ref") or ""), effect_context)
            return {
                key: self._resolve_effect_value(item, effect_context)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_effect_value(item, effect_context) for item in value]
        return deepcopy(value)

    def _scenario_error_result(
        self,
        match: Dict[str, Any],
        *,
        message: str,
        effect_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        scenario_id = str(match.get("scenario_id") or "").strip() or "unknown_scenario"
        detail = f"Failure scenario '{scenario_id}' is misconfigured: {message}"
        self.logger.error("[Robot] %s", detail)
        observations = deepcopy(effect_context.get("emitted_observations") or {})
        observations["scenario_error"] = message
        failure_event = build_failure_event(
            failed_task_id=str(effect_context.get("task_id") or "").strip(),
            failed_resource_jid=str(self.jid or "").strip(),
            failed_function_name=str(effect_context.get("function_name") or "").strip(),
            final_status="failed",
            part_name=str(effect_context.get("matched_part_name") or "").strip(),
            base_failure_context=deepcopy(effect_context.get("base_failure_context") or {}),
            observations=observations,
        )
        failure_context = dict(failure_event.get("failure_context") or {})
        failure_observations = deepcopy(failure_context.get("observations") or {})
        return {
            "status": "failed",
            "content": detail,
            "observations": failure_observations,
            "failure_context": failure_context,
        }

    async def _apply_failure_effect(
        self,
        *,
        match: Dict[str, Any],
        effect: Dict[str, Any],
        effect_context: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        effect_type = str(effect.get("type") or "").strip()
        if not effect_type:
            raise ValueError("effect is missing type")

        if effect_type == "set_agent_field":
            field = str(effect.get("field") or "").strip()
            if not field:
                raise ValueError("set_agent_field requires 'field'")
            value = self._resolve_effect_value(effect.get("value"), effect_context)
            setattr(self, field, value)
            if field in {"_task_ctx", "_pick_ctx"}:
                effect_context["task_ctx"] = self._task_ctx
            return None

        if effect_type == "clear_agent_field":
            field = str(effect.get("field") or "").strip()
            if not field:
                raise ValueError("clear_agent_field requires 'field'")
            setattr(self, field, None)
            if field in {"_task_ctx", "_pick_ctx"}:
                effect_context["task_ctx"] = self._task_ctx
            return None

        if effect_type == "clear_task_context":
            self._task_ctx = {}
            effect_context["task_ctx"] = self._task_ctx
            return None

        if effect_type == "detach_attached_model":
            model_name = str(
                self._resolve_effect_value(effect.get("model_name"), effect_context) or ""
            ).strip()
            if model_name:
                result = await self._execute_primitive("detach_part", {"model_name": model_name})
                if not result.get("success", False):
                    self.logger.warning(
                        "[Robot] Failure effect detach_part did not succeed for %s: %s",
                        model_name,
                        result.get("message"),
                    )
            return None

        if effect_type == "open_gripper":
            result = await self._execute_primitive("open_gripper", {})
            if not result.get("success", False):
                self.logger.warning(
                    "[Robot] Failure effect open_gripper did not succeed: %s",
                    result.get("message"),
                )
            return None

        if effect_type == "set_entity_pose":
            model_name = str(
                self._resolve_effect_value(effect.get("model_name"), effect_context) or ""
            ).strip()
            pose = self._resolve_effect_value(effect.get("pose"), effect_context)
            orientation_quat = self._resolve_effect_value(
                effect.get("orientation_quat"),
                effect_context,
            )
            if not model_name:
                return None
            if not isinstance(pose, dict):
                raise ValueError("set_entity_pose requires pose dict")
            if not isinstance(orientation_quat, dict):
                raise ValueError("set_entity_pose requires orientation_quat dict")
            try:
                x = float(pose["x"])
                y = float(pose["y"])
                z = float(pose["z"])
                qx = float(orientation_quat["qx"])
                qy = float(orientation_quat["qy"])
                qz = float(orientation_quat["qz"])
                qw = float(orientation_quat["qw"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"set_entity_pose has invalid pose/quaternion: {exc}") from exc
            if self.execution_mode == "dry_run" or self._controller is None:
                return None
            try:
                await asyncio.to_thread(
                    self._controller.set_entity_pose,
                    model_name,
                    x=x,
                    y=y,
                    z=z,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                )
            except Exception as exc:
                self.logger.warning(
                    "[Robot] Failure effect set_entity_pose failed for %s: %s",
                    model_name,
                    exc,
                )
            return None

        if effect_type == "emit_observation_fields":
            fields = effect.get("fields")
            if not isinstance(fields, dict):
                raise ValueError("emit_observation_fields requires 'fields' dict")
            resolved_fields = self._resolve_effect_value(fields, effect_context)
            if not isinstance(resolved_fields, dict):
                raise ValueError("emit_observation_fields resolved to non-dict")
            effect_context["emitted_observations"].update(resolved_fields)
            return None

        if effect_type == "return_failure":
            status = str(
                self._resolve_effect_value(effect.get("status") or "failed", effect_context)
                or "failed"
            ).strip()
            content = str(
                self._resolve_effect_value(effect.get("content") or "task failed", effect_context)
                or "task failed"
            ).strip()
            failure_event = build_failure_event(
                failed_task_id=str(effect_context.get("task_id") or "").strip(),
                failed_resource_jid=str(self.jid or "").strip(),
                failed_function_name=str(effect_context.get("function_name") or "").strip(),
                final_status=status,
                part_name=str(effect_context.get("matched_part_name") or "").strip(),
                base_failure_context=deepcopy(effect_context.get("base_failure_context") or {}),
                observations=deepcopy(effect_context.get("emitted_observations") or {}),
            )
            failure_context = dict(failure_event.get("failure_context") or {})
            failure_observations = deepcopy(failure_context.get("observations") or {})
            return {
                "status": status,
                "content": content,
                "observations": failure_observations,
                "failure_context": failure_context,
            }

        raise ValueError(f"unknown failure effect type '{effect_type}'")

    async def _maybe_inject_failure(
        self,
        *,
        function_name: str,
        checkpoint: str,
        part_name: str = "",
        call_args: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        match = self._match_failure_scenario(
            function_name=function_name,
            checkpoint=checkpoint,
            part_name=part_name,
            call_args=call_args,
        )
        if match is None:
            return None

        self.logger.warning(
            "[Robot] Injecting failure scenario '%s' at %s.%s",
            match["scenario_id"],
            function_name,
            checkpoint,
        )
        self._triggered_failure_scenarios.add(str(match.get("trigger_key") or "").strip())

        effect_context: Dict[str, Any] = {
            "agent": self,
            "task_ctx": self._task_ctx,
            "call_args": deepcopy(call_args or {}),
            "injection": deepcopy(dict(match["scenario_config"].get("injection") or {})),
            "matched_part_name": str(match.get("matched_part_name") or "").strip(),
            "task_id": str(dict(call_args or {}).get("task_id") or "").strip(),
            "function_name": str(function_name or "").strip(),
            "base_failure_context": deepcopy(match.get("base_failure_context") or {}),
            "emitted_observations": {},
        }

        for effect in match.get("effects") or []:
            if not isinstance(effect, dict):
                return self._scenario_error_result(
                    match,
                    message="effect entry is not an object",
                    effect_context=effect_context,
                )
            try:
                maybe_result = await self._apply_failure_effect(
                    match=match,
                    effect=effect,
                    effect_context=effect_context,
                )
            except Exception as exc:
                return self._scenario_error_result(
                    match,
                    message=str(exc),
                    effect_context=effect_context,
                )
            if maybe_result is not None:
                return maybe_result

        return self._scenario_error_result(
            match,
            message="scenario completed without a return_failure effect",
            effect_context=effect_context,
        )

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

    def _task_failure(
        self,
        message: str,
        *,
        step: str,
        observations: Optional[Dict[str, Any]] = None,
        failure_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        detail = str(message or "task failed")
        self.logger.error("[Robot] %s failed: %s", step, detail)
        failure_observations: Dict[str, Any] = {"step": step}
        if isinstance(observations, dict):
            failure_observations.update(observations)
        context_payload = deepcopy(failure_context if isinstance(failure_context, dict) else {})
        merged_observations = deepcopy(context_payload.get("observations") or {})
        if not isinstance(merged_observations, dict):
            merged_observations = {}
        merged_observations.update(failure_observations)
        context_payload["observations"] = merged_observations
        return {
            "status": "failed",
            "content": detail,
            "failure_context": context_payload,
        }

    async def _execute_controller_helper(
        self,
        helper_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a controller-only helper that is not exposed as a bridge primitive."""
        if self.execution_mode == "dry_run":
            self.logger.debug("[Robot] dry_run helper: %s", helper_name)
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

        call_args = {
            "origin_resource_location": origin_resource_location,
            "part_name": part_name,
            "speed": speed,
            "product_jid": product_jid,
            "task_id": task_id,
            "product_geometry": product_geometry,
        }
        injected = await self._maybe_inject_failure(
            function_name="pick_approach",
            checkpoint="before_execute",
            part_name=part_name,
            call_args=call_args,
        )
        if injected is not None:
            return injected

        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"Travel empty to pick location {origin_resource_location} for {part_name} "
                f"(speed={speed or 'default'})",
                duration=5.0,
            )
            injected = await self._maybe_inject_failure(
                function_name="pick_approach",
                checkpoint="after_execute_before_commit",
                part_name=part_name,
                call_args=call_args,
            )
            if injected is not None:
                return injected
            self._task_ctx = {
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

        injected = await self._maybe_inject_failure(
            function_name="pick_approach",
            checkpoint="after_execute_before_commit",
            part_name=part_name,
            call_args=call_args,
        )
        if injected is not None:
            return injected

        self._task_ctx = {
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

        call_args = {
            "part_name": part_name,
            "origin_resource_location": origin_resource_location,
            "gripper": gripper,
            "product_jid": product_jid,
            "task_id": task_id,
            "product_geometry": product_geometry,
        }
        injected = await self._maybe_inject_failure(
            function_name="pick_grasp",
            checkpoint="before_execute",
            part_name=part_name,
            call_args=call_args,
        )
        if injected is not None:
            return injected

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
            model_name = self._task_ctx.get("model_name", "")
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
            travel_z = self._task_ctx.get("travel_z", 1.2)
            tx = self._task_ctx.get("tx", 0.0)
            ty = self._task_ctx.get("ty", 0.0)
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

        injected = await self._maybe_inject_failure(
            function_name="pick_grasp",
            checkpoint="after_execute_before_commit",
            part_name=part_name,
            call_args=call_args,
        )
        if injected is not None:
            return injected

        self._held_part = part_name
        self._current_state = "picked"
        self._gripper_state = "closed"
        return {
            "status": "completed",
            "content": f"Picked {part_name}.",
            "observations": {
                "part_name": part_name,
                "origin_pose": {
                    "x": self._task_ctx.get("tx", 0.0),
                    "y": self._task_ctx.get("ty", 0.0),
                    "z": self._task_ctx.get("tz", 0.0),
                },
            },
        }

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

        call_args = {
            "destination_location": destination_location,
            "part_name": part_name,
            "speed": speed,
            "product_jid": product_jid,
            "task_id": task_id,
            "product_geometry": product_geometry,
        }
        injected = await self._maybe_inject_failure(
            function_name="place_approach",
            checkpoint="before_execute",
            part_name=part_name or str(self._held_part or ""),
            call_args=call_args,
        )
        if injected is not None:
            return injected

        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"Move loaded part {self._held_part} to {destination_location} "
                f"(speed={speed or 'default'})",
                duration=5.0,
            )
            injected = await self._maybe_inject_failure(
                function_name="place_approach",
                checkpoint="after_execute_before_commit",
                part_name=part_name or str(self._held_part or ""),
                call_args=call_args,
            )
            if injected is not None:
                return injected
            self._task_ctx.update({
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
            self._task_ctx,
            product_geometry,
            part_name,
            0.0,
            destination_location,
        )
        if not place.get("success"):
            return self._task_failure(
                str(place.get("message") or "failed to compute place targets"),
                step="place_approach.compute_place_targets",
                observations={"part_name": self._held_part},
            )
        travel_z = self._task_ctx.get("travel_z", 1.2)
        tx = self._task_ctx.get("tx", 0.0)
        ty = self._task_ctx.get("ty", 0.0)
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

        injected = await self._maybe_inject_failure(
            function_name="place_approach",
            checkpoint="after_execute_before_commit",
            part_name=part_name or str(self._held_part or ""),
            call_args=call_args,
        )
        if injected is not None:
            return injected

        self._task_ctx.update({
            "slot_x": place["slot_x"],
            "slot_y": place["slot_y"],
            "board_top_z": place["board_top_z"],
            "place_z": place["place_z"],
            "part_height": place["part_height"],
            "destination_location": destination_location,
        })
        if place.get("model_name"):
            self._task_ctx["model_name"] = place["model_name"]

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
        call_args = {
            "destination_location": destination_location,
            "part_name": part_name,
            "orientation": orientation,
            "product_jid": product_jid,
            "task_id": task_id,
            "product_geometry": product_geometry,
        }
        injected = await self._maybe_inject_failure(
            function_name="place_insert",
            checkpoint="before_execute",
            part_name=placed_target,
            call_args=call_args,
        )
        if injected is not None:
            return injected

        if self.execution_mode == "dry_run":
            await self._simulate_action(
                f"Assembling {self._held_part} at {destination_location} "
                f"(orientation={orientation or 'default'})",
                duration=5.0,
            )
            injected = await self._maybe_inject_failure(
                function_name="place_insert",
                checkpoint="after_execute_before_commit",
                part_name=placed_target,
                call_args=call_args,
            )
            if injected is not None:
                return injected
            placed = self._held_part
            self._held_part = None
            self._current_state = "placed"
            self._gripper_state = "open"
            self._task_ctx = {}
            return {
                "status": "completed",
                "content": f"Assembled {placed} at {destination_location}.",
                "placed_location": destination_location,
            }

        # Simulation / physical: open gripper + detach + snap + lift.
        model_name = self._task_ctx.get("model_name", "")
        slot_x = self._task_ctx.get("slot_x", 0.0)
        slot_y = self._task_ctx.get("slot_y", 0.0)
        board_top_z = self._task_ctx.get("board_top_z", 1.025)
        part_height = self._task_ctx.get("part_height", 0.08)
        place_z = self._task_ctx.get("place_z", board_top_z + part_height)
        travel_z = self._task_ctx.get("travel_z", 1.2)
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

        if not released_ok:
            return self._task_failure(
                str(release.get("message") or f"failed to assemble {self._held_part} at {destination_location}"),
                step="place_insert.release",
                observations={"part_name": self._held_part, "destination_location": destination_location},
            )

        injected = await self._maybe_inject_failure(
            function_name="place_insert",
            checkpoint="after_execute_before_commit",
            part_name=placed_target,
            call_args=call_args,
        )
        if injected is not None:
            return injected

        placed = self._held_part
        self._held_part = None
        self._current_state = "placed"
        self._gripper_state = "open"
        self._task_ctx = {}

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

        call_args = {
            "product_jid": product_jid,
            "task_id": task_id,
        }
        injected = await self._maybe_inject_failure(
            function_name="move_home",
            checkpoint="before_execute",
            part_name="",
            call_args=call_args,
        )
        if injected is not None:
            return injected

        if self.execution_mode == "dry_run":
            await self._simulate_action("Moving arm to home position", duration=3.0)
            injected = await self._maybe_inject_failure(
                function_name="move_home",
                checkpoint="after_execute_before_commit",
                part_name="",
                call_args=call_args,
            )
            if injected is not None:
                return injected
            self._current_state = "idle"
            self._position = {"x": 0.0, "y": 0.0, "z": 445.0}
            self._bridge_pose_ref = "home"
            self._task_ctx = {}
            return {"status": "completed", "content": "At home position."}

        self._log_step("move_home", "returning robot to home pose")
        r = await self._execute_controller_helper("move_home", {})
        if r.get("success"):
            injected = await self._maybe_inject_failure(
                function_name="move_home",
                checkpoint="after_execute_before_commit",
                part_name="",
                call_args=call_args,
            )
            if injected is not None:
                return injected
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
                    start_x = self._task_ctx.get("start_x")
                    start_y = self._task_ctx.get("start_y")
                    start_z = self._task_ctx.get("start_z")
                    self._position = {
                        "x": float(start_x or 0.0),
                        "y": float(start_y or 0.0),
                        "z": float(start_z or 0.0),
                    }
                self._bridge_pose_ref = None
            else:
                self._position = {"x": 0.0, "y": 0.0, "z": 445.0}
                self._bridge_pose_ref = "home"
            self._task_ctx = {}
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
        "open_gripper",
        "close_gripper",
        "detect_parts",
        "compute_pick_targets",
        "compute_place_targets",
        "attach_part",
        "detach_part",
        "get_current_pose",
    })
    _BRIDGE_OBSERVATION_PRIMITIVES = frozenset({
        "detect_parts",
        "compute_pick_targets",
        "compute_place_targets",
        "get_current_pose",
    })

    async def execute_bridge_observation(
        self,
        primitive: str,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Execute one planner-approved observation/generation primitive."""
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
            extract_step_output,
        )

        primitive_name = str(primitive or "").strip()
        normalized_params = dict(params or {})
        if primitive_name not in self._BRIDGE_OBSERVATION_PRIMITIVES:
            return {
                "success": False,
                "message": (
                    f"bridge observation primitive '{primitive_name}' is not allowed; "
                    f"allowed={sorted(self._BRIDGE_OBSERVATION_PRIMITIVES)}"
                ),
            }

        step_result = await self._execute_primitive(primitive_name, normalized_params)
        snapshot = self.get_bridge_snapshot()
        if not step_result.get("success", False):
            return {
                "success": False,
                "message": str(step_result.get("message", "") or f"{primitive_name} failed"),
                "primitive": primitive_name,
                "params": normalized_params,
                "primitive_result": step_result,
                "snapshot": snapshot,
            }

        observation, output_error = extract_step_output(
            primitive=primitive_name,
            params=normalized_params,
            step_result=step_result,
        )
        if output_error:
            return {
                "success": False,
                "message": f"{primitive_name} output could not be normalized: {output_error}",
                "primitive": primitive_name,
                "params": normalized_params,
                "primitive_result": step_result,
                "snapshot": snapshot,
            }

        return {
            "success": True,
            "message": str(step_result.get("message", "") or f"{primitive_name} observation ok"),
            "primitive": primitive_name,
            "params": normalized_params,
            "observation": observation,
            "primitive_result": step_result,
            "snapshot": snapshot,
        }

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
        from cais_spade_llm.resources.resource_profile import (
            get_resource_profile_for_agent,
            resource_snapshot_set_field,
        )
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
            apply_effects_to_snapshot,
            expand_composite_steps,
            extract_step_output,
            get_resource_bridge_snapshot,
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

        # Eagerly ensure the controller is prewarmed once for the entire
        # macro so individual primitive steps can skip the async-lock check.
        await self._ensure_controller_prewarmed()

        runtime_snapshot = get_resource_bridge_snapshot(self)
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

        primitive_catalog = self._cached_primitive_catalog()
        try:
            primitive_steps = expand_composite_steps(primitive_steps, primitive_catalog)
        except Exception as exc:
            msg = (
                f"Recovery macro '{macro_name}' could not expand composite primitives: {exc}"
            )
            self.logger.error("[Robot] %s", msg)
            return {
                "status": "failed",
                "content": msg,
                "observations": {
                    "macro_name": macro_name,
                    "step_index": -1,
                },
            }
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
        resource_type = str(
            dict(runtime_snapshot.get("resource_core") or {}).get("resource_type")
            or runtime_snapshot.get("resource_type")
            or "resource"
        ).strip().lower() or "resource"
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
                enhanced_msg = step_result.get('message', '')
                
                # Phase 2: Granular Semantic Error Translation
                # Inject real-time spatial context into the error so the LLM understands WHY it failed.
                if primitive in (
                    "move_cartesian", "move_pose", "move_relative", 
                    "move_to_named_pose", "attach_part", "close_gripper"
                ):
                    try:
                        pose_res = await self._execute_primitive("get_current_pose", {})
                        if pose_res and pose_res.get("success") and "pose" in pose_res:
                            pose = pose_res["pose"]
                            enhanced_msg += (
                                f". Context: The robot's end-effector is currently trapped at "
                                f"[x={pose.get('x', 0):.3f}, y={pose.get('y', 0):.3f}, z={pose.get('z', 0):.3f}]."
                            )
                    except Exception as e:
                        self.logger.warning("[Robot] Failed to capture spatial context during error translation: %s", e)

                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' failed at step {step_idx} "
                        f"({primitive}): {enhanced_msg}"
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
                    resource_type=resource_type,
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
            runtime_snapshot = resource_snapshot_set_field(
                runtime_snapshot,
                "current_state",
                out_state,
                profile=get_resource_profile_for_agent(self),
            )
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

    def _cached_primitive_catalog(self) -> list:
        """Return the cached primitive catalog, building it on first access."""
        if self._primitive_catalog_cache is None:
            from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
                build_execution_primitive_catalog,
            )

            self._primitive_catalog_cache = build_execution_primitive_catalog(self)
        return self._primitive_catalog_cache

    async def _execute_primitive(
        self, primitive: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single controller primitive, handling dry_run and simulation modes."""
        if self.execution_mode == "dry_run":
            self.logger.debug("[Robot] dry_run primitive: %s", primitive)
            return {"success": True, "message": f"Simulated: {primitive}"}

        if not self._controller_prewarm_done:
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
        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
            get_resource_bridge_snapshot,
        )

        return get_resource_bridge_snapshot(self)

    # ------------------------------------------------------------------ #
    # Bridge feasibility oracle
    # ------------------------------------------------------------------ #

    def _is_pose_in_workspace(
        self,
        pose: Dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if a Cartesian pose falls within this robot's workspace bounds.

        Returns (is_inside, reason).
        """
        bounds = self.static_capabilities.get("workspace_bounds")
        if not bounds or not isinstance(bounds, dict):
            return True, "no workspace_bounds configured; defaulting to allowed"

        violations: list[str] = []
        for axis in ("x", "y", "z"):
            val = pose.get(axis)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            lo = bounds.get(f"{axis}_min_m")
            hi = bounds.get(f"{axis}_max_m")
            if lo is not None and val < float(lo):
                violations.append(
                    f"{axis}={val:.4f} < {axis}_min_m={float(lo):.4f}"
                )
            if hi is not None and val > float(hi):
                violations.append(
                    f"{axis}={val:.4f} > {axis}_max_m={float(hi):.4f}"
                )

        if violations:
            return False, f"pose outside workspace: {', '.join(violations)}"
        return True, "pose within workspace bounds"

    def bridge_feasibility_oracle(
        self,
        *,
        operation_kind: str,
        part_name: str | None,
        part_context: Dict[str, Any],
        bridge_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Workspace-aware feasibility check for bridge recovery events.

        For clear/home operations: always allowed.
        For pick/place/pick_place: checks the target pose against workspace bounds.
        Falls back to allowed if no workspace_bounds are configured.
        """
        from copy import deepcopy

        op = str(operation_kind or "").strip().lower()
        evidence: Dict[str, Any] = {
            "part_context": deepcopy(part_context),
            "bridge_snapshot": deepcopy(bridge_snapshot),
            "resource_jid": str(getattr(self, "jid", "") or ""),
        }

        # Clear/home: always feasible (robot moves to its own named pose).
        if op in {"clear", "home"}:
            return {
                "allowed": True,
                "reason": f"{op} operation always feasible for own robot",
                "evidence": evidence,
            }

        # For pick/place/pick_place: check target pose against workspace.
        target_pose: Dict[str, Any] | None = None
        if op in {"pick", "pick_place"}:
            # Pick target: observed_pose from part_context
            target_pose = part_context.get("observed_pose")
            if target_pose is None:
                # Try nested under "pose"
                target_pose = part_context.get("pose")
        elif op == "place":
            # Place target: slot_pose from target info
            target_info = part_context.get("target") or {}
            target_pose = target_info.get("slot_pose") or target_info.get("pose")

        if target_pose is None:
            # No pose to check; allow (can't determine infeasibility).
            return {
                "allowed": True,
                "reason": f"no target pose available for {op}; defaulting to allowed",
                "evidence": evidence,
            }

        inside, reason = self._is_pose_in_workspace(target_pose)
        evidence["checked_pose"] = deepcopy(target_pose)
        evidence["workspace_bounds"] = deepcopy(
            self.static_capabilities.get("workspace_bounds") or {}
        )
        return {
            "allowed": inside,
            "reason": reason,
            "evidence": evidence,
        }

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

        interval = 2.0   # progress tick interval
        elapsed = 0.0

        while elapsed < duration:
            await asyncio.sleep(interval)
            elapsed += interval
            self.logger.debug(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        self.logger.info("[%s] Finished: %s", robot, description)
