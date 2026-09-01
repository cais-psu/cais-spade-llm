"""Robot resource agent exposing pick/move/place primitives for assembly."""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (
    build_failure_event,
    failure_context_from_scenario_config,
    load_failure_scenario_config,
)
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.resources.robot import (
    UR5eGazeboController,
    UR5eHardwareController,
    XArm6GazeboController,
    XArm6HardwareController,
)
from cais_spade_llm.resources.robot.robot_profile import ROBOT_PROFILE
from cais_spade_llm.resources.robot.robot_task_runtime import (
    _MANUAL_FUNCTION_EXECUTION_AUTHORITY,
    complete_place_insert_after_move_insert_trial,
    execute_place_insert_move_insert_trial,
    execute_robot_task,
)
from cais_spade_llm.resources.robot.robot_tasks import (
    resolve_robot_task_names,
    robot_task_names,
    robot_task_registry,
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

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        name: str,
        context_only: bool = False,
        **kw: Any,
    ) -> None:
        self.context_only = bool(context_only)
        raw_failure_scenarios = kw.pop("failure_scenarios", None)
        if self.context_only:
            raw_failure_scenarios = None
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
        if self.context_only:
            self.enable_controller_prewarm = False

        # Pop before super().__init__ to avoid unexpected kwarg error.
        self._injected_controller = kw.pop("prewarmed_controller", None)
        requested_function_names = kw.pop("function_names", None)
        if self.context_only:
            resolved_function_names = []
        else:
            resolved_function_names = self.resolve_registered_function_names(
                static_capabilities=kw.get("static_capabilities") or {},
                named_positions=self.named_positions,
                controller_config=self.controller_config,
                requested_names=requested_function_names,
            )
        kw["function_names"] = resolved_function_names
        super().__init__(jid, password, name=name, **kw)
        if self.context_only:
            # ResourceAgent registers its recovery executor by default. A
            # context-only RobotAgent must expose no executable task surface.
            self.executables.clear()
            self._rebuild_tool_schemas()

        self.agent_name = name
        self._held_part: str | None = None

        # Runtime state tracking for replanning context
        self._current_state: str = "idle"  # idle, at_pick, picked, positioned, placed (placed = at destination, part released)
        self._position: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}  # Simulated position
        self._gripper_state: str = "open"
        self._recovery_pose_ref: str | None = None
        # Shared task execution context threaded across task-level functions.
        # `_pick_ctx` remains as a temporary compatibility alias.
        self._task_ctx: dict[str, Any] = {}
        # Use pre-initialized controller (from Gazebo prewarm) if available,
        # to avoid paying the ROS2 init cost again on first task.
        if self.context_only:
            # Phase 5.1 reads logical state and catalog metadata only. Avoiding
            # controller construction prevents ROS, perception, and motion waits.
            self._controller = None
            self._controller_prewarm_done = True
        elif self._injected_controller is not None:
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
        self._robot_motion_lock = threading.Lock()
        self._primitive_catalog_cache: list | None = None

        if self.context_only:
            self.logger.info(
                "RobotAgent '%s' initialized. mode=%s profile=context_only tools=[]",
                name,
                self.execution_mode,
            )
        else:
            self.logger.info(
                ("RobotAgent '%s' initialized. mode=%s tools=%s failure_scenarios=%s"),
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

    @classmethod
    def resolve_registered_function_names(
        cls,
        *,
        static_capabilities: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        controller_config: dict[str, Any] | None = None,
        requested_names: list[str] | tuple[str, ...] | set[str] | str | None = None,
    ) -> list[str]:
        requested_list: list[str] | None
        if isinstance(requested_names, str):
            token = str(requested_names or "").strip().lower()
            requested_list = None if token in {"", "auto"} else [requested_names]
        elif requested_names is None:
            requested_list = None
        else:
            requested_list = [
                str(name).strip() for name in requested_names if str(name or "").strip()
            ]
        return list(
            resolve_robot_task_names(
                static_capabilities=static_capabilities,
                named_positions=named_positions,
                controller_config=controller_config,
                requested_names=requested_list,
            )
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
            and self.execution_mode != "dry_run"
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
    def _pick_ctx(self) -> dict[str, Any]:
        """Compatibility alias for older code paths that still reference `_pick_ctx`."""
        return self._task_ctx

    @_pick_ctx.setter
    def _pick_ctx(self, value: dict[str, Any]) -> None:
        self._task_ctx = value if isinstance(value, dict) else {}

    async def _execute_registered_robot_task(
        self,
        task_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a registry-backed robot task through the generic DSL runtime."""
        if not self._robot_motion_lock.acquire(blocking=False):
            return {
                "status": "blocked",
                "content": f"{self.agent_name} is already executing a robot task.",
            }
        try:
            return await execute_robot_task(self, task_name, **kwargs)
        finally:
            self._robot_motion_lock.release()

    async def _execute_registered_robot_task_for_manual_function_execution(
        self,
        task_name: str,
        pre_execute: Any = None,
        post_staging_acceptance: Any = None,
        /,
        *,
        operator_confirmed_held_part: bool = False,
        operator_confirmed_held_part_handoff: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute one Control-page function without the assembly sequence token.

        Args:
            task_name: Exact registered Robot Function identifier.
            pre_execute: Optional final bridge-owned readiness callback.
            post_staging_acceptance: Optional assembly-board acceptance callback.
            operator_confirmed_held_part: Whether the operator confirms that the
                exact selected ``part_name`` is physically held for an independent
                ``place_approach`` commissioning run. This value is passed
                positionally to the runtime and is not a public task argument.
            operator_confirmed_held_part_handoff: Bridge-validated complete held-part
                handoff derived from the confirmed ``pick_approach.descend``
                recording.
            **kwargs: Arguments declared by the selected Robot Function.

        Returns:
            Task completion, block, or failure payload.
        """
        if not self._robot_motion_lock.acquire(blocking=False):
            return {
                "status": "blocked",
                "content": f"{self.agent_name} is already executing a robot task.",
            }
        controller = getattr(self, "_controller", None)
        callback_name = "_assembly_board_v1_post_staging_accept_callback"
        install_post_staging_acceptance = bool(
            callable(post_staging_acceptance)
            and task_name == "place_approach"
            and str(kwargs.get("destination_location") or "").strip()
            == "assembly_board-v1"
            and controller is not None
        )
        callback_was_set = bool(
            install_post_staging_acceptance and hasattr(controller, callback_name)
        )
        previous_callback = (
            getattr(controller, callback_name, None)
            if install_post_staging_acceptance
            else None
        )
        if install_post_staging_acceptance:
            setattr(controller, callback_name, post_staging_acceptance)
        try:
            if callable(pre_execute):
                pre_execute_error = str(pre_execute() or "").strip()
                if pre_execute_error:
                    return {
                        "status": "blocked",
                        "content": pre_execute_error,
                        "manual_pre_execute_blocked": True,
                    }
            if (
                operator_confirmed_held_part
                or operator_confirmed_held_part_handoff is not None
            ):
                return await execute_robot_task(
                    self,
                    task_name,
                    _MANUAL_FUNCTION_EXECUTION_AUTHORITY,
                    operator_confirmed_held_part,
                    operator_confirmed_held_part_handoff,
                    **kwargs,
                )
            return await execute_robot_task(
                self,
                task_name,
                _MANUAL_FUNCTION_EXECUTION_AUTHORITY,
                **kwargs,
            )
        finally:
            if install_post_staging_acceptance:
                if callback_was_set:
                    setattr(controller, callback_name, previous_callback)
                else:
                    try:
                        delattr(controller, callback_name)
                    except AttributeError:
                        pass
            self._robot_motion_lock.release()

    async def _execute_place_insert_move_insert_trial(
        self,
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        trial_id: str,
    ) -> dict[str, Any]:
        """Run only the internal move_insert step for supervised qualification."""
        if not self._robot_motion_lock.acquire(blocking=False):
            return {
                "status": "blocked",
                "content": f"{self.agent_name} is already executing a robot task.",
                "trial_id": trial_id,
                "motion_settled": True,
                "dispatch_attempted": False,
                "move_insert_result": {
                    "success": False,
                    "trial_id": trial_id,
                    "state_uncertain": False,
                    "motion_settled": True,
                    "dispatch_attempted": False,
                },
            }
        try:
            if callable(pre_execute):
                pre_execute_error = str(pre_execute() or "").strip()
                if pre_execute_error:
                    return {
                        "status": "blocked",
                        "content": pre_execute_error,
                        "manual_pre_execute_blocked": True,
                        "trial_id": trial_id,
                        "motion_settled": True,
                        "dispatch_attempted": False,
                        "move_insert_result": {
                            "success": False,
                            "trial_id": trial_id,
                            "state_uncertain": False,
                            "motion_settled": True,
                            "dispatch_attempted": False,
                        },
                    }
            return await execute_place_insert_move_insert_trial(
                self,
                destination_location=destination_location,
                part_name=part_name,
                trial_id=trial_id,
            )
        finally:
            self._robot_motion_lock.release()

    async def _complete_place_insert_after_move_insert_trial(
        self,
        pre_execute: Any = None,
        /,
        *,
        destination_location: str,
        part_name: str,
        expected_move_insert_result_sha256: str,
    ) -> dict[str, Any]:
        """Release and lift once after a reviewed move_insert trial."""
        if not self._robot_motion_lock.acquire(blocking=False):
            return {
                "status": "blocked",
                "content": f"{self.agent_name} is already executing a robot task.",
            }
        try:
            if callable(pre_execute):
                pre_execute_error = str(pre_execute() or "").strip()
                if pre_execute_error:
                    return {
                        "status": "blocked",
                        "content": pre_execute_error,
                        "manual_pre_execute_blocked": True,
                    }
            return await complete_place_insert_after_move_insert_trial(
                self,
                destination_location=destination_location,
                part_name=part_name,
                expected_move_insert_result_sha256=(
                    expected_move_insert_result_sha256
                ),
            )
        finally:
            self._robot_motion_lock.release()

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
        call_args: dict[str, Any] | None = None,
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
    def _resolve_effect_ref(ref: str, effect_context: dict[str, Any]) -> Any:
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

    def _resolve_effect_value(self, value: Any, effect_context: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            if set(value.keys()) == {"ref"}:
                return self._resolve_effect_ref(str(value.get("ref") or ""), effect_context)
            return {
                key: self._resolve_effect_value(item, effect_context) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_effect_value(item, effect_context) for item in value]
        return deepcopy(value)

    def _scenario_error_result(
        self,
        match: dict[str, Any],
        *,
        message: str,
        effect_context: dict[str, Any],
    ) -> dict[str, Any]:
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
        match: dict[str, Any],
        effect: dict[str, Any],
        effect_context: dict[str, Any],
    ) -> dict[str, Any] | None:
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
        call_args: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
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

        effect_context: dict[str, Any] = {
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
        execution_mode = str(self.execution_mode or "").strip().lower()
        use_gazebo_controller = execution_mode == "simulation"
        try:
            if robot_scope.startswith("ur5e"):
                if use_gazebo_controller:
                    return UR5eGazeboController(
                        trajectory_topic=_UR5E_GAZEBO_ARM_TRAJECTORY_TOPIC,
                        controller_config=self.controller_config,
                        named_positions=self.named_positions,
                        execution_mode=self.execution_mode,
                    )
                return UR5eHardwareController(
                    controller_config=self.controller_config,
                    named_positions=self.named_positions,
                    execution_mode=self.execution_mode,
                )
            if robot_scope.startswith("xarm6"):
                controller_cls = (
                    XArm6GazeboController if use_gazebo_controller else XArm6HardwareController
                )
                return controller_cls(
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
        observations: dict[str, Any] | None = None,
        failure_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        detail = str(message or "task failed")
        self.logger.error("[Robot] %s failed: %s", step, detail)
        failure_observations: dict[str, Any] = {"step": step}
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
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a controller-only helper that is not exposed as a recovery primitive."""
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

    async def _execute_taught_function_step(
        self,
        *,
        function_name: str,
        taught_function_name: str,
        step_name: str,
    ) -> dict[str, Any]:
        """Execute one user-taught function step through the hardware controller."""
        if self.execution_mode == "dry_run":
            return {"success": True, "message": f"Simulated taught function step: {step_name}"}

        await self._ensure_controller_prewarmed()

        if self._controller is None:
            return {"success": False, "message": "controller is not initialized"}

        method = getattr(self._controller, "replay_taught_function_step", None)
        if not callable(method):
            return {
                "success": False,
                "message": "controller missing taught function replay support",
            }

        try:
            result = await asyncio.to_thread(
                method,
                function_name,
                taught_function_name,
                step_name,
                storage_source="hardware",
            )
            if isinstance(result, dict):
                return dict(result)
            return {
                "success": bool(result),
                "message": f"taught function step {step_name} {'ok' if result else 'failed'}",
            }
        except Exception as exc:
            self.logger.exception("[Robot] Taught function step '%s' execution failed", step_name)
            return {
                "success": False,
                "message": f"taught function step exception: {type(exc).__name__}: {exc}",
            }

    def _log_step(self, step: str, message: str, **fields: Any) -> None:
        details = ", ".join(f"{key}={value}" for key, value in fields.items())
        suffix = f" ({details})" if details else ""
        self.logger.info("[Robot] %s: %s%s", step, message, suffix)

    # ------------------------------------------------------------------ #
    # Recovery-only recovery macro executor
    # ------------------------------------------------------------------ #

    # Controller primitives available for recovery macro steps.
    _RECOVERY_PRIMITIVES = frozenset(
        {
            "move_cartesian",
            "move_pose",
            "move_relative",
            "move_to_named_pose",
            "delay",
            "grasp_part",
            "release_part",
            "open_gripper",
            "close_gripper",
            "detect_parts",
            "localize_assembly_board_v1",
            "compute_pick_targets",
            "compute_place_targets",
            "attach_part",
            "detach_part",
            "snap_part_to_slot",
            "get_current_pose",
        }
    )
    _RECOVERY_OBSERVATION_PRIMITIVES = frozenset(
        {
            "detect_parts",
            "compute_pick_targets",
            "compute_place_targets",
            "get_current_pose",
        }
    )

    async def execute_recovery_observation(
        self,
        primitive: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one planner-approved observation/generation primitive."""
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_primitives import (
            extract_step_output,
        )

        primitive_name = str(primitive or "").strip()
        normalized_params = dict(params or {})
        if primitive_name not in self._RECOVERY_OBSERVATION_PRIMITIVES:
            return {
                "success": False,
                "message": (
                    f"recovery observation primitive '{primitive_name}' is not allowed; "
                    f"allowed={sorted(self._RECOVERY_OBSERVATION_PRIMITIVES)}"
                ),
            }

        step_result = await self._execute_primitive(primitive_name, normalized_params)
        snapshot = self.get_recovery_snapshot()
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

    def _inject_current_pick_ctx_for_place_targets(
        self,
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Thread the current held-part pick context into generated place-target steps."""
        normalized = dict(params or {})
        if str(primitive or "").strip() != "compute_place_targets":
            return normalized
        if "pick_ctx" in normalized:
            return normalized

        part_name = str(normalized.get("part_name") or "").strip()
        held_part = str(self._held_part or "").strip()
        if not part_name or not held_part or part_name != held_part:
            return normalized
        if not isinstance(self._task_ctx, dict) or not self._task_ctx:
            return normalized
        ctx_part = str(self._task_ctx.get("part_name") or "").strip()
        if ctx_part and ctx_part != part_name:
            return normalized

        normalized["pick_ctx"] = deepcopy(self._task_ctx)
        return normalized

    @staticmethod
    def _normalized_xyz_pose(value: Any) -> dict[str, float] | None:
        if not isinstance(value, dict) or not {"x", "y", "z"} <= set(value.keys()):
            return None
        try:
            return {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "z": float(value["z"]),
            }
        except (TypeError, ValueError):
            return None

    def _inject_observed_pose_for_pick_targets(
        self,
        primitive: str,
        params: dict[str, Any],
        macro_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Thread product-provided observed pose into generated recovery pick-target steps."""
        normalized = dict(params or {})
        if str(primitive or "").strip() != "compute_pick_targets":
            return normalized
        if isinstance(normalized.get("target_pose"), dict):
            return normalized

        source_location = str(
            macro_context.get("origin_resource_location")
            or macro_context.get("source_location")
            or ""
        ).strip()
        if source_location != "observed_pose" and not source_location.endswith("_observed_pose"):
            return normalized

        part_name = str(normalized.get("part_name") or "").strip()
        context_part = str(macro_context.get("part_name") or "").strip()
        if context_part and part_name and context_part != part_name:
            return normalized

        observed_pose = self._normalized_xyz_pose(macro_context.get("observed_pose"))
        if observed_pose is None:
            observed_pose = self._normalized_xyz_pose(macro_context.get("target_pose"))
        if observed_pose is None:
            return normalized

        normalized["target_pose"] = observed_pose
        normalized["target_pose_source"] = "observed_pose"
        # When the recovery already carries a grounded observed pose, use it directly.
        # Live perception remains available for cases that do not have target_pose.
        normalized["prefer_live_detection"] = False
        normalized["use_global_min_pick_tcp_z"] = False
        normalized["apply_pick_z_adjustments"] = False
        normalized.setdefault("ignore_current_height_for_travel_z", True)

        part_geometry = macro_context.get("part_geometry")
        if isinstance(part_geometry, dict) and "product_geometry" not in normalized:
            normalized["product_geometry"] = deepcopy(part_geometry)

        if "approach_height_override_m" not in normalized:
            motion_config = getattr(self, "motion_config", {}) or {}
            try:
                recovery_approach_height = float(
                    motion_config.get("recovery_observed_pick_approach_height_m")
                )
            except (TypeError, ValueError):
                recovery_approach_height = None
            if recovery_approach_height is not None and recovery_approach_height > 0.0:
                normalized["approach_height_override_m"] = recovery_approach_height

        if "surface_clearance_override_m" not in normalized:
            motion_config = getattr(self, "motion_config", {}) or {}
            try:
                surface_clearance = float(
                    motion_config.get("recovery_observed_pick_surface_clearance_m", 0.005)
                )
            except (TypeError, ValueError):
                surface_clearance = 0.005
            if surface_clearance > 0.0:
                normalized["surface_clearance_override_m"] = surface_clearance

        return normalized

    def _normalize_simulation_release_part_params(
        self,
        primitive: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make simulated Gazebo releases tolerate detach flakiness after the gripper opens."""
        normalized = dict(params or {})
        if str(primitive or "").strip() != "release_part":
            return normalized
        if "assume_released_if_open" in normalized:
            return normalized
        if str(getattr(self, "execution_mode", "") or "").strip().lower() != "simulation":
            return normalized
        normalized["assume_released_if_open"] = True
        return normalized

    def _remember_pick_targets_from_macro_step(
        self,
        primitive: str,
        params: dict[str, Any],
        step_result: dict[str, Any],
        macro_context: dict[str, Any],
    ) -> None:
        """Keep generated macro pick context current for subsequent place targets."""
        if str(primitive or "").strip() != "compute_pick_targets":
            return
        if not step_result.get("success"):
            return

        def _float_or_none(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        part_name = str(step_result.get("part_name") or params.get("part_name") or "").strip()
        if not part_name:
            return

        context: dict[str, Any] = {"part_name": part_name}
        for key in (
            "model_name",
            "tx",
            "ty",
            "tz",
            "pick_z",
            "travel_z",
            "part_height",
            "tcp_offset_z",
            "pick_tcp_z",
            "pick_tcp_z_offset_from_table_m",
            "start_x",
            "start_y",
            "start_z",
            "target_pose_source",
            "surface_clearance_m",
            "pick_z_adjustment_m",
            "pick_tool0_z_adjustment_m",
            "apply_pick_z_adjustments",
            "source_stl",
            "source_stl_sha256",
            "hub_up",
            "hub_diameter_m",
            "hub_height_m",
            "tooth_diameter_m",
            "tooth_height_m",
            "grasp_width_m",
            "tooth_clearance_m",
            "minimum_hub_overlap_m",
            "finger_tooth_clearance_m",
            "finger_hub_overlap_m",
            "open_gripper_position",
            "mg_gripper_close_position",
            "open_inner_pad_lower_z_from_tcp_m",
            "open_inner_pad_upper_z_from_tcp_m",
            "closed_inner_pad_lower_z_from_tcp_m",
            "closed_inner_pad_upper_z_from_tcp_m",
            "predicted_closing_z_displacement_m",
            "gripper_close_position",
        ):
            if key in step_result:
                context[key] = deepcopy(step_result[key])

        origin_resource_location = str(
            macro_context.get("origin_resource_location")
            or macro_context.get("source_location")
            or params.get("origin_resource_location")
            or params.get("source_location")
            or ""
        ).strip()
        if origin_resource_location:
            context["origin_resource_location"] = origin_resource_location

        tx = _float_or_none(step_result.get("tx"))
        ty = _float_or_none(step_result.get("ty"))
        tz = _float_or_none(step_result.get("tz"))
        if tx is not None and ty is not None and tz is not None:
            context["origin_pose"] = {"x": tx, "y": ty, "z": tz}

        self._task_ctx = context

    async def _stabilize_recovery_release_if_needed(
        self,
        primitive: str,
        params: dict[str, Any],
        step_result: dict[str, Any],
        event_facts: dict[str, Any],
    ) -> None:
        """Snap direct recovery releases into assembly slot depth when geometry says inserted."""
        if str(primitive or "").strip() != "release_part":
            return
        if self.execution_mode != "simulation":
            return

        part_name = str(
            params.get("part_name") or params.get("model_name") or self._held_part or ""
        ).strip()
        if not part_name:
            return

        place_targets = dict(event_facts.get("place_targets") or {})
        place_target = place_targets.get(part_name)
        if not isinstance(place_target, dict):
            return

        target_reference = dict(place_target.get("target_reference") or {})
        if str(target_reference.get("target_point") or "").strip() != "inserted_part_origin":
            return

        model_name = str(
            params.get("model_name")
            or place_target.get("model_name")
            or dict(getattr(self, "_task_ctx", {}) or {}).get("model_name")
            or ""
        ).strip()
        if not model_name:
            return
        release_mode = str((step_result or {}).get("release_mode") or "").strip()
        if release_mode == "assumed_open_after_detach_timeout":
            self.logger.warning(
                "[Robot] skipping recovery release snap_part_to_slot for %s because release_part used release_mode=%s",
                model_name,
                release_mode,
            )
            return

        snap = await self._execute_primitive(
            "snap_part_to_slot",
            {
                "model_name": model_name,
                "slot_x": place_target.get("slot_x", 0.0),
                "slot_y": place_target.get("slot_y", 0.0),
                "part_height": place_target.get("part_height", 0.08),
                "board_top_z": place_target.get("board_top_z", 1.025),
                "part_origin_z": place_target.get("place_part_origin_z"),
            },
        )
        if not snap.get("success"):
            self.logger.warning(
                "[Robot] recovery release snap_part_to_slot failed for %s: %s",
                model_name,
                snap.get("message"),
            )

    async def execute_recovery_macro(
        self,
        macro_name: str,
        primitive_steps: list,
        *,
        expected_start_state: str | None = None,
        expected_snapshot: dict[str, Any] | None = None,
        product_jid: str | None = None,
        task_id: str | None = None,
        in_state: str | None = None,
        out_state: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        """Execute a recovery-generated recovery macro as an ordered primitive sequence.

        This method is registered in self.executables for runtime dispatch
        but is excluded from function_names and the shared tools catalog.
        It is only callable through recovery-approved recovery macro tasks.
        """
        from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_primitives import (
            apply_effects_to_snapshot,
            event_fact_key_for_primitive,
            expand_composite_steps,
            extract_step_output,
            resolve_param_refs,
            snapshot_matches_expected,
            validate_and_project_steps,
        )
        from cais_spade_llm.resources.resource_primitives import (
            get_resource_recovery_snapshot,
            sync_agent_from_recovery_snapshot,
        )
        from cais_spade_llm.resources.resource_profile import (
            get_resource_profile_for_agent,
            resource_snapshot_set_field,
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

        runtime_snapshot = get_resource_recovery_snapshot(self)
        if expected_snapshot:
            matches, mismatch_message = snapshot_matches_expected(
                runtime_snapshot, expected_snapshot
            )
            if not matches:
                msg = (
                    f"Recovery macro '{macro_name}' expected snapshot mismatch: {mismatch_message}"
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
            msg = f"Recovery macro '{macro_name}' could not expand composite primitives: {exc}"
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
        results: list[dict[str, Any]] = []
        event_facts: dict[str, Any] = {}
        resource_type = (
            str(
                dict(runtime_snapshot.get("resource_core") or {}).get("resource_type")
                or runtime_snapshot.get("resource_type")
                or "resource"
            )
            .strip()
            .lower()
            or "resource"
        )
        for step_idx, step in enumerate(primitive_steps):
            primitive = step.get("primitive", "") if isinstance(step, dict) else ""
            raw_params = step.get("params", {}) if isinstance(step, dict) else {}

            if primitive not in self._RECOVERY_PRIMITIVES:
                msg = f"Unknown primitive '{primitive}' at step {step_idx} in macro '{macro_name}'"
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
                    event_facts=event_facts,
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
                        "event_facts": deepcopy(event_facts),
                    },
                }

            params = self._inject_observed_pose_for_pick_targets(primitive, params, context)
            params = self._inject_current_pick_ctx_for_place_targets(primitive, params)
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
                enhanced_msg = step_result.get("message", "")

                # Phase 2: Granular Semantic Error Translation
                # Inject real-time spatial context into the error so the LLM understands WHY it failed.
                if primitive in (
                    "move_cartesian",
                    "move_pose",
                    "move_relative",
                    "move_to_named_pose",
                    "attach_part",
                    "close_gripper",
                    "grasp_part",
                    "release_part",
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
                        self.logger.warning(
                            "[Robot] Failed to capture spatial context during error translation: %s",
                            e,
                        )

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

            self._remember_pick_targets_from_macro_step(
                primitive,
                params,
                step_result,
                context,
            )
            await self._stabilize_recovery_release_if_needed(
                primitive,
                params,
                step_result,
                event_facts,
            )

            primitive_meta = primitive_meta_by_name.get(primitive)
            if primitive_meta is not None:
                runtime_snapshot = apply_effects_to_snapshot(
                    {**dict(step), "params": params},
                    primitive_meta,
                    runtime_snapshot,
                )
                sync_agent_from_recovery_snapshot(self, runtime_snapshot)

            event_fact_key, event_fact_error = event_fact_key_for_primitive(
                primitive=primitive,
                params=params,
                resource_type=resource_type,
            )
            if event_fact_error:
                return {
                    "status": "failed",
                    "content": (
                        f"Macro '{macro_name}' could not publish event facts at step {step_idx} "
                        f"({primitive}): {event_fact_error}"
                    ),
                    "observations": {
                        "macro_name": macro_name,
                        "step_index": step_idx,
                        "primitive": primitive,
                        "primitive_result": step_result,
                        "completed_steps": len(results),
                    },
                }
            if event_fact_key:
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
                            f"Macro '{macro_name}' could not publish event facts at step {step_idx} "
                            f"({primitive}): {output_error}"
                        ),
                        "observations": {
                            "macro_name": macro_name,
                            "step_index": step_idx,
                            "primitive": primitive,
                            "primitive_result": step_result,
                            "event_fact_path": f"event_facts.{event_fact_key}",
                            "completed_steps": len(results),
                        },
                    }
                target = event_facts
                tokens = [token for token in str(event_fact_key).split(".") if token]
                for token in tokens[:-1]:
                    child = target.get(token)
                    if not isinstance(child, dict):
                        child = {}
                        target[token] = child
                    target = child
                if tokens:
                    target[tokens[-1]] = deepcopy(step_output)

        # All steps succeeded. Update logical state if out_state specified.
        if out_state:
            runtime_snapshot = resource_snapshot_set_field(
                runtime_snapshot,
                "current_state",
                out_state,
                profile=get_resource_profile_for_agent(self),
            )
            sync_agent_from_recovery_snapshot(self, runtime_snapshot)
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
                "event_facts": deepcopy(event_facts),
            },
        }

    def recovery_synthesis_primitive_catalog(self) -> list[dict[str, Any]]:
        """Return the robot-owned LLM-facing primitive catalog."""
        if self._recovery_synthesis_primitive_catalog_cache is None:
            from cais_spade_llm.resources.robot.robot_primitives import (
                build_robot_synthesis_primitive_catalog,
            )

            self._recovery_synthesis_primitive_catalog_cache = (
                build_robot_synthesis_primitive_catalog(
                    primitive_catalog=self.recovery_execution_primitive_catalog()
                )
            )
        return deepcopy(self._recovery_synthesis_primitive_catalog_cache)

    def recovery_des_model(
        self,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the robot's private task-level recovery DES model."""
        from cais_spade_llm.resources.resource_primitives import (
            build_recovery_des_model,
        )
        from cais_spade_llm.resources.robot.robot_tasks import (
            robot_recovery_des_descriptor,
        )

        live_snapshot = deepcopy(snapshot or self.get_recovery_snapshot())
        task_names = self.resolve_registered_function_names(
            static_capabilities=self.static_capabilities,
            named_positions=self.named_positions,
            controller_config=self.controller_config,
        )
        raw_descriptor = robot_recovery_des_descriptor(
            resource_jid=str(self.jid),
            snapshot=live_snapshot,
            task_names=task_names,
            reachable_locations=list(
                live_snapshot.get("reachable_locations")
                or self.static_capabilities.get("reachability")
                or []
            ),
            named_poses=list(
                live_snapshot.get("named_poses") or self.named_positions or []
            ),
            marked_state_conditions=list(
                self.static_capabilities.get("recovery_marked_state_conditions")
                or []
            ),
        )
        return build_recovery_des_model(
            self,
            snapshot=live_snapshot,
            descriptor=raw_descriptor,
        )

    def _cached_primitive_catalog(self) -> list:
        """Return the cached primitive catalog, building it on first access."""
        return self.recovery_execution_primitive_catalog()

    def _emit_physical_part_ownership_event(
        self,
        primitive: str,
        params: dict[str, Any],
    ) -> None:
        """Emit a mirror-only event after completed UR5e hardware custody changes."""
        if self.execution_mode != "physical" or str(self.name).strip().lower() != "ur5e":
            return
        primitive_name = str(primitive or "").strip()
        if primitive_name not in {"grasp_part", "release_part"}:
            return
        part_name = str(
            params.get("part_name") or params.get("model_name") or self._held_part or ""
        ).strip()
        model_map = {
            "SG": "gear_small",
            "MG": "gear_medium",
            "LG": "gear_large",
            "SCP": "circ_pin_small",
            "MCP": "circ_pin_medium",
            "LCP": "circ_pin_large",
        }
        model_name = str(params.get("model_name") or model_map.get(part_name) or "").strip()
        if model_name not in model_map.values():
            return
        event = {
            "sequence": str(time.time_ns()),
            "occurred_at": time.time(),
            "action": "held" if primitive_name == "grasp_part" else "released",
            "part_name": part_name,
            "model_name": model_name,
            "source": "hardware_ur5e",
        }
        path = Path("/tmp/cais_physical_part_ownership.json")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(event, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            self.logger.warning(
                "[Robot] hardware action completed but Gazebo ownership event failed: %s",
                exc,
            )

    async def _execute_primitive(self, primitive: str, params: dict[str, Any]) -> dict[str, Any]:
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

        normalized_params = self._normalize_simulation_release_part_params(primitive, params)

        try:
            result = await asyncio.to_thread(method, **normalized_params)
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
                if primitive == "detect_parts" and not result and last_failure:
                    return {
                        "success": False,
                        "message": last_failure,
                    }
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
                release_mode = str(normalized.get("release_mode") or "").strip()
                if (
                    str(primitive or "").strip() == "release_part"
                    and normalized.get("success")
                    and release_mode
                ):
                    self.logger.info(
                        "[Robot] release_part completed using release_mode=%s",
                        release_mode,
                    )
                if normalized.get("success"):
                    self._emit_physical_part_ownership_event(primitive, normalized_params)
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
    def get_recovery_snapshot(self) -> dict[str, Any]:
        """Return the current primitive-level recovery snapshot for this robot."""
        from cais_spade_llm.resources.resource_primitives import (
            get_resource_recovery_snapshot,
        )

        return get_resource_recovery_snapshot(self)

    @staticmethod
    def recovery_physical_validation_snapshot(
        *,
        live_snapshot: dict[str, Any],
        physical_input: dict[str, Any],
        recovery_des_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build RobotAgent-owned evidence for projected outline validation.

        The first outline transition retains the fresh live gripper evidence.
        For later unexecuted transitions, ``held_part`` is the authoritative
        projected custody fact and RobotAgent derives its private gripper
        evidence without requiring PA or the LLM to author that mechanism.
        """
        validation_snapshot = ResourceAgent.recovery_physical_validation_snapshot(
            live_snapshot=live_snapshot,
            physical_input=physical_input,
            recovery_des_model=recovery_des_model,
        )
        projected_snapshot = physical_input.get("projected_recovery_snapshot")
        if (
            physical_input.get("use_projected_recovery_snapshot") is True
            and isinstance(projected_snapshot, dict)
            and "held_part" in projected_snapshot
        ):
            validation_snapshot["gripper_state"] = (
                "closed" if projected_snapshot.get("held_part") else "open"
            )
        return validation_snapshot

    # ------------------------------------------------------------------ #
    # Recovery physical feasibility check
    # ------------------------------------------------------------------ #

    def _is_pose_in_workspace(
        self,
        pose: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if a Cartesian pose falls within this robot's workspace bounds.

        Returns (is_inside, reason).
        """
        bounds = self.static_capabilities.get("workspace_bounds")
        if not bounds or not isinstance(bounds, dict):
            return False, "workspace_bounds capability data is unavailable"

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
                violations.append(f"{axis}={val:.4f} < {axis}_min_m={float(lo):.4f}")
            if hi is not None and val > float(hi):
                violations.append(f"{axis}={val:.4f} > {axis}_max_m={float(hi):.4f}")

        if violations:
            return False, f"pose outside workspace: {', '.join(violations)}"
        return True, "pose within workspace bounds"

    @staticmethod
    def _recovery_pose_gripper_reach_error(
        pose: dict[str, Any],
        gripper_reach: dict[str, Any],
    ) -> str:
        """Return a configured world-frame gripper reachability error."""
        if not isinstance(gripper_reach, dict) or not gripper_reach:
            return "gripper_reach capability data is unavailable"
        if str(gripper_reach.get("frame") or "world").strip() != "world":
            return "gripper_reach frame must be world"
        origin_pose = dict(gripper_reach.get("origin_pose") or {})
        try:
            x = float(pose["x"])
            y = float(pose["y"])
            z = float(pose["z"])
            origin_x = float(origin_pose["x"])
            origin_y = float(origin_pose["y"])
            max_xy_radius_m = float(gripper_reach["max_xy_radius_m"])
            z_min_m = float(gripper_reach["z_min_m"])
            z_max_m = float(gripper_reach["z_max_m"])
            tolerance_m = float(gripper_reach.get("tolerance_m") or 0.0)
        except (KeyError, TypeError, ValueError):
            return "gripper_reach capability data is incomplete"
        values = (
            x,
            y,
            z,
            origin_x,
            origin_y,
            max_xy_radius_m,
            z_min_m,
            z_max_m,
            tolerance_m,
        )
        if not all(math.isfinite(value) for value in values):
            return "gripper_reach capability data is not finite"
        xy_radius_m = math.hypot(x - origin_x, y - origin_y)
        if xy_radius_m > max_xy_radius_m + tolerance_m:
            return (
                "pose is outside gripper_reach: "
                f"xy_radius={xy_radius_m:.4f} m, max={max_xy_radius_m:.4f} m"
            )
        if z < z_min_m - tolerance_m or z > z_max_m + tolerance_m:
            return (
                "pose is outside gripper_reach: "
                f"z={z:.4f} m, range=[{z_min_m:.4f}, {z_max_m:.4f}] m"
            )
        return ""

    def _configured_pick_place_recovery_feasibility(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        *,
        function_name: str,
        part_name: str,
        part_context: dict[str, Any],
        recovery_snapshot: dict[str, Any],
        grounded_action: dict[str, Any],
        source_ref: dict[str, Any],
        target_info: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Validate configured pick/place behavior without taught-function evidence."""
        if function_name not in {"pick_approach", "place_approach"}:
            return None

        resource_jid = str(getattr(self, "jid", "") or "")

        def _result(
            status: str,
            constraint_code: str,
            reason: str,
            *,
            guard_kind: str,
            extra_evidence: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {
                "allowed": False,
                "feasibility_status": status,
                "constraint_code": constraint_code,
                "guard": {
                    "kind": guard_kind,
                    "resource_jid": resource_jid,
                    "part_name": part_name,
                    "function_name": function_name,
                },
                "reason": reason,
                "evidence": {
                    **evidence,
                    "recording_history_used": False,
                    **deepcopy(extra_evidence or {}),
                },
            }

        if not part_name:
            return _result(
                "NEEDS_CONTEXT",
                "source_reference_unavailable",
                f"{function_name} requires an exact part_name",
                guard_kind="part_name_unavailable",
            )

        held_part = str(recovery_snapshot.get("held_part") or "").strip()
        current_holder = str(
            part_context.get("current_holder_resource_jid") or ""
        ).strip()
        if current_holder and current_holder != resource_jid:
            return _result(
                "INFEASIBLE",
                "holder_conflict",
                f"part '{part_name}' is currently held by '{current_holder}'",
                guard_kind="part_held_by_other",
                extra_evidence={"current_holder_resource_jid": current_holder},
            )
        if function_name == "pick_approach" and held_part:
            return _result(
                "INFEASIBLE",
                "holder_conflict",
                f"resource already holds '{held_part}'",
                guard_kind="resource_holds_part",
                extra_evidence={"conflicting_part": held_part},
            )
        if function_name == "place_approach" and held_part != part_name:
            return _result(
                "INFEASIBLE",
                "required_part_not_held",
                f"place_approach requires this robot to hold '{part_name}'",
                guard_kind="required_part_not_held",
                extra_evidence={"held_part": held_part or None},
            )

        static_capabilities = dict(getattr(self, "static_capabilities", {}) or {})
        if static_capabilities.get("supports_manipulator_pick_place") is not True:
            return _result(
                "INFEASIBLE",
                "unsupported_resource_target",
                "supports_manipulator_pick_place is not exposed as true",
                guard_kind="manipulator_pick_place_unavailable",
            )

        executables = getattr(self, "executables", None)
        configured_function_names = {
            str(token).strip()
            for token in (
                getattr(self, "function_names", None)
                or recovery_snapshot.get("function_names")
                or []
            )
            if str(token or "").strip()
        }
        task_enabled = bool(
            isinstance(executables, dict)
            and function_name in executables
            and callable(executables[function_name])
        ) or function_name in configured_function_names
        if not task_enabled:
            return _result(
                "INFEASIBLE",
                "unsupported_resource_target",
                f"configured robot does not expose enabled task {function_name}",
                guard_kind="robot_task_unavailable",
            )

        controller = getattr(self, "_controller", None)
        controller_config = dict(
            getattr(self, "controller_config", {})
            or getattr(controller, "controller_config", {})
            or {}
        )
        move_group = dict(controller_config.get("move_group") or {})
        frame_id = str(
            getattr(controller, "frame_id", "")
            or move_group.get("frame_id")
        ).strip()
        ee_link = str(
            getattr(controller, "ee_link", "")
            or move_group.get("ee_link")
        ).strip()
        tcp_link = str(
            getattr(controller, "tcp_link", "")
            or move_group.get("tcp_link")
        ).strip()
        services = dict(controller_config.get("services") or {})
        gripper_config = dict(controller_config.get("gripper") or {})
        cartesian_behavior = bool(
            callable(getattr(controller, "move_cartesian", None))
            or controller_config.get("hardware_cartesian_service")
            or services.get("cartesian_path")
        )
        pose_feedback_behavior = bool(
            callable(getattr(controller, "get_current_pose", None))
            or controller_config.get("joint_state_topics")
        )
        gripper_behavior = bool(
            gripper_config
            and (
                gripper_config.get("action")
                or gripper_config.get("hardware_action")
                or gripper_config.get("hardware_service")
                or gripper_config.get("topic")
            )
        )
        perception_behavior = bool(
            callable(getattr(controller, "detect_parts", None))
            or services.get("detect_all")
        )
        configuration_evidence = {
            "frame_id": frame_id,
            "ee_link": ee_link,
            "tcp_link": tcp_link,
            "cartesian_behavior": cartesian_behavior,
            "pose_feedback_behavior": pose_feedback_behavior,
            "gripper_behavior": gripper_behavior,
            "perception_behavior": perception_behavior,
        }
        if (
            frame_id != "world"
            or not ee_link
            or not tcp_link
            or not cartesian_behavior
            or not pose_feedback_behavior
            or not gripper_behavior
        ):
            return _result(
                "INFEASIBLE",
                "resource_validation_unavailable",
                "configured Cartesian, pose-feedback, TCP, or gripper behavior is unavailable",
                guard_kind="controller_behavior_unavailable",
                extra_evidence=configuration_evidence,
            )
        if function_name == "pick_approach" and not perception_behavior:
            return _result(
                "INFEASIBLE",
                "resource_validation_unavailable",
                "configured part perception behavior is unavailable",
                guard_kind="perception_unavailable",
                extra_evidence=configuration_evidence,
            )
        for readiness_field in (
            "controller_ready",
            "tf_ready",
            "tcp_ready",
            "perception_ready" if function_name == "pick_approach" else "destination_localization_ready",
        ):
            readiness_value = recovery_snapshot.get(readiness_field)
            if readiness_value is None:
                return _result(
                    "NEEDS_CONTEXT",
                    "resource_validation_unavailable",
                    f"current {readiness_field} evidence is unavailable",
                    guard_kind="runtime_behavior_evidence_unavailable",
                    extra_evidence={readiness_field: None, **configuration_evidence},
                )
            if readiness_value is not True:
                return _result(
                    "INFEASIBLE",
                    "resource_unavailable",
                    f"{readiness_field} is not ready",
                    guard_kind="runtime_behavior_unavailable",
                    extra_evidence={readiness_field: False, **configuration_evidence},
                )

        product_geometry = dict(
            grounded_action.get("product_geometry")
            or grounded_action.get("part_geometry")
            or part_context.get("product_geometry")
            or part_context.get("part_geometry")
            or {}
        )
        if not product_geometry:
            return _result(
                "NEEDS_CONTEXT",
                "resource_validation_unavailable",
                f"product geometry for part '{part_name}' is unavailable",
                guard_kind="product_geometry_unavailable",
            )

        try:
            grasp_width_m = float(product_geometry["grasp_width_m"])
        except (KeyError, TypeError, ValueError):
            grasp_width_m = None
        opening_m: float | None = None
        try:
            if gripper_config.get("open_width_mm") is not None:
                opening_m = float(gripper_config["open_width_mm"]) / 1000.0
            elif gripper_config.get("open") is not None:
                opening_m = float(gripper_config["open"])
        except (TypeError, ValueError):
            opening_m = None
        if grasp_width_m is None or not math.isfinite(grasp_width_m):
            return _result(
                "NEEDS_CONTEXT",
                "resource_validation_unavailable",
                f"grasp_width_m for part '{part_name}' is unavailable",
                guard_kind="grasp_geometry_unavailable",
            )
        if opening_m is None or not math.isfinite(opening_m) or grasp_width_m > opening_m:
            return _result(
                "INFEASIBLE",
                "unsupported_resource_target",
                (
                    f"grasp_width_m={grasp_width_m:.4f} exceeds configured gripper "
                    f"opening_m={opening_m if opening_m is not None else 'unavailable'}"
                ),
                guard_kind="gripper_incompatible",
                extra_evidence={
                    "grasp_width_m": grasp_width_m,
                    "gripper_opening_m": opening_m,
                },
            )

        required_tooling = str(product_geometry.get("required_tooling") or "").strip()
        configured_tooling = str(
            static_capabilities.get("tooling")
            or gripper_config.get("tooling")
            or ""
        ).strip()
        if required_tooling and required_tooling != configured_tooling:
            return _result(
                "INFEASIBLE",
                "unsupported_resource_target",
                f"required_tooling '{required_tooling}' is unavailable",
                guard_kind="tooling_incompatible",
                extra_evidence={"configured_tooling": configured_tooling},
            )
        try:
            payload_kg = float(
                product_geometry.get("payload_kg", product_geometry.get("mass_kg"))
            )
        except (TypeError, ValueError):
            payload_kg = None
        try:
            max_payload_kg = float(static_capabilities.get("max_payload_kg"))
        except (TypeError, ValueError):
            max_payload_kg = None
        if payload_kg is not None and (
            max_payload_kg is None or payload_kg > max_payload_kg
        ):
            return _result(
                "INFEASIBLE",
                "unsupported_resource_target",
                "configured payload capacity is incompatible with the part",
                guard_kind="payload_incompatible",
                extra_evidence={
                    "payload_kg": payload_kg,
                    "max_payload_kg": max_payload_kg,
                },
            )

        reachability = {
            str(token).strip()
            for token in static_capabilities.get("reachability") or []
            if str(token or "").strip()
        }
        relevant_location = str(
            source_ref.get("location")
            if function_name == "pick_approach"
            else target_info.get("destination_location")
            or grounded_action.get("destination_location")
            or ""
        ).strip()
        pose_only_location = (
            relevant_location == "observed_pose"
            or relevant_location.endswith("_observed_pose")
        )
        if relevant_location and not pose_only_location:
            if not reachability:
                return _result(
                    "NEEDS_CONTEXT",
                    "resource_validation_unavailable",
                    "static_capabilities.reachability is unavailable",
                    guard_kind="reachability_unavailable",
                )
            if relevant_location not in reachability:
                return _result(
                    "INFEASIBLE",
                    "workspace_unreachable",
                    f"location '{relevant_location}' is outside configured reachability",
                    guard_kind="location_unreachable",
                    extra_evidence={"reachability": sorted(reachability)},
                )

        poses = dict(grounded_action.get("poses") or {})
        source_pose = dict(
            poses.get("source_pose")
            or source_ref.get("pose")
            or part_context.get("observed_pose")
            or recovery_snapshot.get("current_pose")
            or {}
        )
        target_pose = dict(
            poses.get("target_pose")
            or target_info.get("slot_pose")
            or target_info.get("pose")
            or target_info.get("destination_pose")
            or (source_pose if function_name == "pick_approach" else {})
        )
        if not source_pose or not target_pose:
            return _result(
                "NEEDS_CONTEXT",
                "source_reference_unavailable",
                "fresh source and target pose evidence is unavailable",
                guard_kind="pose_evidence_unavailable",
            )
        captured_at = (
            source_ref.get("captured_at")
            if function_name == "pick_approach"
            else target_info.get("captured_at")
            or target_info.get("destination_captured_at")
        )
        try:
            pose_age_sec = time.time() - float(captured_at)
        except (TypeError, ValueError):
            pose_age_sec = float("inf")
        if pose_age_sec < -1.0 or pose_age_sec > 8.0:
            return _result(
                "NEEDS_CONTEXT",
                "source_reference_unavailable",
                "source or destination pose evidence is missing or stale",
                guard_kind="pose_evidence_stale",
                extra_evidence={"pose_age_sec": pose_age_sec},
            )

        try:
            approach_height_m = float(
                product_geometry.get(
                    "approach_height_m",
                    dict(getattr(self, "motion_config", {}) or {}).get(
                        "recovery_observed_pick_approach_height_m",
                        0.06,
                    ),
                )
            )
            computed_approach = {
                "x": float(target_pose["x"]),
                "y": float(target_pose["y"]),
                "z": float(target_pose["z"]) + approach_height_m,
            }
        except (KeyError, TypeError, ValueError):
            return _result(
                "NEEDS_CONTEXT",
                "source_reference_unavailable",
                "computed approach pose cannot be derived from current geometry",
                guard_kind="pose_evidence_unavailable",
            )
        approach_pose = dict(poses.get("approach_pose") or computed_approach)
        retreat_pose = dict(poses.get("retreat_pose") or approach_pose)
        pose_bundle = {
            "source_pose": source_pose,
            "approach_pose": approach_pose,
            "target_pose": target_pose,
            "retreat_pose": retreat_pose,
        }
        gripper_reach = dict(static_capabilities.get("gripper_reach") or {})
        for pose_name, pose in pose_bundle.items():
            inside, workspace_reason = self._is_pose_in_workspace(pose)
            if not inside:
                status = (
                    "NEEDS_CONTEXT"
                    if "capability data is unavailable" in workspace_reason
                    else "INFEASIBLE"
                )
                return _result(
                    status,
                    (
                        "resource_validation_unavailable"
                        if status == "NEEDS_CONTEXT"
                        else "workspace_unreachable"
                    ),
                    f"{pose_name} {workspace_reason}",
                    guard_kind="pose_unreachable",
                    extra_evidence={"checked_poses": pose_bundle},
                )
            reach_error = RobotAgent._recovery_pose_gripper_reach_error(
                pose,
                gripper_reach,
            )
            if reach_error:
                status = (
                    "NEEDS_CONTEXT"
                    if "unavailable" in reach_error or "incomplete" in reach_error
                    else "INFEASIBLE"
                )
                return _result(
                    status,
                    (
                        "resource_validation_unavailable"
                        if status == "NEEDS_CONTEXT"
                        else "workspace_unreachable"
                    ),
                    f"{pose_name} {reach_error}",
                    guard_kind="pose_unreachable",
                    extra_evidence={"checked_poses": pose_bundle},
                )

        if function_name == "place_approach":
            destination_support_valid = recovery_snapshot.get(
                "destination_support_valid",
                target_info.get("destination_support_valid"),
            )
            if destination_support_valid is None:
                return _result(
                    "NEEDS_CONTEXT",
                    "resource_validation_unavailable",
                    "destination support evidence is unavailable",
                    guard_kind="destination_support_unavailable",
                )
            if destination_support_valid is False:
                return _result(
                    "INFEASIBLE",
                    "unsupported_resource_target",
                    "destination support is invalid",
                    guard_kind="destination_support_invalid",
                )
            destination_occupied = recovery_snapshot.get(
                "destination_occupied",
                target_info.get("destination_occupied"),
            )
            if destination_occupied is None:
                return _result(
                    "NEEDS_CONTEXT",
                    "resource_validation_unavailable",
                    "destination occupancy evidence is unavailable",
                    guard_kind="destination_occupancy_unavailable",
                )
            if destination_occupied is True:
                return _result(
                    "INFEASIBLE",
                    "unsupported_resource_target",
                    "destination is occupied",
                    guard_kind="destination_occupied",
                )

        evidence.update(
            {
                "recording_history_used": False,
                "configured_robot": configuration_evidence,
                "checked_poses": deepcopy(pose_bundle),
                "grasp_width_m": grasp_width_m,
                "gripper_opening_m": opening_m,
            }
        )
        return None

    def check_recovery_physical_feasibility(  # noqa: C901
        self,
        *,
        part_context: dict[str, Any],
        recovery_snapshot: dict[str, Any],
        operation_kind: str = "",
        part_name: str | None = None,
        grounded_action: dict[str, Any] | None = None,
        **_compat_kwargs: Any,
    ) -> dict[str, Any]:
        """Check whether this robot can physically execute a grounded recovery action."""
        from copy import deepcopy

        evidence: dict[str, Any] = {
            "part_context": deepcopy(part_context),
            "recovery_snapshot": deepcopy(recovery_snapshot),
            "resource_jid": str(getattr(self, "jid", "") or ""),
        }

        recovery_snapshot = deepcopy(recovery_snapshot or {})
        part_context = deepcopy(part_context or {})
        grounded_action = deepcopy(grounded_action or {})

        evidence["grounded_action"] = deepcopy(grounded_action)
        target_info = dict(grounded_action.get("target") or part_context.get("target") or {})
        expected_effect = dict(grounded_action.get("expected_effect") or {})
        preconditions = dict(grounded_action.get("preconditions") or {})
        part_preconditions = dict(preconditions.get("part") or {})
        source_ref = dict(preconditions.get("source_ref") or {})
        effect_scope = str(grounded_action.get("effect_scope") or "").strip().lower()
        task_kind = (
            str(grounded_action.get("task_kind") or operation_kind or "")
            .strip()
            .lower()
        )
        function_name = str(
            grounded_action.get("function_name")
            or grounded_action.get("robot_task")
            or target_info.get("function_name")
            or ""
        ).strip()
        part_name = str(part_name or grounded_action.get("part_name") or "").strip() or None
        expected_resource = dict(expected_effect.get("resource") or {})
        expected_part = dict(expected_effect.get("part") or {})
        part_affecting = bool(
            effect_scope in {"part_only", "resource_and_part"}
            or any(
                key in expected_part and expected_part.get(key) not in (None, "", [], {})
                for key in ("state", "location", "pose", "holder")
            )
        )
        named_pose = str(
            target_info.get("named_pose") or part_context.get("named_pose") or ""
        ).strip()
        available_named_poses = {
            str(name).strip()
            for name in (
                recovery_snapshot.get("named_poses")
                or self.static_capabilities.get("named_poses")
                or []
            )
            if str(name).strip()
        }
        if (
            named_pose
            and function_name not in {"pick_approach", "place_approach"}
            and not available_named_poses
        ):
            return {
                "allowed": False,
                "constraint_code": "resource_validation_unavailable",
                "guard": {
                    "kind": "named_pose_capability_unavailable",
                    "resource_jid": str(getattr(self, "jid", "") or ""),
                    "named_pose": named_pose,
                },
                "reason": "named-pose capability data is unavailable on this robot",
                "evidence": {
                    **evidence,
                    "named_pose": named_pose,
                    "available_named_poses": [],
                },
            }
        if (
            named_pose
            and function_name not in {"pick_approach", "place_approach"}
            and named_pose not in available_named_poses
        ):
            return {
                "allowed": False,
                "constraint_code": "named_pose_unavailable",
                "guard": {
                    "kind": "named_pose_unavailable",
                    "resource_jid": str(getattr(self, "jid", "") or ""),
                    "named_pose": named_pose,
                },
                "reason": f"named pose '{named_pose}' is not available on this robot",
                "evidence": {
                    **evidence,
                    "named_pose": named_pose,
                    "available_named_poses": sorted(available_named_poses),
                },
            }

        availability = str(recovery_snapshot.get("availability") or "").strip().lower()
        if availability == "unavailable":
            return {
                "allowed": False,
                "feasibility_status": "INFEASIBLE",
                "constraint_code": "resource_unavailable",
                "guard": {
                    "kind": "resource_not_available",
                    "resource_jid": str(getattr(self, "jid", "") or ""),
                },
                "reason": "resource is currently unavailable",
                "evidence": evidence,
            }

        held_part = str(
            recovery_snapshot.get("held_part") or part_context.get("resource_held_part") or ""
        ).strip()
        gripper_state = (
            str(
                recovery_snapshot.get("gripper_state")
                or part_context.get("resource_gripper_state")
                or ""
            )
            .strip()
            .lower()
        )
        current_holder = str(part_context.get("current_holder_resource_jid") or "").strip()
        resource_jid = str(getattr(self, "jid", "") or "")
        desired_resource_state = str(expected_resource.get("current_state") or "").strip()
        desired_resource_location = str(expected_resource.get("location") or "").strip()
        supported_recovery_states = {
            str(token).strip()
            for token in (
                recovery_snapshot.get("supported_recovery_states")
                or self.static_capabilities.get("supported_recovery_states")
                or []
            )
            if str(token).strip()
        }
        allows_abstract_idle_recovery = (
            effect_scope == "resource_only" and desired_resource_state.lower() == "idle"
        )
        requires_part_acquisition = bool(
            part_name
            and part_affecting
            and bool(part_preconditions.get("requires_acquisition"))
        )
        if (
            effect_scope == "resource_only"
            and desired_resource_state
            and not allows_abstract_idle_recovery
            and not (
                named_pose
                or target_info.get("pose")
                or target_info.get("slot_pose")
                or desired_resource_location
            )
            and (
                not supported_recovery_states
                or desired_resource_state not in supported_recovery_states
            )
        ):
            return {
                "allowed": False,
                "constraint_code": "unsupported_resource_target",
                "guard": {
                    "kind": "unsupported_resource_target",
                    "resource_jid": resource_jid,
                    "resource_state": desired_resource_state,
                },
                "reason": (
                    f"resource-only transition targets state '{desired_resource_state}' "
                    "without a concrete supported recovery pose or advertised recovery target"
                ),
                "evidence": {
                    **evidence,
                    "supported_recovery_states": sorted(supported_recovery_states),
                },
            }
        if requires_part_acquisition and part_name:
            if held_part and held_part != str(part_name).strip():
                return {
                    "allowed": False,
                    "feasibility_status": "INFEASIBLE",
                    "constraint_code": "holder_conflict",
                    "guard": {
                        "kind": "resource_holds_part",
                        "resource_jid": str(getattr(self, "jid", "") or ""),
                        "held_part": held_part,
                    },
                    "reason": (
                        f"resource already holds '{held_part}' and cannot acquire "
                        f"'{str(part_name).strip()}'"
                    ),
                    "evidence": {**evidence, "conflicting_part": held_part},
                }
            if current_holder and current_holder != str(getattr(self, "jid", "") or ""):
                return {
                    "allowed": False,
                    "feasibility_status": "INFEASIBLE",
                    "constraint_code": "holder_conflict",
                    "guard": {
                        "kind": "part_held_by_other",
                        "part_name": str(part_name).strip(),
                        "current_holder_resource_jid": current_holder,
                    },
                    "reason": (
                        f"part '{str(part_name).strip()}' is currently held by "
                        f"'{current_holder}', not this robot"
                    ),
                    "evidence": {**evidence, "current_holder_resource_jid": current_holder},
                }
            if not held_part and gripper_state == "closed":
                return {
                    "allowed": False,
                    "feasibility_status": "INFEASIBLE",
                    "constraint_code": "gripper_occupancy_conflict",
                    "guard": {
                        "kind": "gripper_closed_without_target_part",
                        "resource_jid": str(getattr(self, "jid", "") or ""),
                    },
                    "reason": "gripper is already closed without holding the target part",
                    "evidence": evidence,
                }
            if not source_ref:
                return {
                    "allowed": False,
                    "feasibility_status": "NEEDS_CONTEXT",
                    "constraint_code": "source_reference_unavailable",
                    "guard": {
                        "kind": "source_reference_unavailable",
                        "resource_jid": resource_jid,
                        "part_name": str(part_name).strip(),
                    },
                    "reason": (
                        f"task requires acquiring '{str(part_name).strip()}' first but no "
                        "grounded current source reference is available"
                    ),
                    "evidence": evidence,
                }
        elif part_affecting and part_name and task_kind != "continuation_resume":
            if held_part != str(part_name).strip() and current_holder != resource_jid:
                return {
                    "allowed": False,
                    "feasibility_status": "INFEASIBLE",
                    "constraint_code": "required_part_not_held",
                    "guard": {
                        "kind": "required_part_not_held",
                        "resource_jid": resource_jid,
                        "part_name": str(part_name).strip(),
                    },
                    "reason": (
                        f"task changes part '{str(part_name).strip()}' but resource "
                        f"'{resource_jid}' does not currently hold it"
                    ),
                    "evidence": evidence,
                }

        configured_pick_place_result = (
            RobotAgent._configured_pick_place_recovery_feasibility(
                self,
                function_name=function_name,
                part_name=str(part_name or "").strip(),
                part_context=part_context,
                recovery_snapshot=recovery_snapshot,
                grounded_action=grounded_action,
                source_ref=source_ref,
                target_info=target_info,
                evidence=evidence,
            )
        )
        if configured_pick_place_result is not None:
            return configured_pick_place_result

        target_pose: dict[str, Any] | None = None
        source_location = str(
            source_ref.get("location") or target_info.get("source_location") or ""
        ).strip()
        source_location_is_pose_only = (
            source_location == "observed_pose" or source_location.endswith("_observed_pose")
        )
        if requires_part_acquisition or source_location_is_pose_only:
            source_pose = dict(source_ref.get("pose") or {})
            target_pose = (
                source_pose
                or part_context.get("observed_pose")
                or target_info.get("source_pose")
                or part_context.get("pose")
            )
            if requires_part_acquisition and target_pose is None and source_location_is_pose_only:
                return {
                    "allowed": False,
                    "feasibility_status": "NEEDS_CONTEXT",
                    "constraint_code": "source_reference_unavailable",
                    "guard": {
                        "kind": "source_reference_unavailable",
                        "resource_jid": resource_jid,
                        "part_name": str(part_name or "").strip() or None,
                    },
                    "reason": (
                        f"task requires acquiring '{str(part_name).strip()}' first but its "
                        "grounded source reference has no usable pose or location evidence"
                    ),
                    "evidence": {
                        **evidence,
                        "source_ref": deepcopy(source_ref),
                    },
                }
        if target_pose is None:
            target_pose = target_info.get("slot_pose") or target_info.get("pose") or None

        if target_pose is None:
            return {
                "allowed": True,
                "reason": (
                    "grounded preconditions are satisfied and no pose-dependent "
                    "reachability check is required"
                ),
                "evidence": evidence,
            }

        inside, reason = self._is_pose_in_workspace(target_pose)
        evidence["checked_pose"] = deepcopy(target_pose)
        evidence["workspace_bounds"] = deepcopy(
            self.static_capabilities.get("workspace_bounds") or {}
        )
        workspace_data_unavailable = "capability data is unavailable" in reason
        return {
            "allowed": inside,
            "constraint_code": (
                "resource_validation_unavailable"
                if workspace_data_unavailable
                else "workspace_unreachable" if not inside else None
            ),
            "guard": (
                {
                    "kind": "observed_pose_unreachable",
                    "resource_jid": str(getattr(self, "jid", "") or ""),
                    "part_name": str(part_name or "").strip() or None,
                    "pose": deepcopy(target_pose),
                }
                if not inside
                else None
            ),
            "reason": reason,
            "evidence": evidence,
        }

    def _snapshot_state(self) -> dict[str, Any]:
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

        interval = 2.0  # progress tick interval
        elapsed = 0.0

        while elapsed < duration:
            await asyncio.sleep(interval)
            elapsed += interval
            self.logger.debug(
                "[%s] ... %s (%.1f / %.1f sec)", robot, description, elapsed, duration
            )

        self.logger.info("[%s] Finished: %s", robot, description)


for _robot_task_name in robot_task_names():
    setattr(RobotAgent, _robot_task_name, robot_task_registry()[_robot_task_name].handler)


del _robot_task_name
