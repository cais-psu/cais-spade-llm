"""Hardware/digital_twin pick-place controller helpers for taught functions."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gazebo_pick_place_controller import (
    UR5E_JOINT_NAMES,
    UR5E_JOINT_STATES_TOPIC,
    UR5E_TRAJECTORY_TOPIC,
    XARM6_JOINT_NAMES,
    XARM6_JOINT_STATES_TOPIC,
    GazeboPickPlaceController,
)

TAUGHT_FUNCTIONS_ROOT = Path(__file__).resolve().parent / "taught_functions"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    lower = min(float(lo), float(hi))
    upper = max(float(lo), float(hi))
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class UR5eRG2GripperControllerSettings:
    hostname: str = "192.168.1.172"
    open_position: float = 0.11
    close_position: float = 0.02
    open_width_mm: float = 70.0
    close_width_mm: float = 10.0
    open_force: float = 10.0
    close_force: float = 40.0
    open_settle_sec: float = 1.2
    close_settle_sec: float = 2.0
    rtde_method: str = "function"
    disable_remote_control_check: bool = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        hostname: str | None = None,
    ) -> UR5eRG2GripperControllerSettings:
        gripper = dict(config or {})
        rtde = dict(gripper.get("rtde") or {})
        resolved_hostname = (
            str(hostname or "").strip() or str(rtde.get("hostname") or "").strip() or cls.hostname
        )
        return cls(
            hostname=resolved_hostname,
            open_position=_as_float(gripper.get("open"), cls.open_position),
            close_position=_as_float(gripper.get("close"), cls.close_position),
            open_width_mm=_as_float(rtde.get("open_width_mm"), cls.open_width_mm),
            close_width_mm=_as_float(rtde.get("close_width_mm"), cls.close_width_mm),
            open_force=_as_float(rtde.get("open_force"), cls.open_force),
            close_force=_as_float(rtde.get("close_force"), cls.close_force),
            open_settle_sec=_as_float(rtde.get("open_settle_sec"), cls.open_settle_sec),
            close_settle_sec=_as_float(rtde.get("close_settle_sec"), cls.close_settle_sec),
            rtde_method=str(rtde.get("method") or gripper.get("rtde_method") or cls.rtde_method),
            disable_remote_control_check=_as_bool(
                rtde.get("disable_remote_control_check"),
                cls.disable_remote_control_check,
            ),
        )


class UR5eRG2GripperController:
    """UR5e OnRobot RG2 control over UR RTDE custom script execution."""

    def __init__(
        self,
        *,
        hostname: str = "192.168.1.172",
        open_position: float = 0.11,
        close_position: float = 0.02,
        open_width_mm: float = 70.0,
        close_width_mm: float = 10.0,
        open_force: float = 10.0,
        close_force: float = 40.0,
        open_settle_sec: float = 1.2,
        close_settle_sec: float = 2.0,
        rtde_method: str = "function",
        disable_remote_control_check: bool = False,
        rtde_factory: Callable[..., Any] | None = None,
    ) -> None:
        resolved_rtde_method = str(rtde_method or "function").strip().lower()
        if resolved_rtde_method not in {"function", "script"}:
            resolved_rtde_method = "function"
        self.settings = UR5eRG2GripperControllerSettings(
            hostname=str(hostname or "").strip() or "192.168.1.172",
            open_position=float(open_position),
            close_position=float(close_position),
            open_width_mm=float(open_width_mm),
            close_width_mm=float(close_width_mm),
            open_force=float(open_force),
            close_force=float(close_force),
            open_settle_sec=float(open_settle_sec),
            close_settle_sec=float(close_settle_sec),
            rtde_method=resolved_rtde_method,
            disable_remote_control_check=bool(disable_remote_control_check),
        )
        factory = rtde_factory or self._default_rtde_factory()
        self.rtde = factory(
            hostname=self.settings.hostname,
            **self._rtde_factory_kwargs(self.settings),
        )

    @classmethod
    def from_settings(
        cls,
        settings: UR5eRG2GripperControllerSettings,
        *,
        rtde_factory: Callable[..., Any] | None = None,
    ) -> UR5eRG2GripperController:
        return cls(
            hostname=settings.hostname,
            open_position=settings.open_position,
            close_position=settings.close_position,
            open_width_mm=settings.open_width_mm,
            close_width_mm=settings.close_width_mm,
            open_force=settings.open_force,
            close_force=settings.close_force,
            open_settle_sec=settings.open_settle_sec,
            close_settle_sec=settings.close_settle_sec,
            rtde_method=settings.rtde_method,
            disable_remote_control_check=settings.disable_remote_control_check,
            rtde_factory=rtde_factory,
        )

    @staticmethod
    def _default_rtde_factory() -> Callable[..., Any]:
        from rtde_control import RTDEControlInterface

        return RTDEControlInterface

    @staticmethod
    def _rtde_factory_kwargs(settings: UR5eRG2GripperControllerSettings) -> dict[str, Any]:
        if not settings.disable_remote_control_check:
            return {}
        try:
            from rtde_control import RTDEControlInterface
        except ModuleNotFoundError:
            return {"flags": 0}

        return {
            "flags": (
                RTDEControlInterface.FLAGS_DEFAULT
                | RTDEControlInterface.FLAG_DISABLE_REMOTE_CONTROL_CHECK
            )
        }

    @staticmethod
    def script_body(width_mm: float, force: float) -> str:
        return f"""
      local rg = rpc_factory("xmlrpc","http://localhost:41414")
      local ret = rg.rg_grip(0, {float(width_mm)}, {float(force)})
      textmsg("rg_grip returned: ", ret)
    """

    @staticmethod
    def program_body(width_mm: float, force: float, *, name: str = "rg2_cmd") -> str:
        body = "\n".join(
            f"  {line.strip()}"
            for line in UR5eRG2GripperController.script_body(width_mm, force).strip().splitlines()
        )
        return f"def {name}():\n{body}\nend\n"

    @staticmethod
    def inline_program_body(width_mm: float, force: float) -> str:
        body = "\n".join(
            f"  {line.strip()}"
            for line in UR5eRG2GripperController.script_body(width_mm, force).strip().splitlines()
        )
        return f"def program():\n{body}\nend\nrun program\n"

    @staticmethod
    def secondary_program_body(width_mm: float, force: float, *, name: str = "rg2_cmd") -> str:
        body = "\n".join(
            f"  {line.strip()}"
            for line in UR5eRG2GripperController.script_body(width_mm, force).strip().splitlines()
        )
        return f"sec {name}():\n{body}\nend\n"

    def width_mm_from_position(self, position: float) -> float:
        settings = self.settings
        open_position = float(settings.open_position)
        close_position = float(settings.close_position)
        if abs(open_position - close_position) < 1e-9:
            return float(settings.open_width_mm)
        ratio = (float(position) - close_position) / (open_position - close_position)
        width = settings.close_width_mm + ratio * (settings.open_width_mm - settings.close_width_mm)
        return _clamp(width, settings.close_width_mm, settings.open_width_mm)

    def force_for_position(self, position: float) -> float:
        settings = self.settings
        midpoint = (float(settings.open_position) + float(settings.close_position)) * 0.5
        if settings.open_position >= settings.close_position:
            return settings.open_force if float(position) >= midpoint else settings.close_force
        return settings.open_force if float(position) <= midpoint else settings.close_force

    def settle_sec_for_position(self, position: float) -> float:
        settings = self.settings
        midpoint = (float(settings.open_position) + float(settings.close_position)) * 0.5
        if settings.open_position >= settings.close_position:
            return (
                settings.open_settle_sec
                if float(position) >= midpoint
                else settings.close_settle_sec
            )
        return (
            settings.open_settle_sec if float(position) <= midpoint else settings.close_settle_sec
        )

    def command_width(
        self,
        width_mm: float,
        force: float,
        *,
        settle_sec: float = 0.0,
        blocking: bool = True,
    ) -> None:
        if self.settings.rtde_method == "script":
            result = self.rtde.sendCustomScript(self.inline_program_body(width_mm, force))
        else:
            body = self.script_body(width_mm, force)
            result = self.rtde.sendCustomScriptFunction("rg2_cmd", body)
        if result is False:
            raise RuntimeError(f"RTDE RG2 {self.settings.rtde_method} command timed out")
        if blocking and settle_sec > 0.0:
            time.sleep(float(settle_sec))

    def command_position(self, position: float, *, blocking: bool = True) -> float:
        width_mm = self.width_mm_from_position(position)
        self.command_width(
            width_mm,
            self.force_for_position(position),
            settle_sec=self.settle_sec_for_position(position),
            blocking=blocking,
        )
        return width_mm

    def open_gripper(self, *, blocking: bool = True) -> None:
        settings = self.settings
        self.command_width(
            settings.open_width_mm,
            settings.open_force,
            settle_sec=settings.open_settle_sec,
            blocking=blocking,
        )

    def close_gripper(self, *, blocking: bool = True) -> None:
        settings = self.settings
        self.command_width(
            settings.close_width_mm,
            settings.close_force,
            settle_sec=settings.close_settle_sec,
            blocking=blocking,
        )

    def disconnect(self) -> None:
        disconnect = getattr(self.rtde, "disconnect", None)
        if callable(disconnect):
            disconnect()


class HardwarePickPlaceController(GazeboPickPlaceController):
    """Hardware/digital_twin controller with taught function replay."""

    taught_functions_root = TAUGHT_FUNCTIONS_ROOT

    def attach_part(
        self,
        model_name: str,
        link: str | None = None,
        part_name: str = "",
    ) -> dict[str, Any]:
        """Record physical mechanical custody without calling a Gazebo service."""
        _ = link
        target = str(model_name or part_name or "").strip()
        if not target:
            return {"success": False, "message": "physical grasp target is empty"}
        self._attached_model = str(model_name or target)
        return {"success": True, "message": f"physical grasp custody recorded for {target}"}

    def detach_part(
        self,
        model_name: str = "",
        link: str | None = None,
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """Clear physical custody after the hardware gripper opens."""
        _ = (link, assume_released_if_open)
        target = str(model_name or self._attached_model or "held part").strip()
        self._attached_model = None
        self._attached_link = None
        return {"success": True, "message": f"physical release custody recorded for {target}"}

    @staticmethod
    def _safe_name(name: object) -> str:
        value = str(name or "").strip()
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in value)
        return safe or "default"

    @classmethod
    def function_file_path(
        cls,
        robot: str,
        function_name: str,
        name: str,
        *,
        storage_source: str = "hardware",
    ) -> Path:
        return (
            cls.taught_functions_root
            / str(robot or "").strip().lower()
            / str(function_name or "").strip()
            / f"{cls._safe_name(name)}__{str(storage_source or 'hardware').strip()}.json"
        )

    @classmethod
    def load_function(
        cls,
        robot: str,
        function_name: str,
        name: str,
        *,
        storage_source: str = "hardware",
    ) -> dict[str, Any]:
        path = cls.function_file_path(
            robot,
            function_name,
            name,
            storage_source=storage_source,
        )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"taught function file is not a JSON object: {path}")
        return payload

    @staticmethod
    def _step_positions(step: dict[str, Any]) -> list[float]:
        waypoint = dict(step.get("waypoint") or {})
        positions = waypoint.get("joint_positions") or waypoint.get("positions") or []
        return [float(value) for value in positions]

    def replay_step(self, step: dict[str, Any], *, duration_sec: float = 2.0) -> dict[str, Any]:
        primitive = str(step.get("primitive") or "").strip()
        if primitive == "delay":
            params = dict(step.get("params") or {})
            return self.delay(duration_sec=params.get("duration_sec", 0.0))
        if primitive in {"move_cartesian", "move_relative", "move_to_named_pose"}:
            positions = self._step_positions(step)
            if not positions:
                return {"success": False, "message": f"{primitive} step has no waypoint."}
            ok = self.move_joints(positions, duration_sec=duration_sec)
            return {
                "success": bool(ok),
                "message": f"{primitive} waypoint replay {'succeeded' if ok else 'failed'}.",
            }
        if primitive == "grasp_part":
            params = dict(step.get("params") or {})
            ok = self.close_gripper(position=params.get("position"))
            return {"success": bool(ok), "message": "grasp_part"}
        if primitive == "release_part":
            ok = self.open_gripper()
            return {"success": bool(ok), "message": "release_part"}
        if primitive == "open_gripper":
            ok = self.open_gripper()
            return {"success": bool(ok), "message": "open_gripper"}
        if primitive == "close_gripper":
            ok = self.close_gripper()
            return {"success": bool(ok), "message": "close_gripper"}
        return {"success": False, "message": f"unsupported primitive: {primitive}"}

    def replay_function_payload(
        self,
        payload: dict[str, Any],
        *,
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        steps = list(payload.get("steps") or [])
        if not steps:
            return {"success": False, "message": "function has no steps."}
        results: list[dict[str, Any]] = []
        for step in steps:
            result = self.replay_step(dict(step), duration_sec=duration_sec)
            results.append(result)
            if not result.get("success"):
                return {
                    "success": False,
                    "message": str(result.get("message") or "step replay failed."),
                    "results": results,
                }
        return {"success": True, "message": "function replay succeeded.", "results": results}

    def replay_function(
        self,
        function_name: str,
        name: str,
        *,
        storage_source: str = "hardware",
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        payload = self.load_function(
            self.robot_name,
            function_name,
            name,
            storage_source=storage_source,
        )
        return self.replay_function_payload(payload, duration_sec=duration_sec)

    def taught_function_step(
        self,
        function_name: str,
        name: str,
        step_name: str,
        *,
        storage_source: str = "hardware",
    ) -> tuple[dict[str, Any] | None, Path, str]:
        path = self.function_file_path(
            self.robot_name,
            function_name,
            name,
            storage_source=storage_source,
        )
        try:
            payload = self.load_function(
                self.robot_name,
                function_name,
                name,
                storage_source=storage_source,
            )
        except FileNotFoundError:
            return None, path, f"taught function file not found: {path}"
        except Exception as exc:
            return None, path, f"could not load taught function file {path}: {exc}"
        target_step = str(step_name or "").strip()
        for step in list(payload.get("steps") or []):
            item = dict(step or {})
            if str(item.get("step_name") or "").strip() == target_step:
                return item, path, ""
        return (
            None,
            path,
            f"taught function step not found: {function_name}.{target_step} in {path}",
        )

    def replay_taught_function_step(
        self,
        function_name: str,
        name: str,
        step_name: str,
        *,
        storage_source: str = "hardware",
        duration_sec: float = 2.0,
    ) -> dict[str, Any]:
        step, path, err = self.taught_function_step(
            function_name,
            name,
            step_name,
            storage_source=storage_source,
        )
        if err or step is None:
            return {"success": False, "message": err, "file": str(path)}
        result = self.replay_step(step, duration_sec=duration_sec)
        result.setdefault("file", str(path))
        result.setdefault("step_name", str(step_name or ""))
        return result


class UR5eHardwareController(HardwarePickPlaceController):
    """Config-driven UR5e hardware/digital_twin controller."""

    def __init__(
        self,
        trajectory_topic: str = UR5E_TRAJECTORY_TOPIC,
        joint_states_topic: str = UR5E_JOINT_STATES_TOPIC,
        *,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "physical",
    ) -> None:
        super().__init__(
            robot_name="ur5e",
            node_name=f"ur5e_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=UR5E_JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )
        self._rg2_gripper: UR5eRG2GripperController | None = None
        self._rg2_gripper_attempted = False

    def _get_rg2_gripper(self) -> UR5eRG2GripperController | None:
        if self._rg2_gripper is not None:
            return self._rg2_gripper
        if self._rg2_gripper_attempted:
            return None
        self._rg2_gripper_attempted = True
        try:
            settings = UR5eRG2GripperControllerSettings.from_config(
                dict(self.controller_config.get("gripper") or {})
            )
            self._rg2_gripper = UR5eRG2GripperController.from_settings(settings)
        except Exception as exc:
            self._last_failure_message = f"RG2 gripper unavailable: {exc}"
            self._rg2_gripper = None
        return self._rg2_gripper

    def open_gripper(self) -> bool:
        gripper = self._get_rg2_gripper()
        if gripper is not None:
            try:
                gripper.open_gripper()
                return True
            except Exception as exc:
                self._last_failure_message = f"RG2 open_gripper failed: {exc}"
                return False
        return super().open_gripper()

    def close_gripper(self, position: float | None = None) -> bool:
        gripper = self._get_rg2_gripper()
        if gripper is not None:
            try:
                if position is None:
                    gripper.close_gripper()
                else:
                    gripper.command_position(float(position))
                return True
            except Exception as exc:
                self._last_failure_message = f"RG2 close_gripper failed: {exc}"
                return False
        return super().close_gripper(position=position)

    def shutdown(self) -> None:
        gripper = self._rg2_gripper
        try:
            if gripper is not None:
                gripper.disconnect()
        finally:
            super().shutdown()


class XArm6HardwareController(HardwarePickPlaceController):
    """Config-driven xArm6 hardware/digital_twin controller."""

    def __init__(
        self,
        *,
        trajectory_topic: str | None = None,
        joint_states_topic: str = XARM6_JOINT_STATES_TOPIC,
        controller_config: dict[str, Any] | None = None,
        named_positions: dict[str, Any] | None = None,
        execution_mode: str = "physical",
    ) -> None:
        super().__init__(
            robot_name="xarm6",
            node_name=f"xarm6_controller_{os.getpid()}",
            controller_config=controller_config or {},
            named_positions=named_positions,
            execution_mode=execution_mode,
            arm_joint_names=XARM6_JOINT_NAMES,
            arm_trajectory_topic=trajectory_topic,
            joint_states_topic=joint_states_topic,
        )
