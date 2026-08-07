"""Hardware/digital_twin pick-place controller helpers for taught functions."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from contextlib import suppress
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

    def position_from_width_mm(self, width_mm: float) -> float:
        """Map one physical RG2 width to its configured trajectory position."""
        minimum_width = min(float(self.close_width_mm), float(self.open_width_mm))
        maximum_width = max(float(self.close_width_mm), float(self.open_width_mm))
        requested_width = float(width_mm)
        if not math.isfinite(requested_width):
            raise ValueError("RG2 grasp width must be finite")
        if not minimum_width <= requested_width <= maximum_width:
            raise ValueError(
                f"RG2 grasp width {requested_width:.3f} mm is outside "
                f"[{minimum_width:.3f}, {maximum_width:.3f}] mm"
            )
        width_span = float(self.open_width_mm) - float(self.close_width_mm)
        if abs(width_span) <= 1e-9:
            raise ValueError("RG2 configured width range is zero")
        ratio = (requested_width - float(self.close_width_mm)) / width_span
        return float(self.close_position) + ratio * (
            float(self.open_position) - float(self.close_position)
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
        gripper_config = dict(self.controller_config.get("gripper") or {})
        self._ur5e_hardware_trajectory_action = str(
            self.controller_config.get("hardware_trajectory_action") or ""
        ).strip()
        self._ur5e_hardware_trajectory_client: Any | None = None
        self._rg2_action_name = str(gripper_config.get("action") or "").strip()
        self._rg2_action_client: Any | None = None
        self._FollowJointTrajectory: Any | None = None

    def _ur5e_rg2_settings(self) -> UR5eRG2GripperControllerSettings:
        settings = getattr(self, "_cached_ur5e_rg2_settings", None)
        if isinstance(settings, UR5eRG2GripperControllerSettings):
            return settings
        gripper_config = dict(
            getattr(self, "controller_config", {}).get("gripper") or {}
        )
        settings = UR5eRG2GripperControllerSettings.from_config(gripper_config)
        self._cached_ur5e_rg2_settings = settings
        return settings

    def _derive_gripper_close_position(
        self,
        *,
        model_name: str = "",
        product_geometry: dict[str, Any] | None = None,
    ) -> float | None:
        """Map explicit physical part width to RG2 position without Gazebo geometry."""
        _ = model_name
        geometry = product_geometry if isinstance(product_geometry, dict) else {}
        grasp_width_m = None
        for key in (
            "grasp_width_m",
            "part_width_m",
            "part_diameter_m",
            "diameter_m",
            "width_m",
        ):
            if key not in geometry:
                continue
            candidate = _as_float(geometry.get(key), 0.0)
            if candidate > 0.0 and math.isfinite(candidate):
                grasp_width_m = candidate
                break
        if grasp_width_m is None:
            self._last_failure_message = (
                "physical grasp geometry has no explicit width; Gazebo model geometry is not "
                "accepted"
            )
            return None
        try:
            return self._ur5e_rg2_settings().position_from_width_mm(
                grasp_width_m * 1000.0
            )
        except ValueError as exc:
            self._last_failure_message = str(exc)
            return None

    def _physical_stl_pick_readiness(
        self,
        product_geometry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate the actual-STL MG hub against the configured stock RG2 fingertips."""
        geometry = dict(product_geometry or {})
        source_stl = str(geometry.get("source_stl") or "").strip()
        if not source_stl:
            return {
                "success": False,
                "message": "physical MG grasp requires the actual source_stl geometry",
            }
        if not Path(source_stl).is_file():
            return {
                "success": False,
                "message": f"physical MG actual source_stl is unavailable: {source_stl}",
            }
        if geometry.get("hub_up") is not True:
            return {
                "success": False,
                "message": "physical MG actual STL grasp requires hub_up=true",
            }
        source_stl_sha256 = str(geometry.get("source_stl_sha256") or "").strip()
        try:
            hash_is_valid = (
                len(source_stl_sha256) == 64 and int(source_stl_sha256, 16) >= 0
            )
        except ValueError:
            hash_is_valid = False
        if not hash_is_valid:
            return {
                "success": False,
                "message": "physical MG actual source_stl_sha256 is missing or invalid",
            }

        numeric_fields: dict[str, float] = {}
        for field_name in (
            "part_height_m",
            "hub_diameter_m",
            "hub_height_m",
            "tooth_diameter_m",
            "tooth_height_m",
            "grasp_width_m",
            "tooth_clearance_m",
            "minimum_hub_overlap_m",
        ):
            try:
                value = float(geometry[field_name])
            except (KeyError, TypeError, ValueError, OverflowError):
                return {
                    "success": False,
                    "message": f"physical MG actual STL geometry is missing {field_name}",
                }
            if not math.isfinite(value) or value <= 0.0:
                return {
                    "success": False,
                    "message": (
                        f"physical MG actual STL geometry has invalid {field_name}={value!r}"
                    ),
                }
            numeric_fields[field_name] = value

        gripper_position = self._derive_gripper_close_position(
            model_name=str(geometry.get("model_name") or ""),
            product_geometry=geometry,
        )
        if gripper_position is None:
            return {
                "success": False,
                "message": self._last_failure_message or "physical MG RG2 width is invalid",
            }

        gripper_config = dict(
            getattr(self, "controller_config", {}).get("gripper") or {}
        )
        stock_fingertip = dict(gripper_config.get("stock_fingertip") or {})
        try:
            open_gripper_position = float(stock_fingertip["open_gripper_position"])
            mg_gripper_close_position = float(
                stock_fingertip["mg_gripper_close_position"]
            )
            open_inner_pad_lower_z = float(
                stock_fingertip["open_inner_pad_lower_z_from_tcp_m"]
            )
            open_inner_pad_upper_z = float(
                stock_fingertip["open_inner_pad_upper_z_from_tcp_m"]
            )
            closed_inner_pad_lower_z = float(
                stock_fingertip["inner_pad_lower_z_from_tcp_m"]
            )
            closed_inner_pad_upper_z = float(
                stock_fingertip["inner_pad_upper_z_from_tcp_m"]
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return {
                "success": False,
                "message": (
                    "physical MG grasp requires the configured stock RG2 open and closed "
                    "fingertip bands"
                ),
            }
        fingertip_values = (
            open_gripper_position,
            mg_gripper_close_position,
            open_inner_pad_lower_z,
            open_inner_pad_upper_z,
            closed_inner_pad_lower_z,
            closed_inner_pad_upper_z,
        )
        if not all(math.isfinite(value) for value in fingertip_values):
            return {
                "success": False,
                "message": "stock RG2 open or closed fingertip band contains non-finite values",
            }
        if (
            open_inner_pad_lower_z >= open_inner_pad_upper_z
            or closed_inner_pad_lower_z >= closed_inner_pad_upper_z
        ):
            return {
                "success": False,
                "message": "stock RG2 open or closed fingertip band is inverted",
            }
        position_tolerance_m = 5e-6
        if (
            abs(open_gripper_position - 0.11) > position_tolerance_m
            or abs(float(self.gripper_open) - open_gripper_position)
            > position_tolerance_m
        ):
            return {
                "success": False,
                "message": (
                    "physical MG stock RG2 open fingertip band requires gripper "
                    "position 0.11"
                ),
            }
        if (
            abs(mg_gripper_close_position - 0.047) > position_tolerance_m
            or abs(gripper_position - mg_gripper_close_position) > position_tolerance_m
        ):
            return {
                "success": False,
                "message": (
                    "physical MG stock RG2 closed fingertip band requires calculated "
                    "gripper position approximately 0.047"
                ),
            }

        lower_closing_z_displacement_m = (
            closed_inner_pad_lower_z - open_inner_pad_lower_z
        )
        upper_closing_z_displacement_m = (
            closed_inner_pad_upper_z - open_inner_pad_upper_z
        )
        predicted_closing_z_displacement_m = (
            lower_closing_z_displacement_m + upper_closing_z_displacement_m
        ) / 2.0
        closing_displacement_tolerance_m = 5e-5
        if (
            abs(lower_closing_z_displacement_m - upper_closing_z_displacement_m)
            > closing_displacement_tolerance_m
            or abs(predicted_closing_z_displacement_m - (-0.02616))
            > closing_displacement_tolerance_m
        ):
            return {
                "success": False,
                "message": (
                    "physical MG stock RG2 closing displacement must be approximately "
                    "-0.02616 m"
                ),
            }

        tooth_height_m = numeric_fields["tooth_height_m"]
        part_height_m = numeric_fields["part_height_m"]
        tooth_clearance_m = numeric_fields["tooth_clearance_m"]
        minimum_hub_overlap_m = numeric_fields["minimum_hub_overlap_m"]
        pick_tcp_z_offset_from_table_m = (
            tooth_height_m + tooth_clearance_m - closed_inner_pad_lower_z
        )
        mg_pick_z_adjustment_m = float(self.pick_z_adjustments_m.get("MG", 0.0))
        if (
            not math.isfinite(mg_pick_z_adjustment_m)
            or mg_pick_z_adjustment_m < 0.0
            or mg_pick_z_adjustment_m > 0.002
        ):
            return {
                "success": False,
                "message": (
                    "physical MG pick_z_adjustments_m.MG must remain between 0.000 "
                    "and 0.002 m"
                ),
            }
        adjusted_tcp_z_offset_from_table_m = (
            pick_tcp_z_offset_from_table_m + mg_pick_z_adjustment_m
        )
        lowest_closing_endpoint_z_from_tcp_m = min(
            open_inner_pad_lower_z,
            closed_inner_pad_lower_z,
        )
        pad_lower_m = (
            adjusted_tcp_z_offset_from_table_m
            + lowest_closing_endpoint_z_from_tcp_m
        )
        closed_pad_lower_m = (
            adjusted_tcp_z_offset_from_table_m + closed_inner_pad_lower_z
        )
        closed_pad_upper_m = (
            adjusted_tcp_z_offset_from_table_m + closed_inner_pad_upper_z
        )
        hub_overlap_m = max(
            0.0,
            min(closed_pad_upper_m, part_height_m)
            - max(closed_pad_lower_m, tooth_height_m),
        )
        measured_tooth_clearance_m = pad_lower_m - tooth_height_m
        if measured_tooth_clearance_m + 1e-9 < tooth_clearance_m:
            return {
                "success": False,
                "message": (
                    "stock RG2 fingertip tooth clearance is insufficient: "
                    f"{measured_tooth_clearance_m * 1000.0:.2f} mm"
                ),
            }
        if hub_overlap_m + 1e-9 < minimum_hub_overlap_m:
            return {
                "success": False,
                "message": (
                    "stock RG2 fingertip hub overlap is insufficient: "
                    f"{hub_overlap_m * 1000.0:.2f} mm"
                ),
            }
        return {
            "success": True,
            "source_stl": source_stl,
            "source_stl_sha256": source_stl_sha256,
            "hub_up": True,
            **numeric_fields,
            "gripper_close_position": gripper_position,
            "pick_tcp_z_offset_from_table_m": pick_tcp_z_offset_from_table_m,
            "pick_z_adjustment_m": mg_pick_z_adjustment_m,
            "finger_tooth_clearance_m": measured_tooth_clearance_m,
            "finger_hub_overlap_m": hub_overlap_m,
            "open_gripper_position": open_gripper_position,
            "mg_gripper_close_position": mg_gripper_close_position,
            "open_inner_pad_lower_z_from_tcp_m": open_inner_pad_lower_z,
            "open_inner_pad_upper_z_from_tcp_m": open_inner_pad_upper_z,
            "closed_inner_pad_lower_z_from_tcp_m": closed_inner_pad_lower_z,
            "closed_inner_pad_upper_z_from_tcp_m": closed_inner_pad_upper_z,
            "predicted_closing_z_displacement_m": predicted_closing_z_displacement_m,
            "lowest_closing_endpoint_z_from_tcp_m": (
                lowest_closing_endpoint_z_from_tcp_m
            ),
            "inner_pad_lower_z_from_tcp_m": closed_inner_pad_lower_z,
            "inner_pad_upper_z_from_tcp_m": closed_inner_pad_upper_z,
        }

    def init(self) -> bool:
        """Initialize the serialized physical arm and RG2 action clients."""
        if not super().init():
            return False
        if (
            self._ur5e_hardware_trajectory_client is not None
            and self._rg2_action_client is not None
        ):
            return True
        if not self._ur5e_hardware_trajectory_action:
            self._last_failure_message = "UR5e hardware trajectory action is not configured"
            return False
        if not self._rg2_action_name:
            self._last_failure_message = "RG2 gripper action is not configured"
            return False
        try:
            from control_msgs.action import FollowJointTrajectory

            self._FollowJointTrajectory = FollowJointTrajectory
            self._ur5e_hardware_trajectory_client = self._ActionClient(
                self._node,
                FollowJointTrajectory,
                self._ur5e_hardware_trajectory_action,
                callback_group=self._cb_group,
            )
            self._rg2_action_client = self._ActionClient(
                self._node,
                FollowJointTrajectory,
                self._rg2_action_name,
                callback_group=self._cb_group,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"physical UR5e action client unavailable: {exc}"
            arm_client = self._ur5e_hardware_trajectory_client
            self._ur5e_hardware_trajectory_client = None
            self._rg2_action_client = None
            if arm_client is not None:
                destroy = getattr(arm_client, "destroy", None)
                if callable(destroy):
                    with suppress(AttributeError, RuntimeError):
                        destroy()
            return False
        return True

    def _cancel_ur5e_hardware_trajectory_goal(
        self,
        goal_handle: Any,
        *,
        label: str,
    ) -> str:
        """Request cancellation and wait briefly for the RTDE goal to become terminal."""
        cancel_accepted = False
        try:
            cancel_future = goal_handle.cancel_goal_async()
            cancel_response = self._wait_future(
                cancel_future,
                timeout_sec=3.0,
                label=f"cancel:{label}",
                timeout_log_level="warning",
            )
            cancel_accepted = bool(
                cancel_response is not None
                and list(getattr(cancel_response, "goals_canceling", []) or [])
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return f"cancel request failed ({exc})"

        try:
            terminal_future = goal_handle.get_result_async()
            terminal = self._wait_future(
                terminal_future,
                timeout_sec=5.0,
                label=f"cancel-result:{label}",
                timeout_log_level="warning",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            terminal = None
            terminal_error = str(exc)
        else:
            terminal_error = ""
        if terminal is not None:
            try:
                terminal_status = int(getattr(terminal, "status", -1))
            except (TypeError, ValueError):
                terminal_status = -1
            if terminal_status in {4, 5, 6}:
                return f"goal reached terminal status {terminal_status} after cancel request"
        if terminal_error:
            return f"cancel {'accepted' if cancel_accepted else 'unconfirmed'}; result failed ({terminal_error})"
        return (
            "cancel accepted but terminal result was not confirmed"
            if cancel_accepted
            else "cancel request was not confirmed"
        )

    def _command_ur5e_hardware_trajectory_action(  # noqa: C901 - explicit action gates.
        self,
        positions: list[float],
        *,
        duration_sec: float,
        label: str,
    ) -> bool:
        """Send one guarded two-point arm trajectory to the physical RTDE action."""
        if not self.wait_for_services():
            return False
        client = self._ur5e_hardware_trajectory_client
        action_type = self._FollowJointTrajectory
        if client is None or action_type is None:
            self._last_failure_message = "UR5e hardware trajectory action client is unavailable"
            return False
        if len(positions) != len(self.arm_joint_names):
            self._last_failure_message = (
                f"{label} expected {len(self.arm_joint_names)} joints, got {len(positions)}"
            )
            return False
        try:
            target_positions = [float(position) for position in positions]
        except (TypeError, ValueError) as exc:
            self._last_failure_message = f"{label} contains invalid joint values: {exc}"
            return False
        if not all(math.isfinite(position) for position in target_positions):
            self._last_failure_message = f"{label} contains non-finite joint values"
            return False

        current_positions, missing = self._get_arm_joint_positions(timeout_sec=1.0)
        if current_positions is None:
            missing_text = ", ".join(missing) if missing else "unknown"
            self._last_failure_message = (
                f"UR5e current joint state is unavailable; missing={missing_text}"
            )
            return False
        try:
            current_joint_positions = [float(position) for position in current_positions]
        except (TypeError, ValueError):
            current_joint_positions = []
        if len(current_joint_positions) != len(self.arm_joint_names) or not all(
            math.isfinite(position) for position in current_joint_positions
        ):
            self._last_failure_message = "UR5e current joint state contains invalid values"
            return False
        try:
            action_ready = bool(client.wait_for_server(timeout_sec=2.0))
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: wait failed ({exc})"
            )
            return False
        if not action_ready:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action} is unavailable"
            )
            return False

        goal = action_type.Goal()
        goal.trajectory.joint_names = list(self.arm_joint_names)
        start_point = self._JointTrajectoryPoint()
        start_point.positions = current_joint_positions
        start_point.time_from_start = self._Duration(sec=0, nanosec=0)
        target_point = self._JointTrajectoryPoint()
        target_point.positions = target_positions
        duration = max(0.1, float(duration_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        target_point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [start_point, target_point]

        try:
            send_future = client.send_goal_async(goal)
            goal_handle = self._wait_future(
                send_future,
                timeout_sec=3.0,
                label=f"send:{label}",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: send failed ({exc})"
            )
            return False
        if goal_handle is None:
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: send timeout"
            )
            return False
        if not bool(getattr(goal_handle, "accepted", False)):
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: goal rejected"
            )
            return False

        try:
            result_future = goal_handle.get_result_async()
            # The RTDE server may lengthen the nominal duration to enforce its
            # velocity, acceleration, and jerk limits before starting motion.
            wrapped = self._wait_future(
                result_future,
                timeout_sec=max(45.0, duration + 45.0),
                label=f"result:{label}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            cancel_detail = self._cancel_ur5e_hardware_trajectory_goal(
                goal_handle,
                label=label,
            )
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: result failed ({exc}); "
                f"{cancel_detail}"
            )
            return False
        if wrapped is None:
            cancel_detail = self._cancel_ur5e_hardware_trajectory_goal(
                goal_handle,
                label=label,
            )
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: result timeout; {cancel_detail}"
            )
            return False
        result = getattr(wrapped, "result", None)
        try:
            goal_status = int(getattr(wrapped, "status", -1))
            error_code = int(getattr(result, "error_code", -1))
        except (TypeError, ValueError):
            goal_status = -1
            error_code = -1
        if goal_status != 4 or error_code != 0:
            error_string = str(getattr(result, "error_string", "") or "").strip()
            detail = f"goal_status={goal_status} error_code={error_code}"
            if error_string:
                detail = f"{detail} {error_string}"
            self._last_failure_message = (
                f"{self._ur5e_hardware_trajectory_action}: {detail}"
            )
            return False
        self._last_failure_message = ""
        return True

    def move_joints(self, positions: list[float], duration_sec: float = 2.0) -> bool:
        """Replay physical UR5e joint waypoints through the guarded RTDE action."""
        return self._command_ur5e_hardware_trajectory_action(
            positions,
            duration_sec=duration_sec,
            label="move_joints",
        )

    def move_to_named_pose(
        self,
        pose_name: str,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """Move to one configured physical UR5e named position through RTDE."""
        positions = self.named_positions.get(str(pose_name))
        if not isinstance(positions, (list, tuple)) or not positions:
            available = sorted(self.named_positions.keys()) if self.named_positions else []
            return {
                "success": False,
                "message": f"unknown pose '{pose_name}'; available={available}",
            }
        duration_sec = self._scaled_joint_duration(self.named_pose_duration_sec, speed)
        if self._command_ur5e_hardware_trajectory_action(
            list(positions),
            duration_sec=duration_sec,
            label=f"move_to_named_pose:{pose_name}",
        ):
            return {"success": True, "message": f"moved to named pose '{pose_name}'"}
        detail = str(self._last_failure_message or "").strip()
        message = f"failed to move to named pose '{pose_name}'"
        if detail:
            message = f"{message}: {detail}"
        return {"success": False, "message": message}

    def move_home(self, speed: float | None = None) -> dict[str, Any]:
        """Move to the configured physical UR5e home position through RTDE."""
        if not self.wait_for_services():
            detail = str(self._last_failure_message or "").strip()
            message = "services not ready"
            if detail:
                message = f"{message}: {detail}"
            return {"success": False, "message": message}
        home = self.named_positions.get("home")
        if not isinstance(home, (list, tuple)) or not home:
            return {"success": False, "message": "no home pose available"}
        target_positions = [float(position) for position in home]
        duration_sec = self._scaled_joint_duration(self.move_home_duration_sec, speed)
        if self._command_ur5e_hardware_trajectory_action(
            target_positions,
            duration_sec=duration_sec,
            label="move_home",
        ):
            self._last_start_pose = None
            return {"success": True, "message": "moved to named home pose"}
        detail = str(self._last_failure_message or "").strip()
        message = "failed to move to named home pose"
        if detail:
            message = f"{message}: {detail}"
        return {"success": False, "message": message}

    def _command_rg2_gripper_action(self, position: float, label: str) -> bool:
        try:
            target_position = float(position)
        except (TypeError, ValueError, OverflowError):
            self._last_failure_message = "RG2 gripper position is invalid"
            return False
        lower_position = min(float(self.gripper_close), float(self.gripper_open))
        upper_position = max(float(self.gripper_close), float(self.gripper_open))
        if not math.isfinite(target_position):
            self._last_failure_message = "RG2 gripper position must be finite"
            return False
        if not lower_position <= target_position <= upper_position:
            self._last_failure_message = (
                f"RG2 gripper position {target_position:.6f} is outside "
                f"[{lower_position:.6f}, {upper_position:.6f}]"
            )
            return False
        if not self.wait_for_services():
            return False
        client = self._rg2_action_client
        action_type = self._FollowJointTrajectory
        if client is None or action_type is None:
            self._last_failure_message = "RG2 gripper action client is unavailable"
            return False
        try:
            action_ready = bool(client.wait_for_server(timeout_sec=2.0))
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: wait failed ({exc})"
            return False
        if not action_ready:
            self._last_failure_message = f"{self._rg2_action_name} is unavailable"
            return False

        goal = action_type.Goal()
        goal.trajectory.joint_names = [self.gripper_joint]
        point = self._JointTrajectoryPoint()
        point.positions = [target_position]
        duration = max(0.05, float(self.gripper_move_time_sec))
        sec = int(duration)
        nsec = int((duration - sec) * 1_000_000_000)
        point.time_from_start = self._Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]

        try:
            send_future = client.send_goal_async(goal)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: send failed ({exc})"
            return False
        try:
            goal_handle = self._wait_future(
                send_future,
                timeout_sec=3.0,
                label=f"send:{label}",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: send failed ({exc})"
            return False
        if goal_handle is None:
            self._last_failure_message = f"{self._rg2_action_name}: send timeout"
            return False
        if not bool(getattr(goal_handle, "accepted", False)):
            self._last_failure_message = f"{self._rg2_action_name}: goal rejected"
            return False

        try:
            result_future = goal_handle.get_result_async()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: result failed ({exc})"
            return False
        result_timeout = max(
            8.0,
            duration + float(self.gripper_feedback_timeout_pad_sec) + 4.0,
        )
        try:
            wrapped = self._wait_future(
                result_future,
                timeout_sec=result_timeout,
                label=f"result:{label}",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_failure_message = f"{self._rg2_action_name}: result failed ({exc})"
            return False
        if wrapped is None:
            self._last_failure_message = f"{self._rg2_action_name}: result timeout"
            return False
        result = getattr(wrapped, "result", None)
        try:
            goal_status = int(getattr(wrapped, "status", -1))
            error_code = int(getattr(result, "error_code", -1))
        except (TypeError, ValueError):
            goal_status = -1
            error_code = -1
        if goal_status != 4 or error_code != 0:
            error_string = str(getattr(result, "error_string", "") or "").strip()
            detail = f"goal_status={goal_status} error_code={error_code}"
            if error_string:
                detail = f"{detail} {error_string}"
            self._last_failure_message = f"{self._rg2_action_name}: {detail}"
            return False
        self._last_failure_message = ""
        return True

    def open_gripper(self) -> bool:
        return self._command_rg2_gripper_action(self.gripper_open, "open_gripper")

    def close_gripper(self, position: float | None = None) -> bool:
        target = self.gripper_close if position is None else float(position)
        return self._command_rg2_gripper_action(target, "close_gripper")

    def shutdown(self) -> None:
        """Release physical action clients before destroying their ROS node."""
        arm_client = getattr(self, "_ur5e_hardware_trajectory_client", None)
        gripper_client = getattr(self, "_rg2_action_client", None)
        self._ur5e_hardware_trajectory_client = None
        self._rg2_action_client = None
        self._FollowJointTrajectory = None
        for client in (arm_client, gripper_client):
            if client is None:
                continue
            destroy = getattr(client, "destroy", None)
            if callable(destroy):
                with suppress(AttributeError, RuntimeError):
                    destroy()
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
