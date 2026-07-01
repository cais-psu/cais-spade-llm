"""UR5e OnRobot RG2 control over UR RTDE custom script execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


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
    ) -> "UR5eRG2GripperControllerSettings":
        gripper = dict(config or {})
        rtde = dict(gripper.get("rtde") or {})
        resolved_hostname = (
            str(hostname or "").strip()
            or str(rtde.get("hostname") or "").strip()
            or cls.hostname
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
    """Small RG2 commander independent of test/demo files."""

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
    ) -> "UR5eRG2GripperController":
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
        from rtde_control import RTDEControlInterface

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
        width = settings.close_width_mm + ratio * (
            settings.open_width_mm - settings.close_width_mm
        )
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
            return settings.open_settle_sec if float(position) >= midpoint else settings.close_settle_sec
        return settings.open_settle_sec if float(position) <= midpoint else settings.close_settle_sec

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
