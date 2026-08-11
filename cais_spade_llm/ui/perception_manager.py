"""Three-camera configuration and operator actions for the Perception page."""

from __future__ import annotations

import grp
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from cais_spade_llm.ui.bridge import SystemBridge

CAMERA_ROLES = ("ur5e", "xarm6", "stationary")
CONFIG_PATH = Path("~/.config/cais-spade-llm/perception_cameras.yaml").expanduser()
PREVIEW_ROOT = Path("/tmp/cais_perception_previews")
SNAPSHOT_ROOT = Path("/tmp")
CALIBRATION_ROOT = Path("~/.config/cais-spade-llm").expanduser()
DIAGNOSTIC_ROOT = Path("~/.local/share/cais-spade-llm/perception").expanduser()
CAMERA_STARTUP_GRACE_SEC = 1.0
USB_ATTACH_WAIT_SEC = 15.0
USB_ATTACH_POLL_SEC = 0.5
CAMERA_FIRST_FRAME_GRACE_SEC = 20.0
CAMERA_FRAME_STALE_RECOVERY_SEC = 5.0
CAMERA_RECOVERY_DELAYS_SEC = (2.0, 5.0, 10.0)
CAMERA_RESET_SETTLE_SEC = 2.0
CAMERA_DEVICE_DISCOVERY_CACHE_SEC = 30.0
CAMERA_DEVICE_DISCOVERY_FAILURE_CACHE_SEC = 10.0
MINIMUM_CALIBRATION_JOINT_DELTA_RAD = math.radians(5.0)
UR5E_CALIBRATION_MONITOR_STATUS = Path(
    "/tmp/cais_ur5e_calibration_monitor_status.json"
)
UR5E_RTDE_TRAJECTORY_STATUS = Path("/tmp/cais_ur5e_rtde_trajectory_status.json")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_CONFIG: dict[str, Any] = {
    "version": 1,
    "cameras": {
        "ur5e": {
            "serial": "",
            "camera_namespace": "camera",
            "camera_name": "camera",
            "parent_frame": "tool0",
            "calibration_mode": "hand_eye",
            "calibration_path": str(CALIBRATION_ROOT / "ur5e_realsense_hand_eye.yaml"),
            "samples_path": "/tmp/ur5e_hand_eye_samples.json",
            "poses_path": str(CALIBRATION_ROOT / "ur5e_calibration_poses.yaml"),
            "joint_state_topic": "/joint_states",
            "planning_group": "ur_manipulator",
            "arm_joint_candidates": [
                [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                [
                    "ur5e_shoulder_pan_joint",
                    "ur5e_shoulder_lift_joint",
                    "ur5e_elbow_joint",
                    "ur5e_wrist_1_joint",
                    "ur5e_wrist_2_joint",
                    "ur5e_wrist_3_joint",
                ],
            ],
            "color_profile": "640x480x6",
            "depth_profile": "640x480x6",
            "authoritative": True,
        },
        "xarm6": {
            "serial": "",
            "camera_namespace": "xarm6_camera",
            "camera_name": "xarm6_camera",
            "parent_frame": "xarm6_link_eef",
            "calibration_mode": "hand_eye",
            "calibration_path": str(CALIBRATION_ROOT / "xarm6_realsense_hand_eye.yaml"),
            "samples_path": "/tmp/xarm6_hand_eye_samples.json",
            "poses_path": str(CALIBRATION_ROOT / "xarm6_calibration_poses.yaml"),
            "joint_state_topic": "/xarm6/xarm/joint_states",
            "planning_group": "xarm6",
            "arm_joint_candidates": [
                ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
                [
                    "xarm6_joint1",
                    "xarm6_joint2",
                    "xarm6_joint3",
                    "xarm6_joint4",
                    "xarm6_joint5",
                    "xarm6_joint6",
                ],
            ],
            "color_profile": "640x480x6",
            "depth_profile": "640x480x6",
            "authoritative": False,
        },
        "stationary": {
            "serial": "",
            "camera_namespace": "stationary_camera",
            "camera_name": "stationary_camera",
            "parent_frame": "world",
            "calibration_mode": "stationary",
            "calibration_path": str(CALIBRATION_ROOT / "stationary_realsense_extrinsic.yaml"),
            "samples_path": "/tmp/stationary_camera_samples.json",
            "poses_path": "",
            "color_profile": "640x480x6",
            "depth_profile": "640x480x6",
            "authoritative": False,
        },
    },
    "stationary_board_world_pose": {
        "configured": False,
        "x": 0.0,
        "y": 0.0,
        "z": 1.015,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
}


def _atomic_yaml_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_realsense_devices(output: str) -> list[dict[str, str]]:
    """Parse `rs-enumerate-devices` output into stable device rows."""
    devices: list[dict[str, str]] = []
    current: dict[str, str] = {}
    key_map = {
        "name": "model",
        "serial number": "serial",
        "firmware version": "firmware",
        "usb type descriptor": "usb_type",
        "physical port": "physical_port",
    }
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([^:]+):\s*(.+)$", line)
        if match is None:
            continue
        key = key_map.get(match.group(1).strip().lower())
        if key is None:
            continue
        if key == "serial" and current.get("serial"):
            devices.append(current)
            current = {}
        current[key] = match.group(2).strip()
    if current.get("serial"):
        devices.append(current)
    unique: dict[str, dict[str, str]] = {}
    for row in devices:
        unique[str(row["serial"])] = row
    return [unique[serial] for serial in sorted(unique)]


def parse_usbipd_realsense_devices(output: str) -> list[dict[str, str]]:
    """Parse RealSense rows from Windows `usbipd list` output."""
    rows: list[dict[str, str]] = []
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(\d+-\d+)\s+([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\s+(.+)$", line)
        if match is None or "realsense" not in match.group(3).lower():
            continue
        description_and_state = match.group(3).strip()
        state_match = re.search(
            r"\s{2,}(Not shared|Shared|Attached)\s*$",
            description_and_state,
            flags=re.IGNORECASE,
        )
        state = state_match.group(1) if state_match is not None else "Unknown"
        description = (
            description_and_state[: state_match.start()].strip()
            if state_match is not None
            else description_and_state
        )
        state = {
            "attached": "Attached",
            "shared": "Shared",
            "not shared": "Not shared",
        }.get(state.lower(), "Unknown")
        rows.append(
            {
                "busid": match.group(1),
                "vid_pid": match.group(2).lower(),
                "description": description,
                "state": state,
            }
        )
    return rows


def validate_camera_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate assignments while retaining the three exact camera roles."""
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("perception camera configuration is missing cameras")
    missing = [role for role in CAMERA_ROLES if role not in cameras]
    if missing:
        raise ValueError("perception camera configuration is missing: " + ", ".join(missing))
    serials = [
        str((cameras.get(role) or {}).get("serial") or "").strip() for role in CAMERA_ROLES
    ]
    assigned = [serial for serial in serials if serial]
    if len(assigned) != len(set(assigned)):
        raise ValueError("the same RealSense serial cannot be assigned to multiple camera roles")
    for role in CAMERA_ROLES:
        camera = cameras.get(role)
        if not isinstance(camera, dict):
            raise ValueError(f"camera role {role} must be a mapping")
        for field in ("camera_namespace", "camera_name", "parent_frame", "calibration_path"):
            if not str(camera.get(field) or "").strip():
                raise ValueError(f"camera role {role} is missing {field}")
    return payload


class PerceptionManager:
    """Own camera deployment state while `SystemBridge` owns subprocesses."""

    def __init__(self, bridge: SystemBridge, *, project_root: Path, venv_python: Path) -> None:
        self.bridge = bridge
        self.project_root = Path(project_root)
        self.venv_python = Path(venv_python)
        self.config_path = CONFIG_PATH
        self._devices_cache: list[dict[str, str]] = []
        self._devices_cache_at = 0.0
        self._device_discovery_error = ""
        self._wsl_devices_cache: list[dict[str, str]] = []
        self._wsl_devices_cache_at = 0.0
        self._devices_discovery_lock = threading.Lock()
        self._wsl_devices_discovery_lock = threading.Lock()
        self._connection_lock = threading.RLock()
        self._desired_connected: set[str] = set()
        self._desired_perception: set[str] = set()
        self._recovery: dict[str, dict[str, Any]] = {
            role: {
                "state": "disconnected",
                "attempt_count": 0,
                "last_attempt_at": None,
                "next_retry_at": None,
                "last_error": "",
                "connected_at": None,
            }
            for role in CAMERA_ROLES
        }
        self.diagnostic_root = DIAGNOSTIC_ROOT
        try:
            (self.diagnostic_root / "logs").mkdir(parents=True, exist_ok=True)
        except OSError:
            self.diagnostic_root = Path("/tmp/cais_perception_diagnostics")
            (self.diagnostic_root / "logs").mkdir(parents=True, exist_ok=True)
        self._ensure_config()

    def _ensure_config(self) -> None:
        if self.config_path.is_file():
            return
        payload = deepcopy(DEFAULT_CAMERA_CONFIG)
        legacy_serial = str(os.environ.get("REALSENSE_SERIAL", "")).strip()
        if legacy_serial:
            payload["cameras"]["ur5e"]["serial"] = legacy_serial
        _atomic_yaml_write(self.config_path, payload)

    def config(self) -> dict[str, Any]:
        """Return merged machine-local camera configuration."""
        try:
            saved = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            saved = {}
        merged = deepcopy(DEFAULT_CAMERA_CONFIG)
        if isinstance(saved, dict):
            saved_cameras = saved.get("cameras")
            if isinstance(saved_cameras, dict):
                for role in CAMERA_ROLES:
                    row = saved_cameras.get(role)
                    if isinstance(row, dict):
                        merged["cameras"][role].update(row)
            board_pose = saved.get("stationary_board_world_pose")
            if isinstance(board_pose, dict):
                merged["stationary_board_world_pose"].update(board_pose)
        return validate_camera_config(merged)

    def save_assignments(self, assignments: dict[str, str]) -> dict[str, Any]:
        """Persist camera serial assignments after duplicate validation."""
        payload = self.config()
        for role in CAMERA_ROLES:
            if role in assignments:
                payload["cameras"][role]["serial"] = str(assignments[role] or "").strip()
        validate_camera_config(payload)
        _atomic_yaml_write(self.config_path, payload)
        return payload

    def save_stationary_board_pose(self, pose: dict[str, Any]) -> dict[str, Any]:
        """Persist the surveyed fixed-board pose used by the stationary camera."""
        payload = self.config()
        values = {name: float(pose[name]) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
        payload["stationary_board_world_pose"] = {"configured": True, **values}
        _atomic_yaml_write(self.config_path, payload)
        return payload

    @staticmethod
    def _run(command: list[str], *, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)

    def discover_devices(self, *, force: bool = False) -> list[dict[str, str]]:
        """Discover RealSense devices without requiring root."""
        cache_age = time.monotonic() - self._devices_cache_at
        cache_ttl = (
            CAMERA_DEVICE_DISCOVERY_FAILURE_CACHE_SEC
            if self._device_discovery_error
            else CAMERA_DEVICE_DISCOVERY_CACHE_SEC
        )
        if not force and 0.0 <= cache_age < cache_ttl:
            return deepcopy(self._devices_cache)
        with self._devices_discovery_lock:
            cache_age = time.monotonic() - self._devices_cache_at
            cache_ttl = (
                CAMERA_DEVICE_DISCOVERY_FAILURE_CACHE_SEC
                if self._device_discovery_error
                else CAMERA_DEVICE_DISCOVERY_CACHE_SEC
            )
            if not force and 0.0 <= cache_age < cache_ttl:
                return deepcopy(self._devices_cache)
            executable = shutil.which("rs-enumerate-devices")
            if executable is None:
                self._device_discovery_error = "rs-enumerate-devices is unavailable"
                self._devices_cache_at = time.monotonic()
                return deepcopy(self._devices_cache)
            try:
                result = self._run([executable], timeout=12.0)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._device_discovery_error = f"RealSense discovery failed: {exc}"
                self._devices_cache_at = time.monotonic()
                return deepcopy(self._devices_cache)
            devices = parse_realsense_devices(result.stdout + "\n" + result.stderr)
            if result.returncode != 0:
                lines = [
                    line.strip()
                    for line in (result.stderr + "\n" + result.stdout).splitlines()
                    if line.strip()
                ]
                detail = next(
                    (
                        line
                        for line in lines
                        if "could not initialize udev monitor" in line.lower()
                    ),
                    lines[-1] if lines else "rs-enumerate-devices failed",
                )
                self._device_discovery_error = detail
                self._devices_cache_at = time.monotonic()
                return deepcopy(self._devices_cache)
            self._device_discovery_error = ""
            self._devices_cache = devices
            self._devices_cache_at = time.monotonic()
            return deepcopy(devices)

    def discover_wsl_attachments(self, *, force: bool = False) -> list[dict[str, str]]:
        """List Windows RealSense USB rows available through usbipd-win."""
        if "microsoft" not in Path("/proc/version").read_text(encoding="utf-8").lower():
            return []
        if (
            not force
            and 0.0 <= time.monotonic() - self._wsl_devices_cache_at
            < CAMERA_DEVICE_DISCOVERY_CACHE_SEC
        ):
            return deepcopy(self._wsl_devices_cache)
        with self._wsl_devices_discovery_lock:
            if (
                not force
                and 0.0 <= time.monotonic() - self._wsl_devices_cache_at
                < CAMERA_DEVICE_DISCOVERY_CACHE_SEC
            ):
                return deepcopy(self._wsl_devices_cache)
            executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
            if executable is None:
                self._wsl_devices_cache_at = time.monotonic()
                return deepcopy(self._wsl_devices_cache)
            try:
                result = self._run(
                    [executable, "-NoProfile", "-Command", "usbipd list"],
                    timeout=8.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                self._wsl_devices_cache_at = time.monotonic()
                return deepcopy(self._wsl_devices_cache)
            rows = parse_usbipd_realsense_devices(result.stdout)
            self._wsl_devices_cache = rows
            self._wsl_devices_cache_at = time.monotonic()
            return deepcopy(rows)

    def attach_wsl_camera(self, busid: str) -> str | None:
        """Attach a previously administrator-bound Windows USB device to WSL."""
        normalized = str(busid or "").strip()
        if re.fullmatch(r"\d+-\d+", normalized) is None:
            return "select a valid RealSense BUSID first"
        rows = self.discover_wsl_attachments(force=True)
        selected = next(
            (row for row in rows if str(row.get("busid") or "") == normalized),
            None,
        )
        if selected is None:
            return f"RealSense BUSID {normalized} is not present in Windows USB inventory"
        state = str(selected.get("state") or "Unknown")
        if state == "Not shared":
            return (
                f"RealSense BUSID {normalized} is Not shared. In Administrator Windows "
                f"PowerShell run: usbipd bind --busid {normalized}. Then refresh WSL USB "
                "and attach it."
            )
        if state == "Attached":
            self._devices_cache_at = 0.0
            self._wsl_devices_cache_at = 0.0
            return None
        executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if executable is None:
            return "Windows PowerShell interoperability is unavailable"
        try:
            result = self._run(
                [
                    executable,
                    "-NoProfile",
                    "-Command",
                    f"usbipd attach --wsl --busid {normalized}",
                ],
                timeout=20.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"usbipd attach failed: {exc}"
        if result.returncode != 0:
            return result.stderr.strip() or result.stdout.strip() or "usbipd attach failed"
        self._devices_cache_at = 0.0
        self._wsl_devices_cache_at = 0.0
        return None

    def _ensure_assigned_camera_available(self, serial: str) -> str | None:
        """Attach bound RealSense rows until the assigned serial is visible in WSL."""
        normalized = str(serial or "").strip()
        if any(row.get("serial") == normalized for row in self.discover_devices(force=True)):
            return None
        if "microsoft" not in Path("/proc/version").read_text(encoding="utf-8").lower():
            return f"RealSense serial {normalized} is not connected"

        rows = self.discover_wsl_attachments(force=True)
        shared = [row for row in rows if row.get("state") == "Shared"]
        attached = [row for row in rows if row.get("state") == "Attached"]
        not_shared = [row for row in rows if row.get("state") == "Not shared"]
        attach_errors: list[str] = []
        for row in shared:
            error = self.attach_wsl_camera(str(row.get("busid") or ""))
            if error:
                attach_errors.append(f"{row.get('busid')}: {error}")

        deadline = time.monotonic() + max(0.0, USB_ATTACH_WAIT_SEC)
        while True:
            if any(
                row.get("serial") == normalized for row in self.discover_devices(force=True)
            ):
                return None
            if time.monotonic() >= deadline:
                break
            time.sleep(min(USB_ATTACH_POLL_SEC, max(0.0, deadline - time.monotonic())))

        if attach_errors:
            return "RealSense USB attachment failed: " + "; ".join(attach_errors)
        if not_shared and not shared and not attached:
            commands = "; ".join(
                f"usbipd bind --busid {str(row.get('busid') or '')}"
                for row in not_shared
            )
            return (
                "RealSense USB is not shared with usbipd-win. In Administrator Windows "
                f"PowerShell run the one-time command: {commands}. The UI never requests "
                "administrator credentials."
            )
        if not rows:
            return "No Windows RealSense USB rows were found; reconnect the camera and retry."
        return (
            f"RealSense serial {normalized} did not appear in WSL after USB attachment; "
            "check device permissions and the System Preflight section."
        )

    def _reset_recovery(self, role: str) -> None:
        self._recovery[role] = {
            "state": "connecting",
            "attempt_count": 0,
            "last_attempt_at": None,
            "next_retry_at": None,
            "last_error": "",
            "connected_at": time.time(),
        }

    def _set_recovery_error(self, role: str, message: str) -> None:
        state = self._recovery[role]
        state["state"] = "degraded"
        state["next_retry_at"] = None
        state["last_error"] = str(message or "camera recovery failed")

    def _stop_camera_stack(self, role: str) -> None:
        names = self._process_names(role)
        for process in (
            "calibration",
            "calibration_preview",
            "calibration_state_publisher",
            "calibration_rtde_monitor",
            "viewer",
            "perception",
            "preview",
            "camera",
        ):
            self.bridge.ros2_stop(names[process])
        if role == "ur5e":
            for process_name in (
                "digital_twin_ur5e_only_physical_perception",
                "digital_twin_dual_robots_physical_perception",
                "digital_twin_ur5e_only_realsense_camera",
                "digital_twin_dual_robots_realsense_camera",
            ):
                if self.bridge.ros2_proc_status(process_name) == "running":
                    self.bridge.ros2_stop(process_name)

    @staticmethod
    def _stale_process_fragments(
        role: str,
        camera: dict[str, Any],
        process: str,
    ) -> tuple[str, ...]:
        """Return the exact CAIS command fragments for one camera process."""
        if process == "camera":
            return (
                "realsense2_camera_node",
                f"-r __node:={camera['camera_name']}",
                f"-r __ns:=/{camera['camera_namespace']}",
            )
        if process == "preview":
            return (
                "cais_spade_llm.resources.sensor.physical.realsense_preview_node",
                f"--camera-role {role}",
                f"--output-root {PREVIEW_ROOT}",
            )
        if process == "perception":
            return (
                "cais_spade_llm.resources.sensor.physical.realsense_roboflow_node",
                f"camera_role:={role}",
                f"detect_all_service:=/perception/{role}/detect_all",
            )
        raise ValueError(f"unknown camera process: {process}")

    @staticmethod
    def _matching_stale_process_groups(
        required_fragments: tuple[str, ...],
        *,
        proc_root: Path = Path("/proc"),
    ) -> set[int]:
        """Find CAIS process groups through any surviving matching process."""
        current_group = os.getpgrp()
        groups: set[int] = set()
        for entry in proc_root.glob("[0-9]*"):
            try:
                pid = int(entry.name)
                process_group = os.getpgid(pid)
                if process_group == current_group:
                    continue
                command = (
                    (entry / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode("utf-8", errors="replace")
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
                continue
            if all(fragment in command for fragment in required_fragments):
                groups.add(process_group)
        return groups

    @staticmethod
    def _process_group_alive(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _stop_stale_role_processes(
        self,
        role: str,
        camera: dict[str, Any],
        process: str,
    ) -> str | None:
        """Stop only orphaned CAIS processes matching one camera role and purpose."""
        groups = self._matching_stale_process_groups(
            self._stale_process_fragments(role, camera, process)
        )
        if not groups:
            return None
        errors: list[str] = []
        for process_group in groups:
            try:
                os.killpg(process_group, signal.SIGINT)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                errors.append(f"process group {process_group}: {exc}")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and any(
            self._process_group_alive(process_group) for process_group in groups
        ):
            time.sleep(0.05)
        for process_group in groups:
            if not self._process_group_alive(process_group):
                continue
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                errors.append(f"process group {process_group}: {exc}")
        if errors:
            return "Could not stop stale CAIS camera processes: " + "; ".join(errors)
        logger.info(
            "Stopped stale CAIS %s process groups for %s: %s",
            process,
            role,
            sorted(groups),
        )
        return None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _log_tail(path: Path, *, line_count: int = 20) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-line_count:])

    @staticmethod
    def _last_process_error(path: Path) -> str:
        """Return the newest actionable process error from a diagnostic log."""
        tail = PerceptionManager._log_tail(path, line_count=120)
        for raw_line in reversed(tail.splitlines()):
            line = _ANSI_ESCAPE.sub("", raw_line).strip()
            lowered = line.lower()
            if "process has died" in lowered:
                continue
            if "device has been disconnected" in lowered or any(
                marker.lower() in lowered
                for marker in (
                    "[ERROR]",
                    "Permission denied",
                    "No RealSense devices were found",
                    "Wrong parameter type",
                )
            ):
                return line
        return ""

    def _camera(self, role: str) -> dict[str, Any]:
        key = str(role or "").strip().lower()
        if key not in CAMERA_ROLES:
            raise ValueError(f"unknown camera role: {role}")
        return dict(self.config()["cameras"][key])

    @staticmethod
    def _topics(camera: dict[str, Any]) -> dict[str, str]:
        root = f"/{camera['camera_namespace']}/{camera['camera_name']}"
        return {
            "color": f"{root}/color/image_raw",
            "depth": f"{root}/aligned_depth_to_color/image_raw",
            "camera_info": f"{root}/color/camera_info",
        }

    @staticmethod
    def _snapshot_path(role: str) -> Path:
        if role == "ur5e":
            return SNAPSHOT_ROOT / "cais_physical_perception.json"
        return SNAPSHOT_ROOT / f"cais_physical_perception_{role}.json"

    @staticmethod
    def _process_names(role: str) -> dict[str, str]:
        return {
            "camera": "realsense_camera" if role == "ur5e" else f"realsense_camera_{role}",
            "preview": f"realsense_preview_{role}",
            "perception": "physical_perception" if role == "ur5e" else f"physical_perception_{role}",
            "viewer": f"realsense_viewer_{role}",
            "calibration": f"realsense_calibration_{role}",
            "calibration_preview": f"realsense_calibration_preview_{role}",
            "calibration_rtde_monitor": f"{role}_calibration_rtde_monitor",
            "calibration_state_publisher": f"{role}_calibration_state_publisher",
        }

    def _domain_id(self) -> int:
        hardware_processes = getattr(self.bridge, "_DIGITAL_TWIN_HARDWARE_PROCESS_NAMES", set())
        active_from_status = getattr(self.bridge, "_active_digital_twin_target_from_status", None)
        active_target = active_from_status() if callable(active_from_status) else None
        if any(
            self.bridge.ros2_proc_status(name) == "running" for name in hardware_processes
        ) or active_target is not None:
            domains = self.bridge._digital_twin_domain_ids()
            target_lookup = getattr(self.bridge, "_digital_twin_target", None)
            domain_lookup = getattr(self.bridge, "_digital_twin_hardware_domain_id", None)
            target_cfg = (
                target_lookup(active_target)
                if active_target and callable(target_lookup)
                else None
            )
            if isinstance(target_cfg, dict) and callable(domain_lookup):
                return int(
                    domain_lookup(
                        target_cfg,
                        "ur5e",
                        domains,
                    )
                )
            return int(domains["hardware"])
        return int(self.bridge._default_ros_domain_id())

    def _ur5e_digital_twin_process_running(self, process: str) -> bool:
        names = {
            "camera": {
                "digital_twin_ur5e_only_realsense_camera",
                "digital_twin_dual_robots_realsense_camera",
            },
            "perception": {
                "digital_twin_ur5e_only_physical_perception",
                "digital_twin_dual_robots_physical_perception",
            },
        }
        return any(
            self.bridge.ros2_proc_status(name) == "running"
            for name in names.get(process, set())
        )

    def _ur5e_full_state_publisher_running(self) -> bool:
        """Return whether this UI owns a live full UR5e state publisher."""
        process_names = {
            "hardware_ur5e_moveit",
            "digital_twin_ur5e_only_hardware_ur5e_moveit",
            "digital_twin_dual_robots_hardware_moveit",
        }
        if any(
            self.bridge.ros2_proc_status(name) == "running"
            for name in process_names
        ):
            return True
        if self.bridge.ros2_proc_status("hardware_robot_state_publisher") != "running":
            return False
        active_stack_lookup = getattr(
            self.bridge,
            "_active_normal_hardware_stack",
            None,
        )
        selected_stack_lookup = getattr(
            self.bridge,
            "_selected_normal_hardware_stack",
            None,
        )
        active_stack = (
            str(active_stack_lookup() or "").strip().lower()
            if callable(active_stack_lookup)
            else ""
        )
        selected_stack = (
            str(selected_stack_lookup() or "").strip().lower()
            if callable(selected_stack_lookup)
            else ""
        )
        return active_stack in {"ur5e", "dual robots"} or selected_stack in {
            "ur5e",
            "dual robots",
        }

    def _ur5e_joint_state_publisher_running(self) -> bool:
        monitor_process = self._process_names("ur5e")["calibration_rtde_monitor"]
        if self.bridge.ros2_proc_status(monitor_process) == "running":
            return True
        process_names = {
            "hardware_ur5e_rtde_trajectory_server",
            "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server",
            "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server",
        }
        if not any(
            self.bridge.ros2_proc_status(name) == "running" for name in process_names
        ):
            return False
        status = self._read_json(UR5E_RTDE_TRAJECTORY_STATUS)
        try:
            status_age_sec = time.time() - float(status["updated_at"])
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            0.0 <= status_age_sec <= 3.0
            and status.get("rtde_receive_connected")
            and status.get("joint_states_fresh")
            and not status.get("rtde_reset_required")
        )

    def _ensure_ur5e_calibration_monitor(self, *, domain_id: int | None = None) -> None:
        """Start only the read-only UR5e state and TF publishers needed by calibration."""
        names = self._process_names("ur5e")
        resolved_domain_id = self._domain_id() if domain_id is None else int(domain_id)
        state_publisher = names["calibration_state_publisher"]
        full_state_publisher_running = self._ur5e_full_state_publisher_running()
        if full_state_publisher_running:
            if self.bridge.ros2_proc_status(state_publisher) == "running":
                self.bridge.ros2_stop(state_publisher)
        elif self.bridge.ros2_proc_status(state_publisher) != "running":
            command = (
                "ros2 launch cais_lab_robotics ur5e_rg2_hardware_moveit.launch.py "
                "launch_move_group:=false launch_rviz:=false"
            )
            error = self.bridge._start_tracked_ros2_command(
                state_publisher,
                self._logged_command(state_publisher, command),
                ros_domain_id=resolved_domain_id,
            )
            if error and "already running" not in error:
                raise RuntimeError(f"cannot start UR5e calibration state publisher: {error}")

        rtde_monitor = names["calibration_rtde_monitor"]
        if not self._ur5e_joint_state_publisher_running():
            robot_ip = str(self.bridge.get_hardware_ips().get("ur5e") or "").strip()
            if not robot_ip:
                raise RuntimeError("UR5e robot_ip is unavailable for calibration monitoring")
            with suppress(OSError):
                UR5E_CALIBRATION_MONITOR_STATUS.unlink(missing_ok=True)
            server_path = (
                self.project_root
                / "ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py"
            )
            config_path = (
                self.project_root
                / "ros2/cais_lab_robotics/config/hardware_runtime/"
                "xarm6_ur5e_hardware_runtime.yaml"
            )
            command = (
                f"{self.venv_python} {server_path} --robot-ip {shlex.quote(robot_ip)} "
                f"--status-file {UR5E_CALIBRATION_MONITOR_STATUS} "
                f"--config {config_path} --monitor-only"
            )
            error = self.bridge._start_tracked_ros2_command(
                rtde_monitor,
                self._logged_command(rtde_monitor, command),
                ros_domain_id=resolved_domain_id,
            )
            if error and "already running" not in error:
                self.bridge.ros2_stop(state_publisher)
                raise RuntimeError(f"cannot start read-only UR5e calibration monitor: {error}")

        if self.bridge.ros2_proc_status(rtde_monitor) == "running":
            deadline = time.monotonic() + 10.0
            status: dict[str, Any] = {}
            while time.monotonic() < deadline:
                status = self._read_json(UR5E_CALIBRATION_MONITOR_STATUS)
                if bool(status.get("rtde_receive_connected")) and bool(
                    status.get("joint_states_fresh")
                ):
                    return
                if self.bridge.ros2_proc_status(rtde_monitor) != "running":
                    break
                time.sleep(0.1)
            detail = str(
                status.get("blocked_reason") or status.get("message") or ""
            ).strip()
            raise RuntimeError(
                "read-only UR5e calibration monitor did not provide fresh joint feedback"
                + (f": {detail}" if detail else "")
            )

        result = self._run_ros_command(
            [
                "timeout",
                "10",
                "ros2",
                "topic",
                "echo",
                "--once",
                "--no-daemon",
                "--spin-time",
                "2.0",
                "/joint_states",
                "sensor_msgs/msg/JointState",
            ],
            timeout=12.0,
        )
        if result.returncode == 0:
            return
        status = self._read_json(UR5E_CALIBRATION_MONITOR_STATUS)
        detail = str(status.get("blocked_reason") or status.get("message") or "").strip()
        if not detail:
            detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "read-only UR5e calibration monitor did not publish /joint_states"
            + (f": {detail}" if detail else "")
        )

    def ensure_ur5e_calibration_monitor(self, *, domain_id: int | None = None) -> None:
        """Ensure read-only UR5e joint and TF feedback without enabling robot control."""
        self._ensure_ur5e_calibration_monitor(domain_id=domain_id)

    def _reconcile_part_twin_sync(self, role: str) -> None:
        """Restore late-start gear mirroring without coupling camera readiness to Gazebo."""
        if role != "ur5e":
            return
        reconcile = getattr(self.bridge, "_reconcile_physical_part_twin_sync", None)
        if not callable(reconcile):
            return
        error = reconcile()
        if error:
            logger.warning("Physical part twin synchronization is waiting: %s", error)

    def _log_path(self, process_name: str) -> Path:
        return self.diagnostic_root / "logs" / f"{process_name}.log"

    def _logged_command(self, process_name: str, command: str) -> str:
        return f"{command} >> {shlex.quote(str(self._log_path(process_name)))} 2>&1"

    def _clean_stale_stack_before_start(
        self,
        key: str,
        camera: dict[str, Any],
        names: dict[str, str],
        *,
        digital_twin_camera: bool,
    ) -> str | None:
        """Remove matching orphans only when the current UI owns no live process."""
        if (
            self.bridge.ros2_proc_status(names["camera"]) != "running"
            and not digital_twin_camera
        ):
            cleanup_error = self._stop_stale_role_processes(key, camera, "camera")
            if cleanup_error:
                return cleanup_error
        if self.bridge.ros2_proc_status(names["preview"]) != "running":
            return self._stop_stale_role_processes(key, camera, "preview")
        return None

    def _start_camera_stack(self, key: str) -> str | None:
        """Start an assigned RealSense stack without changing operator intent."""
        camera = self._camera(key)
        serial = str(camera.get("serial") or "").strip()
        if not serial:
            return f"Assign a RealSense serial to {key} before connecting."
        names = self._process_names(key)
        digital_twin_camera = key == "ur5e" and self._ur5e_digital_twin_process_running(
            "camera"
        )
        cleanup_error = self._clean_stale_stack_before_start(
            key,
            camera,
            names,
            digital_twin_camera=digital_twin_camera,
        )
        if cleanup_error:
            return cleanup_error
        attachment_error = self._ensure_assigned_camera_available(serial)
        if attachment_error:
            return attachment_error
        if (
            self.bridge.ros2_proc_status(names["camera"]) != "running"
            and self.bridge.ros2_proc_status(names["preview"]) == "running"
        ):
            self.bridge.ros2_stop(names["preview"])
        camera_log = self._log_path(names["camera"])
        with suppress(OSError):
            camera_log.write_text("", encoding="utf-8")
        command = (
            "ros2 launch cais_lab_robotics realsense_camera.launch.py "
            f"serial_no:={serial} camera_namespace:={camera['camera_namespace']} "
            f"camera_name:={camera['camera_name']} "
            f"rgb_camera.color_profile:={camera['color_profile']} "
            f"depth_module.depth_profile:={camera['depth_profile']}"
        )
        error = None
        if not digital_twin_camera:
            error = self.bridge._start_tracked_ros2_command(
                names["camera"],
                self._logged_command(names["camera"], command),
                ros_domain_id=self._domain_id(),
            )
        if error and "already running" not in error:
            return error
        if CAMERA_STARTUP_GRACE_SEC > 0:
            time.sleep(CAMERA_STARTUP_GRACE_SEC)
        camera_running = self.bridge.ros2_proc_status(names["camera"]) == "running"
        if key == "ur5e" and self._ur5e_digital_twin_process_running("camera"):
            camera_running = True
        detail = self._last_process_error(camera_log)
        if not camera_running or detail:
            if camera_running and not self._ur5e_digital_twin_process_running("camera"):
                self.bridge.ros2_stop(names["camera"])
            return detail or f"{key} RealSense camera stopped during startup; see {camera_log}"

        preview_dir = PREVIEW_ROOT / key
        for filename in ("status.json", "color.jpg", "depth.jpg"):
            with suppress(OSError):
                (preview_dir / filename).unlink(missing_ok=True)
        topics = self._topics(camera)
        expected_rate_hz = str(camera["color_profile"]).rsplit("x", maxsplit=1)[-1]
        preview_command = (
            f"{self.venv_python} -m "
            "cais_spade_llm.resources.sensor.physical.realsense_preview_node "
            f"--camera-role {key} --color-topic {topics['color']} "
            f"--depth-topic {topics['depth']} --output-root {PREVIEW_ROOT}"
            f" --expected-rate-hz {expected_rate_hz}"
        )
        preview_error = self.bridge._start_tracked_ros2_command(
            names["preview"],
            self._logged_command(names["preview"], preview_command),
            ros_domain_id=self._domain_id(),
        )
        if preview_error and "already running" not in preview_error:
            return preview_error
        return None

    def start_camera(self, role: str) -> str | None:
        """Connect one role and enable bounded automatic USB recovery."""
        key = str(role).strip().lower()
        camera = self._camera(key)
        if not str(camera.get("serial") or "").strip():
            return f"Assign a RealSense serial to {key} before connecting."
        with self._connection_lock:
            self._desired_connected.add(key)
            self._reset_recovery(key)
            error = self._start_camera_stack(key)
            if error:
                state = self._recovery[key]
                state["state"] = "waiting"
                state["last_error"] = error
                state["next_retry_at"] = time.time() + CAMERA_RECOVERY_DELAYS_SEC[0]
                return error
            self._recovery[key]["state"] = "connecting"
            self._recovery[key]["connected_at"] = time.time()
            return None

    def stop_camera(self, role: str) -> None:
        """Stop perception, preview, viewer, and driver for one camera role."""
        key = str(role).strip().lower()
        with self._connection_lock:
            self._desired_connected.discard(key)
            self._desired_perception.discard(key)
            self._stop_camera_stack(key)
            self._recovery[key] = {
                "state": "disconnected",
                "attempt_count": 0,
                "last_attempt_at": None,
                "next_retry_at": None,
                "last_error": "",
                "connected_at": None,
            }

    def start_all(self) -> dict[str, str]:
        """Start camera and detection for every assigned role."""
        outcomes: dict[str, str] = {}
        for role in CAMERA_ROLES:
            if not str(self._camera(role).get("serial") or "").strip():
                outcomes[role] = "unassigned"
                continue
            error = self.start_detection(role)
            outcomes[role] = error or "started"
        return outcomes

    def stop_all(self) -> None:
        """Stop detection and the complete camera stack for every role."""
        for role in reversed(CAMERA_ROLES):
            self.stop_detection(role)

    def reconcile_connections(self, *, now: float | None = None) -> dict[str, dict[str, Any]]:
        """Recover explicitly connected cameras after a WSL USB disconnect."""
        current = time.time() if now is None else float(now)
        with self._connection_lock:
            for role in CAMERA_ROLES:
                if role not in self._desired_connected:
                    continue
                names = self._process_names(role)
                camera_running = self.bridge.ros2_proc_status(names["camera"]) == "running"
                if role == "ur5e" and self._ur5e_digital_twin_process_running("camera"):
                    camera_running = True
                preview = self._read_json(PREVIEW_ROOT / role / "status.json")
                frame_at = float(preview.get("frame_captured_at", 0.0) or 0.0)
                frame_fresh = bool(
                    frame_at and current - frame_at <= CAMERA_FRAME_STALE_RECOVERY_SEC
                )
                state = self._recovery[role]
                if camera_running and frame_fresh:
                    perception_error = None
                    if (
                        role in self._desired_perception
                        and self.bridge.ros2_proc_status(names["perception"]) != "running"
                    ):
                        perception_error = self._start_perception_stack(role)
                    if role in self._desired_perception and perception_error is None:
                        self._reconcile_part_twin_sync(role)
                    state["state"] = "connected"
                    state["next_retry_at"] = None
                    state["last_error"] = perception_error or ""
                    continue

                connected_at = float(state.get("connected_at", 0.0) or 0.0)
                startup_error = self._last_process_error(self._log_path(names["camera"]))
                if (
                    camera_running
                    and state.get("state") == "connecting"
                    and connected_at
                    and not startup_error
                    and current - connected_at <= CAMERA_FIRST_FRAME_GRACE_SEC
                ):
                    continue
                if state.get("state") == "degraded":
                    continue
                if state.get("state") not in {"waiting", "recovering"}:
                    detail = startup_error
                    if not detail and not list(Path("/dev").glob("video*")):
                        detail = "RealSense USB attachment vanished: /dev/video* is unavailable"
                    state["state"] = "waiting"
                    state["last_error"] = detail or "camera frames became stale"
                    attempt_count = int(state.get("attempt_count", 0) or 0)
                    state["next_retry_at"] = (
                        current + CAMERA_RECOVERY_DELAYS_SEC[min(attempt_count, 2)]
                    )
                    self._stop_camera_stack(role)

                retry_at = float(state.get("next_retry_at", 0.0) or 0.0)
                if retry_at and current < retry_at:
                    continue
                attempt_count = int(state.get("attempt_count", 0) or 0)
                if attempt_count >= len(CAMERA_RECOVERY_DELAYS_SEC):
                    self._set_recovery_error(role, str(state.get("last_error") or ""))
                    continue

                state["state"] = "recovering"
                state["attempt_count"] = attempt_count + 1
                state["last_attempt_at"] = current
                state["next_retry_at"] = None
                self._stop_camera_stack(role)
                error = self._start_camera_stack(role)
                if error:
                    state["last_error"] = error
                    if state["attempt_count"] >= len(CAMERA_RECOVERY_DELAYS_SEC):
                        self._set_recovery_error(role, error)
                    else:
                        state["state"] = "waiting"
                        state["next_retry_at"] = (
                            current + CAMERA_RECOVERY_DELAYS_SEC[state["attempt_count"]]
                        )
                    continue

                perception_error = None
                if role in self._desired_perception:
                    perception_error = self._start_perception_stack(role)
                    if perception_error is None:
                        self._reconcile_part_twin_sync(role)
                state["state"] = "connecting"
                state["connected_at"] = current
                state["last_error"] = perception_error or ""
            return deepcopy(self._recovery)

    def _start_perception_stack(self, key: str) -> str | None:
        """Start one Roboflow process without changing operator intent."""
        camera = self._camera(key)
        calibration_path = Path(str(camera["calibration_path"])).expanduser()
        if not calibration_path.is_file():
            return f"Calibration is missing for {key}: {calibration_path}"
        if not str(os.environ.get("ROBOFLOW_API_KEY", "")).strip():
            return "ROBOFLOW_API_KEY is not configured in the ignored .env file."
        if key == "ur5e":
            try:
                self._ensure_ur5e_calibration_monitor(domain_id=self._domain_id())
            except RuntimeError as exc:
                return f"UR5e state/TF monitoring is not ready: {exc}"
        if key == "ur5e" and self._ur5e_digital_twin_process_running("perception"):
            return None
        process_name = self._process_names(key)["perception"]
        if self.bridge.ros2_proc_status(process_name) != "running":
            cleanup_error = self._stop_stale_role_processes(key, camera, "perception")
            if cleanup_error:
                return cleanup_error
        topics = self._topics(camera)
        config_path = (
            self.project_root
            / "ros2/cais_lab_robotics/config/perception/realsense_roboflow.yaml"
        )
        detect_root = f"/perception/{key}"
        table_plane_path = self._camera("ur5e")["calibration_path"]
        canonical = "true" if key == "ur5e" else "false"
        background_rate = "0.2" if key == "ur5e" else "0.0"
        command = (
            f"{self.venv_python} -m "
            "cais_spade_llm.resources.sensor.physical.realsense_roboflow_node "
            "--ros-args "
            f"-r __node:={key}_realsense_roboflow_perception "
            f"--params-file {config_path} "
            f"-p camera_role:={key} -p color_topic:={topics['color']} "
            f"-p aligned_depth_topic:={topics['depth']} "
            f"-p camera_info_topic:={topics['camera_info']} "
            f"-p tool_frame:={camera['parent_frame']} "
            f"-p camera_link_frame:={camera['camera_name']}_link "
            f"-p camera_optical_frame:={camera['camera_name']}_color_optical_frame "
            f"-p hand_eye_config:={calibration_path} "
            f"-p table_plane_config:={table_plane_path} "
            f"-p snapshot_path:={self._snapshot_path(key)} "
            f"-p preview_root:={PREVIEW_ROOT} "
            f"-p detect_all_service:={detect_root}/detect_all "
            f"-p detect_part_service:={detect_root}/detect_part "
            f"-p publish_canonical_services:={canonical} "
            f"-p background_rate_hz:={background_rate}"
        )
        return self.bridge._start_tracked_ros2_command(
            process_name,
            self._logged_command(process_name, command),
            ros_domain_id=self._domain_id(),
        )

    def start_perception(self, role: str) -> str | None:
        """Start an isolated Roboflow instance and restore it after camera recovery."""
        key = str(role).strip().lower()
        with self._connection_lock:
            error = self._start_perception_stack(key)
            if error is None or "already running" in error:
                self._desired_perception.add(key)
                self._reconcile_part_twin_sync(key)
                return None
            return error

    def stop_perception(self, role: str) -> None:
        """Stop one role-specific perception instance."""
        key = str(role).strip().lower()
        with self._connection_lock:
            self._desired_perception.discard(key)
            self.bridge.ros2_stop(self._process_names(key)["perception"])

    def _clear_detection_preview(self, role: str) -> None:
        preview_dir = PREVIEW_ROOT / role
        for filename in ("detection.jpg", "detection_status.json"):
            with suppress(OSError):
                (preview_dir / filename).unlink(missing_ok=True)

    def _wait_for_camera_frame(self, role: str) -> str | None:
        deadline = time.monotonic() + CAMERA_FIRST_FRAME_GRACE_SEC
        status_path = PREVIEW_ROOT / role / "status.json"
        names = self._process_names(role)
        while time.monotonic() < deadline:
            status = self._read_json(status_path)
            captured_at = float(status.get("frame_captured_at", 0.0) or 0.0)
            if captured_at and time.time() - captured_at <= CAMERA_FRAME_STALE_RECOVERY_SEC:
                return None
            if self.bridge.ros2_proc_status(names["camera"]) != "running":
                error = self._last_process_error(self._log_path(names["camera"]))
                if error:
                    return error
            time.sleep(0.2)
        return (
            f"{role} camera did not produce a synchronized frame within "
            f"{CAMERA_FIRST_FRAME_GRACE_SEC:.0f} seconds; automatic recovery remains active"
        )

    def start_detection(self, role: str) -> str | None:
        """Start the complete camera stack and request one visual detection."""
        key = str(role).strip().lower()
        self._clear_detection_preview(key)
        camera_error = self.start_camera(key)
        with self._connection_lock:
            if key in self._desired_connected:
                self._desired_perception.add(key)
        if camera_error:
            return camera_error
        frame_error = self._wait_for_camera_frame(key)
        if frame_error:
            return frame_error
        perception_error = self.start_perception(key)
        if perception_error:
            return perception_error
        result = self.test_detection(key)
        if not bool(result.get("visual_detection_ready", False)):
            return str(result.get("message") or "visual detection did not complete")
        return None

    def reset_camera(self, role: str) -> str | None:
        """Clean tracked and orphaned role processes, then restart detection."""
        key = str(role).strip().lower()
        camera = self._camera(key)
        self.stop_detection(key)
        errors = [
            error
            for process in ("perception", "preview", "camera")
            if (error := self._stop_stale_role_processes(key, camera, process))
        ]
        if errors:
            return "; ".join(errors)
        time.sleep(CAMERA_RESET_SETTLE_SEC)
        return self.start_detection(key)

    def stop_detection(self, role: str) -> None:
        """Stop detection, camera, preview, viewer, calibration, and recovery."""
        key = str(role).strip().lower()
        self.stop_camera(key)
        preview_dir = PREVIEW_ROOT / key
        for filename in (
            "status.json",
            "color.jpg",
            "depth.jpg",
            "detection.jpg",
            "detection_status.json",
        ):
            with suppress(OSError):
                (preview_dir / filename).unlink(missing_ok=True)

    def open_external_viewer(self, role: str) -> str | None:
        """Open `rqt_image_view` for the selected color topic."""
        key = str(role).strip().lower()
        topic = self._topics(self._camera(key))["color"]
        process_name = self._process_names(key)["viewer"]
        return self.bridge._start_tracked_ros2_command(
            process_name,
            self._logged_command(
                process_name,
                f"ros2 run rqt_image_view rqt_image_view {topic}",
            ),
            ros_domain_id=self._domain_id(),
        )

    def save_snapshot(self, role: str, stream: str = "color") -> Path:
        """Copy the latest preview frame to the operator diagnostics directory."""
        key = str(role).strip().lower()
        source = PREVIEW_ROOT / key / f"{stream}.jpg"
        if not source.is_file():
            raise RuntimeError(f"no {stream} preview is available for {key}")
        target_dir = self.diagnostic_root / "snapshots"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{key}_{stream}_{int(time.time())}.jpg"
        shutil.copy2(source, target)
        return target

    def record_diagnostics(self, *, duration_sec: float = 5.0) -> Path:
        """Record a short set of preview frames and camera status without inference."""
        duration = min(15.0, max(1.0, float(duration_sec)))
        target = self.diagnostic_root / f"recording_{int(time.time())}"
        target.mkdir(parents=True, exist_ok=False)
        started_at = time.time()
        index = 0
        while time.time() - started_at < duration:
            for role in CAMERA_ROLES:
                for stream in ("color", "depth", "detection"):
                    source = PREVIEW_ROOT / role / f"{stream}.jpg"
                    if source.is_file():
                        shutil.copy2(source, target / f"{role}_{stream}_{index:03d}.jpg")
                for status_name in ("status.json", "detection_status.json"):
                    status = PREVIEW_ROOT / role / status_name
                    if status.is_file():
                        shutil.copy2(
                            status,
                            target / f"{role}_{status_name.replace('.json', '')}_{index:03d}.json",
                        )
            index += 1
            time.sleep(0.5)
        _atomic_json_write(
            target / "recording.json",
            {
                "started_at": started_at,
                "completed_at": time.time(),
                "duration_sec": duration,
                "sample_count": index,
                "roboflow_invoked": False,
                "robot_motion_requested": False,
            },
        )
        return target

    def _calibration_capture_command(self, role: str) -> list[str]:
        camera = self._camera(role)
        topics = self._topics(camera)
        command = [
            str(self.venv_python),
            "-m",
            "cais_spade_llm.resources.sensor.physical.calibrate_hand_eye",
            "capture-once",
            str(Path(str(camera["samples_path"])).expanduser()),
            "--camera-role",
            role,
            "--world-frame",
            "world",
            "--tool-frame",
            str(camera["parent_frame"]),
            "--camera-link-frame",
            f"{camera['camera_name']}_link",
            "--camera-optical-frame",
            f"{camera['camera_name']}_color_optical_frame",
            "--color-topic",
            topics["color"],
            "--camera-info-topic",
            topics["camera_info"],
        ]
        if role == "stationary":
            command.append("--stationary-camera")
        return command

    def _run_ros_command(
        self,
        arguments: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        command = (
            self.bridge._ROS2_ENV
            + self.bridge._ros2_domain_export(self._domain_id())
            + " ".join(shlex.quote(argument) for argument in arguments)
        )
        return subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _read_joint_state(self, role: str) -> dict[str, Any]:
        camera = self._camera(role)
        if role == "ur5e":
            return self._read_ur5e_rtde_joint_state(camera)
        topic = str(camera.get("joint_state_topic") or "").strip()
        if not topic:
            raise RuntimeError(f"{role} does not use a reviewed robot pose set")
        result = self._run_ros_command(
            [
                "timeout",
                "8",
                "ros2",
                "topic",
                "echo",
                "--once",
                topic,
                "sensor_msgs/msg/JointState",
            ],
            timeout=10.0,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or f"{topic} is unavailable"
            )
        try:
            document = next(
                row for row in yaml.safe_load_all(result.stdout) if isinstance(row, dict)
            )
            names = [str(value) for value in document["name"]]
            positions = [float(value) for value in document["position"]]
        except (KeyError, StopIteration, TypeError, ValueError, yaml.YAMLError) as exc:
            raise RuntimeError(f"could not parse {topic}") from exc
        if not names or len(names) != len(positions):
            raise RuntimeError(f"{topic} returned an incomplete joint state")
        by_name = dict(zip(names, positions, strict=True))
        selected = next(
            (
                [str(name) for name in candidate]
                for candidate in camera.get("arm_joint_candidates", [])
                if all(str(name) in by_name for name in candidate)
            ),
            None,
        )
        if not selected:
            raise RuntimeError(f"{topic} does not contain the six {role} arm joints")
        return {
            "names": selected,
            "positions": [by_name[name] for name in selected],
            "captured_at": time.time(),
        }

    def _read_ur5e_rtde_joint_state(self, camera: dict[str, Any]) -> dict[str, Any]:
        """Read one UR5e calibration pose without requesting robot motion."""
        robot_ip = str(self.bridge.get_hardware_ips().get("ur5e") or "").strip()
        if not robot_ip:
            raise RuntimeError("UR5e robot_ip is unavailable for read-only RTDE capture")
        try:
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError("ur-rtde is unavailable for read-only UR5e capture") from exc

        receive = None
        try:
            receive = rtde_receive.RTDEReceiveInterface(robot_ip)
            positions = [float(value) for value in list(receive.getActualQ())]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"UR5e read-only RTDE capture failed at {robot_ip}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if receive is not None:
                with suppress(OSError, RuntimeError):
                    receive.disconnect()

        names = next(
            (
                [str(name) for name in candidate]
                for candidate in camera.get("arm_joint_candidates", [])
                if len(candidate) == len(positions)
            ),
            None,
        )
        if not names or len(positions) != 6:
            raise RuntimeError("UR5e read-only RTDE capture did not return six arm joints")
        return {
            "names": names,
            "positions": positions,
            "captured_at": time.time(),
        }

    @staticmethod
    def _read_reviewed_poses(camera: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
        pose_path = Path(str(camera["poses_path"])).expanduser()
        try:
            pose_payload = yaml.safe_load(pose_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            pose_payload = {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f"cannot read reviewed poses: {pose_path}") from exc
        poses = pose_payload.get("poses", []) if isinstance(pose_payload, dict) else []
        if not isinstance(poses, list) or not all(isinstance(row, dict) for row in poses):
            raise RuntimeError(f"reviewed poses are invalid: {pose_path}")
        return pose_path, poses

    @staticmethod
    def _require_new_reviewed_pose(
        role: str,
        joint_state: dict[str, Any],
        poses: list[dict[str, Any]],
    ) -> None:
        names = [str(name) for name in joint_state.get("names", [])]
        positions = [float(value) for value in joint_state.get("positions", [])]
        nearest: tuple[int, float] | None = None
        for pose_index, pose in enumerate(poses):
            existing_names = [str(name) for name in pose.get("names", [])]
            existing_positions = pose.get("positions", [])
            if len(existing_names) != len(existing_positions):
                continue
            by_name = dict(zip(existing_names, existing_positions, strict=True))
            if not names or not all(name in by_name for name in names):
                continue
            try:
                largest_delta = max(
                    abs(math.remainder(current - float(by_name[name]), math.tau))
                    for name, current in zip(names, positions, strict=True)
                )
            except (TypeError, ValueError):
                continue
            if nearest is None or largest_delta < nearest[1]:
                nearest = (pose_index, largest_delta)
        if nearest is None or nearest[1] >= MINIMUM_CALIBRATION_JOINT_DELTA_RAD:
            return
        pose_number = nearest[0] + 1
        largest_delta_deg = math.degrees(nearest[1])
        minimum_delta_deg = math.degrees(MINIMUM_CALIBRATION_JOINT_DELTA_RAD)
        raise RuntimeError(
            f"{role} pose is not a new calibration pose: it is too close to reviewed pose "
            f"{pose_number} (largest joint change {largest_delta_deg:.1f} deg; at least "
            f"{minimum_delta_deg:.1f} deg required). Move the robot to a new position or "
            "orientation, stop it, then retry Save Pose + Capture."
        )

    @staticmethod
    def _calibration_capture_error(result: subprocess.CompletedProcess[str]) -> str:
        output = "\n".join(
            text for text in (result.stderr.strip(), result.stdout.strip()) if text
        )
        if "fewer than four ChArUco markers are visible" in output:
            return (
                "ChArUco board is not visible enough: fewer than four markers were detected. "
                "Place the complete board in the color view, avoid glare or blur, then retry "
                "Save Pose + Capture."
            )
        if "fewer than six ChArUco corners are visible" in output:
            return (
                "ChArUco board is only partially visible: fewer than six corners were detected. "
                "Show more of the board in the color view, then retry Save Pose + Capture."
            )
        if "color image or CameraInfo is unavailable" in output:
            return "UR5e color image or CameraInfo is unavailable; wait for camera topics and retry."
        for line in reversed(output.splitlines()):
            if "RuntimeError:" in line:
                return line.split("RuntimeError:", maxsplit=1)[1].strip()
        return output or "ChArUco capture was rejected"

    def save_pose_and_capture(self, role: str) -> dict[str, Any]:
        """Save one reviewed robot pose and one stationary board observation."""
        key = str(role).strip().lower()
        camera = self._camera(key)
        if key == "ur5e":
            self._ensure_ur5e_calibration_monitor()
        joint_state = None if key == "stationary" else self._read_joint_state(key)
        pose_path = None
        poses: list[dict[str, Any]] = []
        if joint_state is not None:
            pose_path, poses = self._read_reviewed_poses(camera)
            self._require_new_reviewed_pose(key, joint_state, poses)
        try:
            result = self._run_ros_command(
                self._calibration_capture_command(key),
                timeout=18.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ChArUco capture timed out") from exc
        if result.returncode != 0:
            raise RuntimeError(self._calibration_capture_error(result))
        samples = self._read_json(Path(str(camera["samples_path"])).expanduser())
        sample_count = len(samples.get("samples", []))
        if joint_state is not None:
            if pose_path is None:
                raise RuntimeError("reviewed poses path is unavailable")
            poses.append({"sample_index": sample_count, **joint_state})
            _atomic_yaml_write(
                pose_path,
                {
                    "camera_role": key,
                    "planning_group": camera["planning_group"],
                    "poses": poses,
                },
            )
        return {
            "camera_role": key,
            "sample_count": sample_count,
            "pose_count": self._pose_count(camera),
            "message": (
                f"Accepted {key} ChArUco sample {sample_count}; no robot motion was requested."
            ),
        }

    @staticmethod
    def _pose_count(camera: dict[str, Any]) -> int:
        raw_path = str(camera.get("poses_path") or "").strip()
        if not raw_path:
            return 0
        try:
            payload = yaml.safe_load(Path(raw_path).expanduser().read_text(encoding="utf-8")) or {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return 0
        poses = payload.get("poses", []) if isinstance(payload, dict) else []
        return len(poses) if isinstance(poses, list) else 0

    def solve_calibration(self, role: str) -> Path:
        """Create an accepted candidate calibration without activating it."""
        key = str(role).strip().lower()
        camera = self._camera(key)
        active = Path(str(camera["calibration_path"])).expanduser()
        candidate = active.with_name(f"{active.stem}.candidate{active.suffix}")
        arguments = [
            str(self.venv_python),
            "-m",
            "cais_spade_llm.resources.sensor.physical.calibrate_hand_eye",
        ]
        if key == "stationary":
            board = self.config()["stationary_board_world_pose"]
            if not bool(board.get("configured", False)):
                raise RuntimeError("configure the surveyed stationary ChArUco board pose first")
            arguments.extend(
                [
                    "solve-stationary",
                    str(Path(str(camera["samples_path"])).expanduser()),
                    "--output",
                    str(candidate),
                    "--camera-role",
                    key,
                    "--parent-frame",
                    str(camera["parent_frame"]),
                ]
            )
            for field in ("x", "y", "z", "roll", "pitch", "yaw"):
                arguments.extend([f"--board-{field}", str(float(board[field]))])
        else:
            arguments.extend(
                [
                    "solve",
                    str(Path(str(camera["samples_path"])).expanduser()),
                    "--output",
                    str(candidate),
                    "--camera-role",
                    key,
                    "--parent-frame",
                    str(camera["parent_frame"]),
                ]
            )
        try:
            result = self._run_ros_command(arguments, timeout=40.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("calibration solve timed out") from exc
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "calibration was rejected"
            )
        return candidate

    def activate_calibration(self, role: str) -> Path:
        """Activate an accepted candidate and retain the previous calibration."""
        camera = self._camera(role)
        active = Path(str(camera["calibration_path"])).expanduser()
        candidate = active.with_name(f"{active.stem}.candidate{active.suffix}")
        previous = active.with_name(f"{active.stem}.previous{active.suffix}")
        try:
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise RuntimeError(f"candidate calibration is missing: {candidate}") from exc
        if not bool((payload.get("validation") or {}).get("accepted", False)):
            raise RuntimeError("candidate calibration is not marked accepted")
        if active.is_file():
            shutil.copy2(active, previous)
        os.replace(candidate, active)
        return active

    def rollback_calibration(self, role: str) -> Path:
        """Restore the previous accepted calibration without deleting the current file."""
        camera = self._camera(role)
        active = Path(str(camera["calibration_path"])).expanduser()
        previous = active.with_name(f"{active.stem}.previous{active.suffix}")
        if not previous.is_file():
            raise RuntimeError(f"previous calibration is missing: {previous}")
        current = active.with_name(f"{active.stem}.rollback{active.suffix}")
        if active.is_file():
            shutil.copy2(active, current)
        shutil.copy2(previous, active)
        return active

    def start_table_plane_calibration(self) -> str | None:
        """Start the authoritative 10-frame UR5e table-plane calibration."""
        camera = self._camera("ur5e")
        command = (
            f"{self.venv_python} -m "
            "cais_spade_llm.resources.sensor.physical.calibrate_hand_eye "
            f"table-plane --output {shlex.quote(str(camera['calibration_path']))} --frames 10 "
            "--service /perception/ur5e/table_plane_measurement"
        )
        process_name = self._process_names("ur5e")["calibration"]
        return self.bridge._start_tracked_ros2_command(
            process_name,
            self._logged_command(process_name, command),
            ros_domain_id=self._domain_id(),
        )

    def calibration_replay_control(self, role: str, action: str) -> Path:
        """Write a Pause, Resume, Skip, or Abort command for a calibration replay."""
        key = str(role).strip().lower()
        if key not in {"ur5e", "xarm6"}:
            raise RuntimeError("only ur5e and xarm6 use teach-then-replay calibration")
        normalized = str(action).strip().lower()
        if normalized not in {"pause", "resume", "skip", "abort"}:
            raise ValueError(f"unknown calibration replay action: {action}")
        path = Path(f"/tmp/cais_{key}_calibration_replay_control.json")
        _atomic_json_write(path, {"action": normalized, "requested_at": time.time()})
        return path

    def start_calibration_replay(self, role: str, *, confirmed: bool) -> str | None:
        """Start explicitly confirmed MoveIt replay of a reviewed wrist pose set."""
        key = str(role).strip().lower()
        if key not in {"ur5e", "xarm6"}:
            return "only ur5e and xarm6 use teach-then-replay calibration"
        if not confirmed:
            return "automatic calibration requires explicit operator confirmation"
        camera = self._camera(key)
        if self._pose_count(camera) < 20:
            return f"at least 20 reviewed {key} calibration poses are required"
        status_path = Path(f"/tmp/cais_{key}_calibration_replay_status.json")
        if self._read_json(status_path).get("state") != "preview_ready":
            return "preview all reviewed calibration poses successfully before confirmation"
        sample_path = Path(str(camera["samples_path"])).expanduser()
        if sample_path.is_file():
            teach_path = sample_path.with_name(f"{sample_path.stem}.teach{sample_path.suffix}")
            shutil.copy2(sample_path, teach_path)
        _atomic_json_write(
            sample_path,
            {
                "camera_role": key,
                "world_frame": "world",
                "tool_frame": camera["parent_frame"],
                "samples": [],
            },
        )
        control_path = Path(f"/tmp/cais_{key}_calibration_replay_control.json")
        capture_command = self._calibration_capture_command(key)
        arguments = [
            str(self.venv_python),
            "-m",
            "cais_spade_llm.resources.sensor.physical.calibration_pose_replay",
            "--camera-role",
            key,
            "--planning-group",
            str(camera["planning_group"]),
            "--poses",
            str(Path(str(camera["poses_path"])).expanduser()),
            "--control",
            str(control_path),
            "--status",
            str(status_path),
            "--confirmed",
            "--",
            *capture_command,
        ]
        command = " ".join(shlex.quote(argument) for argument in arguments)
        process_name = self._process_names(key)["calibration"]
        return self.bridge._start_tracked_ros2_command(
            process_name,
            self._logged_command(process_name, command),
            ros_domain_id=self._domain_id(),
        )

    def preview_calibration_replay(self, role: str) -> str | None:
        """Plan every reviewed wrist pose without executing or capturing."""
        key = str(role).strip().lower()
        if key not in {"ur5e", "xarm6"}:
            return "only ur5e and xarm6 use teach-then-replay calibration"
        camera = self._camera(key)
        if self._pose_count(camera) < 20:
            return f"at least 20 reviewed {key} calibration poses are required"
        arguments = [
            str(self.venv_python),
            "-m",
            "cais_spade_llm.resources.sensor.physical.calibration_pose_replay",
            "--camera-role",
            key,
            "--planning-group",
            str(camera["planning_group"]),
            "--poses",
            str(Path(str(camera["poses_path"])).expanduser()),
            "--control",
            f"/tmp/cais_{key}_calibration_replay_control.json",
            "--status",
            f"/tmp/cais_{key}_calibration_replay_status.json",
            "--preview-only",
        ]
        process_name = self._process_names(key)["calibration_preview"]
        command = " ".join(shlex.quote(argument) for argument in arguments)
        return self.bridge._start_tracked_ros2_command(
            process_name,
            self._logged_command(process_name, command),
            ros_domain_id=self._domain_id(),
        )

    def test_detection(self, role: str) -> dict[str, Any]:
        """Request one role-specific detection without initiating robot motion."""
        key = str(role).strip().lower()
        service = f"/perception/{key}/detect_all"
        requested_at = time.time()
        command = (
            self.bridge._ROS2_ENV
            + self.bridge._ros2_domain_export(self._domain_id())
            + f"timeout 35 ros2 service call {service} std_srvs/srv/Trigger '{{}}'"
        )
        try:
            result = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True, timeout=40.0, check=False
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "visual_detection_ready": False,
                "world_pose_ready": False,
                "message": "Test Detection timed out",
                "detections": [],
                "visual_detections": [],
            }
        snapshot = self._read_json(self._snapshot_path(key))
        visual = self._read_json(PREVIEW_ROOT / key / "detection_status.json")
        error = str(snapshot.get("last_error") or "").strip()
        detections = snapshot.get("detections")
        visual_detections = visual.get("detections")
        visual_ready = bool(
            float(visual.get("updated_at", 0.0) or 0.0) >= requested_at - 0.25
            and isinstance(visual_detections, list)
        )
        world_pose_ready = bool(visual.get("world_pose_ready", False))
        pose_error = str(visual.get("pose_error") or error).strip()
        world_rows = detections if isinstance(detections, list) else []
        if visual_ready and world_pose_ready:
            message = "Detection and world pose validated; no robot motion was requested."
        elif visual_ready:
            message = "2D detection completed; world pose unavailable"
            if pose_error:
                message += f": {pose_error}"
        else:
            message = (
                error
                or result.stderr.strip()
                or result.stdout.strip()
                or "Detection failed"
            )
        return {
            "success": visual_ready,
            "visual_detection_ready": visual_ready,
            "world_pose_ready": world_pose_ready,
            "pose_error": pose_error,
            "message": message,
            "detections": world_rows,
            "visual_detections": (
                visual_detections if isinstance(visual_detections, list) else []
            ),
        }

    def preflight(self) -> dict[str, Any]:
        """Return non-mutating host, WSL, permission, ROS, and secret checks."""
        video_nodes = sorted(Path("/dev").glob("video*"))
        inaccessible = [str(path) for path in video_nodes if not os.access(path, os.R_OK | os.W_OK)]
        groups = {grp.getgrgid(gid).gr_name for gid in os.getgroups()}
        groups.add(grp.getgrgid(os.getgid()).gr_name)
        ros_distro = str(os.environ.get("ROS_DISTRO") or "humble")
        ros_share = Path("/opt/ros") / ros_distro / "share"
        udev_rules = list(Path("/lib/udev/rules.d").glob("*realsense*")) + list(
            Path("/etc/udev/rules.d").glob("*realsense*")
        )
        checks = {
            "video_group": "video" in groups,
            "video_nodes": bool(video_nodes),
            "video_permissions": bool(video_nodes) and not inaccessible,
            "realsense_cli": shutil.which("rs-enumerate-devices") is not None,
            "realsense2_camera_package": (ros_share / "realsense2_camera").is_dir(),
            "realsense2_description_package": (ros_share / "realsense2_description").is_dir(),
            "realsense_udev_rules": bool(udev_rules),
            "ros2": shutil.which("ros2") is not None,
            "roboflow_api_key": bool(str(os.environ.get("ROBOFLOW_API_KEY", "")).strip()),
            "wsl": "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower(),
        }
        return {
            "ready": all(
                checks[name]
                for name in (
                    "video_group",
                    "video_nodes",
                    "video_permissions",
                    "realsense_cli",
                    "realsense2_camera_package",
                    "realsense2_description_package",
                    "ros2",
                )
            ),
            "checks": checks,
            "inaccessible_video_nodes": inaccessible,
            "setup_message": (
                "One-time setup is required; the UI never stores or requests a sudo password."
                if not all(
                    checks[name]
                    for name in (
                        "video_group",
                        "video_nodes",
                        "video_permissions",
                        "realsense_cli",
                        "realsense2_camera_package",
                        "realsense2_description_package",
                        "ros2",
                    )
                )
                else (
                    "Host camera prerequisites are ready."
                    if checks["roboflow_api_key"]
                    else "Camera prerequisites are ready; ROBOFLOW_API_KEY is still required for detection."
                )
            ),
        }

    @staticmethod
    def _calibration_file_status(camera: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(camera["calibration_path"])).expanduser()
        candidate = path.with_name(f"{path.stem}.candidate{path.suffix}")
        previous = path.with_name(f"{path.stem}.previous{path.suffix}")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            payload = {}
        validation = payload.get("validation", {}) if isinstance(payload, dict) else {}
        table_plane = payload.get("table_plane", {}) if isinstance(payload, dict) else {}
        return {
            "ready": bool(validation.get("accepted", False)),
            "calibration_id": str(payload.get("calibration_id") or ""),
            "method": str(payload.get("method") or ""),
            "validation": validation if isinstance(validation, dict) else {},
            "table_plane": table_plane if isinstance(table_plane, dict) else {},
            "candidate_ready": candidate.is_file(),
            "previous_ready": previous.is_file(),
        }

    @staticmethod
    def _cross_camera_comparisons(cameras: dict[str, Any]) -> list[dict[str, Any]]:
        authoritative = (
            (cameras.get("ur5e") or {}).get("perception", {}).get("detections", [])
        )
        stationary = (
            (cameras.get("stationary") or {}).get("perception", {}).get("detections", [])
        )
        if not isinstance(authoritative, list) or not isinstance(stationary, list):
            return []
        stationary_by_part = {
            str(row.get("part_name")): row
            for row in stationary
            if isinstance(row, dict) and row.get("frame_id") == "world"
        }
        rows: list[dict[str, Any]] = []
        for ur5e_row in authoritative:
            if not isinstance(ur5e_row, dict) or ur5e_row.get("frame_id") != "world":
                continue
            part_name = str(ur5e_row.get("part_name") or "")
            stationary_row = stationary_by_part.get(part_name)
            if stationary_row is None:
                continue
            try:
                disagreement_m = math.sqrt(
                    sum(
                        (float(ur5e_row[axis]) - float(stationary_row[axis])) ** 2
                        for axis in ("x", "y", "z")
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "part_name": part_name,
                    "disagreement_m": disagreement_m,
                    "warning": disagreement_m > 0.010,
                }
            )
        return rows

    def status(self) -> dict[str, Any]:
        """Return all page state without starting inference or robot motion."""
        config = self.config()
        devices = self.discover_devices()
        device_by_serial = {str(row.get("serial")): row for row in devices}
        wsl_rows = self.discover_wsl_attachments()
        wsl_states = {str(row.get("state") or "Unknown") for row in wsl_rows}
        assigned_role_names = [
            role
            for role in CAMERA_ROLES
            if str((config["cameras"].get(role) or {}).get("serial") or "").strip()
        ]
        camera_inventory = {
            "windows_d435_devices": len(wsl_rows),
            "wsl_attached_devices": sum(
                1 for row in wsl_rows if str(row.get("state") or "") == "Attached"
            ),
            "wsl_discovered_devices": len(devices),
            "assigned_roles": len(assigned_role_names),
            "assigned_role_names": assigned_role_names,
            "total_roles": len(CAMERA_ROLES),
        }
        with self._connection_lock:
            recovery_by_role = deepcopy(self._recovery)
            desired_connected = set(self._desired_connected)
        cameras: dict[str, Any] = {}
        now = time.time()
        for role in CAMERA_ROLES:
            camera = dict(config["cameras"][role])
            serial = str(camera.get("serial") or "")
            preview = self._read_json(PREVIEW_ROOT / role / "status.json")
            perception = self._read_json(self._snapshot_path(role))
            detection_preview = self._read_json(
                PREVIEW_ROOT / role / "detection_status.json"
            )
            frame_at = float(preview.get("frame_captured_at", 0.0) or 0.0)
            detection_at = float(detection_preview.get("captured_at", 0.0) or 0.0)
            calibration_path = Path(str(camera["calibration_path"])).expanduser()
            samples = self._read_json(Path(str(camera["samples_path"])).expanduser())
            names = self._process_names(role)
            calibration = self._calibration_file_status(camera)
            camera_process = self.bridge.ros2_proc_status(names["camera"])
            perception_process = self.bridge.ros2_proc_status(names["perception"])
            if role == "ur5e" and self._ur5e_digital_twin_process_running("camera"):
                camera_process = "running"
            if role == "ur5e" and self._ur5e_digital_twin_process_running("perception"):
                perception_process = "running"
            preview_process = self.bridge.ros2_proc_status(names["preview"])
            if camera_process != "running" and preview_process == "running":
                self.bridge.ros2_stop(names["preview"])
                preview_process = "stopped"
            frame_ready = bool(frame_at and now - frame_at <= 2.5)
            camera_last_error = (
                self._last_process_error(self._log_path(names["camera"]))
                if camera_process != "running" or not frame_ready
                else ""
            )
            if serial in device_by_serial:
                attachment_state = "Attached"
            elif "Shared" in wsl_states:
                attachment_state = "Shared"
            elif "Not shared" in wsl_states:
                attachment_state = "Not shared"
            elif "Attached" in wsl_states:
                attachment_state = "Attached serial not discovered"
            else:
                attachment_state = "Unavailable"
            cameras[role] = {
                **camera,
                "device": device_by_serial.get(serial, {}),
                "assigned": bool(serial),
                "connected": serial in device_by_serial,
                "desired_connected": role in desired_connected,
                "attachment_state": attachment_state,
                "recovery": recovery_by_role[role],
                "camera_process": camera_process,
                "preview_process": preview_process,
                "perception_process": perception_process,
                "viewer_process": self.bridge.ros2_proc_status(names["viewer"]),
                "frame_age_sec": max(0.0, now - frame_at) if frame_at else None,
                "ros_topic_ready": frame_ready,
                "visual_detection_ready": bool(
                    detection_preview.get("visual_detection_ready", False)
                ),
                "world_pose_ready": bool(
                    detection_preview.get("world_pose_ready", False)
                ),
                "tf_ready": bool(detection_preview.get("world_pose_ready", False)),
                "preview": preview,
                "perception": perception,
                "detection_preview": {
                    **detection_preview,
                    "frame_age_sec": (
                        max(0.0, now - detection_at) if detection_at else None
                    ),
                },
                "camera_last_error": camera_last_error,
                "calibration_path": str(calibration_path),
                "calibration_ready": calibration["ready"],
                "calibration": calibration,
                "sample_count": len(samples.get("samples", [])),
                "pose_count": self._pose_count(camera),
                "replay": self._read_json(
                    Path(f"/tmp/cais_{role}_calibration_replay_status.json")
                ),
                "logs": {
                    process: {
                        "path": str(self._log_path(names[process])),
                        "tail": self._log_tail(self._log_path(names[process])),
                    }
                    for process in (
                        "camera",
                        "preview",
                        "perception",
                        "calibration",
                        "calibration_preview",
                    )
                },
                "topics": self._topics(camera),
            }
        preflight = self.preflight()
        preflight["camera_inventory"] = camera_inventory
        preflight["device_discovery_error"] = self._device_discovery_error
        return {
            "config_path": str(self.config_path),
            "device_discovery_error": self._device_discovery_error,
            "preflight": preflight,
            "camera_inventory": camera_inventory,
            "devices": devices,
            "wsl_devices": wsl_rows,
            "cameras": cameras,
            "stationary_board_world_pose": config["stationary_board_world_pose"],
            "digital_twin": self._read_json(Path("/tmp/cais_physical_part_twin_status.json")),
            "cross_camera_comparisons": self._cross_camera_comparisons(cameras),
            "lg_status": "LG unavailable: model has no large_gear class",
        }
