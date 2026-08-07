"""Persistent ownership records for subprocesses launched by the operator UI."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_UI_PROCESS_REGISTRY = Path("/tmp/cais_spade_ui_processes.json")


class UIProcessRegistry:
    """Record UI-owned process groups so a later UI can clean up after a crash."""

    def __init__(
        self,
        path: Path = DEFAULT_UI_PROCESS_REGISTRY,
        *,
        proc_root: Path = Path("/proc"),
    ) -> None:
        self.path = Path(path)
        self.proc_root = Path(proc_root)
        self.owner_pid = os.getpid()
        self._lock = threading.Lock()

    def register(self, name: str, process_group: int, command: str) -> None:
        """Persist ownership of one newly created process group."""
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("process name is empty")
        with self._lock:
            payload = self._read()
            processes = payload.setdefault("processes", {})
            processes[normalized_name] = {
                "name": normalized_name,
                "owner_pid": self.owner_pid,
                "process_group": int(process_group),
                "command": str(command or ""),
                "started_at": time.time(),
            }
            self._write(payload)

    def unregister(self, name: str, *, process_group: int | None = None) -> None:
        """Remove one ownership record after its process group has stopped."""
        normalized_name = str(name or "").strip()
        if not normalized_name:
            return
        with self._lock:
            payload = self._read()
            processes = payload.setdefault("processes", {})
            entry = processes.get(normalized_name)
            if not isinstance(entry, dict):
                return
            if process_group is not None:
                try:
                    recorded_group = int(entry.get("process_group", 0))
                except (TypeError, ValueError):
                    return
                if recorded_group != int(process_group):
                    return
            processes.pop(normalized_name, None)
            self._write(payload)

    def cleanup_exited_process(
        self,
        name: str,
        *,
        process_group: int,
    ) -> str | None:
        """Stop children left in a tracked group after its launcher exits."""
        normalized_name = str(name or "").strip()
        if not normalized_name or int(process_group) <= 0:
            return None
        with self._lock:
            payload = self._read()
            processes = payload.setdefault("processes", {})
            entry = processes.get(normalized_name)
            if not isinstance(entry, dict):
                return None
            recorded_owner = self._safe_int(entry.get("owner_pid"))
            recorded_group = self._safe_int(entry.get("process_group"))
            if recorded_owner != self.owner_pid or recorded_group != int(process_group):
                return None
            error = None
            if self._process_group_alive(recorded_group):
                if recorded_group == os.getpgrp() or not self._verified_cais_group(recorded_group):
                    return f"refused to stop unverified process group {recorded_group}"
                error = self._terminate_process_group(recorded_group)
            if error is None:
                processes.pop(normalized_name, None)
                self._write(payload)
            return error

    def release_current_owner(self) -> None:
        """Forget current UI ownership without stopping preserved debug processes."""
        with self._lock:
            payload = self._read()
            processes = payload.setdefault("processes", {})
            payload["processes"] = {
                name: entry
                for name, entry in processes.items()
                if not isinstance(entry, dict)
                or self._safe_int(entry.get("owner_pid")) != self.owner_pid
            }
            self._write(payload)

    def cleanup_previous(self) -> dict[str, list[str]]:
        """Stop verified groups whose recorded UI owner no longer exists."""
        result: dict[str, list[str]] = {
            "stopped": [],
            "active_owner": [],
            "discarded": [],
            "errors": [],
        }
        with self._lock:
            payload = self._read()
            processes = payload.setdefault("processes", {})
            remaining: dict[str, Any] = {}
            for raw_name, raw_entry in list(processes.items()):
                name = str(raw_name)
                if not isinstance(raw_entry, dict):
                    result["discarded"].append(name)
                    continue
                owner_pid = self._safe_int(raw_entry.get("owner_pid"))
                process_group = self._safe_int(raw_entry.get("process_group"))
                if owner_pid == self.owner_pid:
                    remaining[name] = raw_entry
                    continue
                if owner_pid > 0 and self._pid_alive(owner_pid):
                    remaining[name] = raw_entry
                    result["active_owner"].append(name)
                    continue
                if process_group <= 0 or not self._process_group_alive(process_group):
                    result["discarded"].append(name)
                    continue
                if process_group == os.getpgrp() or not self._verified_cais_group(process_group):
                    result["discarded"].append(name)
                    continue
                error = self._terminate_process_group(process_group)
                if error:
                    remaining[name] = raw_entry
                    result["errors"].append(f"{name}: {error}")
                else:
                    result["stopped"].append(name)
            payload["processes"] = remaining
            self._write(payload)
        return result

    def _verified_cais_group(self, process_group: int) -> bool:
        command_lines = self._process_group_command_lines(process_group)
        project_root = str(Path(__file__).resolve().parents[2])
        markers = (
            f"{project_root}/",
            "cais_spade_llm.",
            "cais_lab_robotics",
        )
        return any(any(marker in command for marker in markers) for command in command_lines)

    def _process_group_command_lines(self, process_group: int) -> list[str]:
        commands: list[str] = []
        for entry in self.proc_root.glob("[0-9]*"):
            try:
                pid = int(entry.name)
                if os.getpgid(pid) != process_group:
                    continue
                command = (
                    (entry / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode("utf-8", errors="replace")
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
                continue
            if command:
                commands.append(command)
        return commands

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _process_group_alive(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _terminate_process_group(self, process_group: int) -> str | None:
        try:
            os.killpg(process_group, signal.SIGINT)
        except ProcessLookupError:
            return None
        except PermissionError as exc:
            return str(exc)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not self._process_group_alive(process_group):
                return None
            time.sleep(0.05)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return None
        except PermissionError as exc:
            return str(exc)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if not self._process_group_alive(process_group):
                return None
            time.sleep(0.05)
        return f"process group {process_group} did not stop"

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"version": 1, "processes": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("processes"), dict):
            return {"version": 1, "processes": {}}
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{self.owner_pid}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                log.debug("Could not remove UI process registry temporary file", exc_info=True)

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
