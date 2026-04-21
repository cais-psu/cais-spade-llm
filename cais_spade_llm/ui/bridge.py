"""SystemBridge: in-process bridge between SPADE agents and the NiceGUI operator console."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import os
import re
import select
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from cais_spade_llm.bundles import BundleCompiler, BundleStore
from cais_spade_llm.bundles.models import (
    BUNDLE_STATUS_DRAFT,
    BUNDLE_STATUS_INVALID,
    BUNDLE_STATUS_STALE,
    BUNDLE_STATUS_VERIFIED,
    atomic_json_write,
    slug,
    sha256_file,
    sha256_text,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    DEFAULT_BRIDGE_RUNTIME_DATA_DIR,
)

log = logging.getLogger("ui.bridge")

# Filesystem locations (mirror spade_main.py constants).
_BASE = Path(__file__).resolve().parent.parent          # cais_spade_llm/
_PROJECT_ROOT = _BASE.parent                             # repo root
_PRODUCT_DIR = _BASE / "initialization" / "products"
_RESOURCE_DIR = _BASE / "initialization" / "resources"
_TOOLS_OUT = _BASE / "initialization" / "tools.json"
_CCA_INIT = _BASE / "initialization" / "cca.json"
_MONITOR = _BASE / "monitor"
_BRIDGE_RUNTIME_DATA_DIR = Path(DEFAULT_BRIDGE_RUNTIME_DATA_DIR)
_LOG_DIR = _BASE / "log"
_PRODUCT_REQUIREMENTS_DIR = _BASE / "specification" / "products" / "requirements"
_SAFETY_REQUIREMENTS_DIR = _BASE / "specification" / "safety"
_XARM6_RESOURCE = _RESOURCE_DIR / "robot_xarm6.json"
_UR5E_RESOURCE = _RESOURCE_DIR / "robot_ur5e.json"
_UR5E_GAZEBO_ARM_TRAJECTORY_TOPIC = "/ur5e_joint_trajectory_controller/joint_trajectory"
_USER_VERIFIED_PLAN = _BASE / "user_verified_plan"
_USER_VERIFIED_SAFETY = _BASE / "user_verified_safety"
_SAFETY_INTENT_APPROVALS = _USER_VERIFIED_SAFETY / "intent_approvals.json"
_SAFETY_INTENT_PREVIEWS = _USER_VERIFIED_SAFETY / "intent_previews.json"
_SAFETY_PREVIEW_DIR = _USER_VERIFIED_SAFETY / "previews"
_SAFETY_VERIFIED_DIR = _USER_VERIFIED_SAFETY / "verified"
_SAFETY_PREVIEW_HISTORY_LIMIT = 10
_GAZEBO_WORLD_FILE = _PROJECT_ROOT / "ros2" / "cais_lab_gazebo" / "worlds" / "table.world"
_RESETTABLE_GAZEBO_MODEL_PREFIXES = ("gear_", "rect_pin_", "circ_pin_")


def _legacy_safety_intent_approvals_path() -> Path:
    return _SAFETY_REQUIREMENTS_DIR / "intent_approvals.json"


def _legacy_safety_intent_previews_path() -> Path:
    return _SAFETY_REQUIREMENTS_DIR / "intent_previews.json"


def _legacy_safety_preview_dir() -> Path:
    return _USER_VERIFIED_PLAN / "safety_previews"


class SystemBridge:
    """Singleton that owns the SPADE lifecycle and exposes agent state to the UI."""

    _instance: Optional[SystemBridge] = None
    _HW_IP_DEFAULTS = {
        "xarm6": "192.168.1.240",
        "ur5e": "192.168.1.172",
    }
    _GAZEBO_PROCESS_NAMES = {"gazebo_dual", "gazebo_xarm6", "gazebo_ur5e"}
    _HARDWARE_PROCESS_NAMES = {
        "hardware_xarm6_driver",
        "hardware_xarm6_moveit",
        "hardware_ur5e_driver",
        "hardware_ur5e_moveit",
    }
    _HARDWARE_STACKS = {
        # xArm6 MoveIt realmove includes UFRobotSystemHardware (embedded driver path).
        "xarm6": ("hardware_xarm6_moveit",),
        # UR5e still uses explicit driver + MoveIt bring-up.
        "ur5e": ("hardware_ur5e_driver", "hardware_ur5e_moveit"),
    }
    _GAZEBO_PREWARM_TIMEOUT_S = 60.0
    _GAZEBO_PREWARM_START_DELAY_S = 0.5
    _GAZEBO_PREWARM_READY_WAIT_S = 60.0
    _SIM_CORE_SERVICES = (
        "/compute_cartesian_path",
        "/ATTACHLINK",
        "/DETACHLINK",
    )
    _SIM_PERCEPTION_SERVICES = ("/detect_all",)
    _SIM_PREWARM_SHELL_SERVICES = ("/compute_cartesian_path",)
    _ENABLE_GAZEBO_TIMING_LOGS = str(
        os.getenv("CAIS_SPADE_GAZEBO_TIMING", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    _GAZEBO_WORKSPACE_LAUNCH_FILES = {
        "gazebo_dual": "dual_moveit_gazebo.launch.py",
        "gazebo_xarm6": "xarm6_moveit_single_gazebo.launch.py",
        "gazebo_ur5e": "ur5e_rg2_moveit_gazebo.launch.py",
    }

    @classmethod
    def instance(cls) -> SystemBridge:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        # Agent references (populated after system start).
        self.user_agent = None
        self.resource_agents: list = []
        self.product_agents: list = []
        self.cca = None

        # Embedded XMPP server process.
        self._xmpp_proc: Optional[subprocess.Popen] = None
        self._xmpp_host: str = "127.0.0.1"
        self._xmpp_port: int = 5222
        self._gazebo_reset_pose_cache: dict[str, tuple[float, float, float, float, float, float]] | None = None

        # Lifecycle flags.
        self.system_running: bool = False
        self._starting: bool = False
        self._stopping: bool = False
        self.last_error: Optional[str] = None
        self.last_notice: Optional[str] = None
        self._safety_preview_failures: dict[str, dict[str, Any]] = {}

        # Configuration (set from UI before start).
        self.execution_mode: str = "simulation"
        self.robot_env: str = "gazebo"
        self.selected_product: str = ""
        self.selected_requirement_file: str = ""
        # Empty string -> use manifest default safety, "__NONE__" -> disable safety,
        # any other value -> explicit safety text file path.
        self.selected_safety_file: str = ""
        self.runtime_bridge_mode: str = "pre_ran"
        self.runtime_bridge_validation_policy: str = "validated"
        self.runtime_bridge_archive_path: str = ""
        self.runtime_bridge_archive_label: str = ""
        self._runtime_bridge_archive_cache_signature: tuple[Any, ...] | None = None
        self._runtime_bridge_archive_cache_entries: list[dict[str, Any]] | None = None
        preferred_bridge_archive = self._preferred_runtime_bridge_archive_entry()
        if isinstance(preferred_bridge_archive, dict):
            self.runtime_bridge_archive_path = str(
                preferred_bridge_archive.get("path") or ""
            ).strip()
            self.runtime_bridge_archive_label = str(
                preferred_bridge_archive.get("label") or ""
            ).strip()
        self.bundle_store = BundleStore(_USER_VERIFIED_PLAN)
        self.bundle_compiler = BundleCompiler(
            store=self.bundle_store,
            project_root=_PROJECT_ROOT,
            product_init_dir=_PRODUCT_DIR,
            resource_init_dir=_RESOURCE_DIR,
            cca_init_path=_CCA_INIT,
            tools_path=_TOOLS_OUT,
            prompts_path=_BASE / "prompts.py",
        )

        # ROS2 subprocess tracking.
        self._ros2_procs: dict[str, subprocess.Popen] = {}
        self._teleop_server_proc: Optional[subprocess.Popen] = None
        self._teleop_server_lock = threading.Lock()
        self._gazebo_prewarm_lock = threading.Lock()
        self._gazebo_prewarm_thread: Optional[threading.Thread] = None
        self._gazebo_prewarm_pending: set[str] = set()
        self._gazebo_prewarm_controllers: dict[str, Any] = {}
        self._gazebo_prewarm_cancel = threading.Event()
        self._gazebo_prewarm_done = threading.Event()  # Set when prewarm completes successfully.
        self._sim_ready_cache_ts: float = 0.0
        self._sim_ready_cache: tuple[bool, str] = (False, "Simulation startup check pending.")
        self._sim_ready_probe_inflight: bool = False
        self._gazebo_launch_seq: int = 0
        self._gazebo_launch_timing_lock = threading.Lock()
        self._gazebo_launch_timing: dict[str, Any] | None = None
        self.hardware_ips = self._load_hardware_ips()
        self._hw_ping_cache: dict[str, dict[str, Any]] = {}
        self._hw_ping_last_ts: float = 0.0
        self._startup_seq: int = 0
        self._startup_phase: str = "idle"
        self._startup_phase_ts: float = time.monotonic()
        self._cached_plan_safety_alerts: list[dict[str, Any]] = []
        self._agent_creator_cached: Any | None = None
        self._agent_creator_prefetch_started: bool = False
        self._agent_creator_prefetch_lock = threading.Lock()
        self._agent_creator_prefetch_thread: Optional[threading.Thread] = None
        self._ui_diag_enabled: bool = str(os.getenv("CAIS_UI_DIAG", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._maybe_start_agent_creator_prefetch()

    @staticmethod
    def _normalize_runtime_bridge_mode(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"manual", "auto", "pre_ran"}:
            return token
        if token in {"preran", "pre_ran_mode"}:
            return "pre_ran"
        return "pre_ran"

    @staticmethod
    def _normalize_runtime_bridge_validation_policy(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"validated", "no_validation"}:
            return token
        if token in {"no_validation_mode", "skip_validation", "unvalidated"}:
            return "no_validation"
        return "validated"

    def _apply_runtime_bridge_session_settings(self) -> None:
        for agent in list(self.product_agents or []):
            setter = getattr(agent, "set_runtime_bridge_session_settings", None)
            if not callable(setter):
                continue
            try:
                setter(
                    mode=self.runtime_bridge_mode,
                    validation_policy=self.runtime_bridge_validation_policy,
                    selected_archive_path=self.runtime_bridge_archive_path,
                    selected_archive_label=self.runtime_bridge_archive_label,
                )
            except Exception:
                log.exception(
                    "[ui.bridge] Failed to apply runtime bridge session settings to %s",
                    getattr(agent, "jid", "<unknown>"),
                )

    def get_runtime_bridge_settings(self) -> dict[str, Any]:
        if (
            self._normalize_runtime_bridge_mode(self.runtime_bridge_mode) == "pre_ran"
            and not str(self.runtime_bridge_archive_path or "").strip()
        ):
            preferred_entry = self._preferred_runtime_bridge_archive_entry()
            if isinstance(preferred_entry, dict):
                self.runtime_bridge_archive_path = str(
                    preferred_entry.get("path") or ""
                ).strip()
                self.runtime_bridge_archive_label = str(
                    preferred_entry.get("label") or ""
                ).strip()
        fixture_replay_path = str(
            os.environ.get("CAIS_RUNTIME_BRIDGE_FIXTURE_FINAL_OUTPUT") or ""
        ).strip()
        if fixture_replay_path:
            fixture_path = Path(fixture_replay_path).expanduser()
            try:
                fixture_path = fixture_path.resolve()
            except Exception:
                pass
            fixture_replay_path = str(fixture_path)
        return {
            "mode": self._normalize_runtime_bridge_mode(self.runtime_bridge_mode),
            "validation_policy": self._normalize_runtime_bridge_validation_policy(
                self.runtime_bridge_validation_policy
            ),
            "selected_archive_path": str(self.runtime_bridge_archive_path or "").strip(),
            "selected_archive_label": str(self.runtime_bridge_archive_label or "").strip(),
            "fixture_replay_path": fixture_replay_path,
            "fixture_replay_active": bool(fixture_replay_path),
        }

    def _preferred_runtime_bridge_archive_entry(self) -> dict[str, Any] | None:
        entries = self.list_runtime_bridge_archives()
        if not entries:
            return None

        for entry in entries:
            relative_path = str(entry.get("relative_path") or "").strip().replace("\\", "/")
            if relative_path.startswith("imported/worked/1/"):
                return dict(entry)
        return dict(entries[0])

    def set_runtime_bridge_mode(self, mode: str) -> dict[str, Any]:
        previous_mode = self._normalize_runtime_bridge_mode(self.runtime_bridge_mode)
        self.runtime_bridge_mode = self._normalize_runtime_bridge_mode(mode)
        if self.runtime_bridge_mode == "pre_ran" and previous_mode != "pre_ran":
            preferred_entry = self._preferred_runtime_bridge_archive_entry()
            if isinstance(preferred_entry, dict):
                self.runtime_bridge_archive_path = str(
                    preferred_entry.get("path") or ""
                ).strip()
                self.runtime_bridge_archive_label = str(
                    preferred_entry.get("label") or ""
                ).strip()
        self._apply_runtime_bridge_session_settings()
        return self.get_runtime_bridge_settings()

    def set_runtime_bridge_validation_policy(
        self,
        validation_policy: str,
    ) -> dict[str, Any]:
        self.runtime_bridge_validation_policy = (
            self._normalize_runtime_bridge_validation_policy(validation_policy)
        )
        self._apply_runtime_bridge_session_settings()
        return self.get_runtime_bridge_settings()

    def set_runtime_bridge_archive_selection(
        self,
        artifact_path: str | None,
        label: str | None = None,
    ) -> dict[str, Any]:
        path = str(artifact_path or "").strip()
        normalized_label = str(label or "").strip()
        if path:
            try:
                resolved = Path(path).expanduser().resolve()
            except Exception:
                resolved = Path(path).expanduser()
            self.runtime_bridge_archive_path = str(resolved)
        else:
            self.runtime_bridge_archive_path = ""
        self.runtime_bridge_archive_label = normalized_label
        self._apply_runtime_bridge_session_settings()
        return self.get_runtime_bridge_settings()

    def _invalidate_runtime_bridge_archive_cache(self) -> None:
        self._runtime_bridge_archive_cache_signature = None
        self._runtime_bridge_archive_cache_entries = None

    def _runtime_bridge_archive_scan_signature(self) -> tuple[Any, ...]:
        archive_root = _BRIDGE_RUNTIME_DATA_DIR
        if not archive_root.exists():
            return ("missing",)

        try:
            root_stat = archive_root.stat()
            children: list[tuple[str, bool, int]] = []
            for entry in archive_root.iterdir():
                try:
                    stat = entry.stat()
                except FileNotFoundError:
                    continue
                children.append((entry.name, entry.is_dir(), int(stat.st_mtime_ns)))
            children.sort()
            return (
                "ready",
                int(root_stat.st_mtime_ns),
                tuple(children),
            )
        except FileNotFoundError:
            return ("missing",)
        except Exception:
            log.exception(
                "Failed to build runtime bridge archive cache signature: %s",
                archive_root,
            )
            return ("error",)

    def list_runtime_bridge_archives(self) -> list[dict[str, Any]]:
        archive_root = _BRIDGE_RUNTIME_DATA_DIR
        if not archive_root.exists():
            self._invalidate_runtime_bridge_archive_cache()
            return []

        signature = self._runtime_bridge_archive_scan_signature()
        cached_entries = self._runtime_bridge_archive_cache_entries
        if (
            cached_entries is not None
            and self._runtime_bridge_archive_cache_signature == signature
        ):
            return [dict(entry) for entry in cached_entries]

        entries: list[dict[str, Any]] = []
        try:
            artifact_paths = list(
                archive_root.rglob("multi_turn_turn*_final_output_response_*.txt")
            )
        except FileNotFoundError:
            self._invalidate_runtime_bridge_archive_cache()
            return []
        except Exception:
            log.exception("Failed to scan runtime bridge archive directory: %s", archive_root)
            return []

        for artifact_path in artifact_paths:
            if not artifact_path.is_file():
                continue
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            engine = str(payload.get("engine") or "").strip().lower()
            if engine and not engine.startswith("multi_turn"):
                continue
            if str(payload.get("final_output_stage") or "").strip() != "primitive_program_ready":
                continue
            if not bool(payload.get("primitive_program_complete")):
                continue
            try:
                resolved = artifact_path.resolve()
            except Exception:
                resolved = artifact_path
            rel_path = str(resolved)
            try:
                rel_path = str(resolved.relative_to(archive_root))
            except Exception:
                pass
            stat = artifact_path.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            accepted_trace_length = int(
                payload.get("accepted_trace_length")
                or len(payload.get("accepted_primitive_program") or [])
                or len(payload.get("outline_tasks") or [])
                or 0
            )
            label = (
                f"{timestamp} | trace={accepted_trace_length} | {rel_path}"
            )
            entries.append(
                {
                    "path": str(resolved),
                    "label": label,
                    "relative_path": rel_path,
                    "accepted_trace_length": accepted_trace_length,
                    "final_output_stage": "primitive_program_ready",
                    "updated_at_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        entries.sort(
            key=lambda item: (
                str(item.get("updated_at_utc") or ""),
                str(item.get("path") or ""),
            ),
            reverse=True,
        )
        self._runtime_bridge_archive_cache_signature = signature
        self._runtime_bridge_archive_cache_entries = [dict(entry) for entry in entries]
        return entries

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _diag_emit(self, message: str) -> None:
        """Emit high-signal startup diagnostics to stdout and logger."""
        if not self._ui_diag_enabled:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        line = f"{ts} - ui.bridge - INFO - [UI-DIAG] {message}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        log.info("[UI-DIAG] %s", message)

    def _set_startup_phase(self, phase: str) -> None:
        self._startup_phase = str(phase)
        self._startup_phase_ts = time.monotonic()
        self._diag_emit(f"startup phase -> {self._startup_phase}")

    def _gazebo_timing_emit(self, message: str) -> None:
        """Emit Gazebo/MoveIt startup timing lines when explicitly enabled."""
        if not self._ENABLE_GAZEBO_TIMING_LOGS:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        line = f"{ts} - ui.bridge - INFO - [GAZEBO-TIMING] {message}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        log.info("[GAZEBO-TIMING] %s", message)

    def _begin_gazebo_launch_timing(self, launch_name: str, pid: int) -> int:
        launch_key = str(launch_name or "").strip().lower()
        with self._gazebo_launch_timing_lock:
            self._gazebo_launch_seq += 1
            launch_id = self._gazebo_launch_seq
            self._gazebo_launch_timing = {
                "id": launch_id,
                "name": launch_key,
                "pid": int(pid),
                "t0": time.monotonic(),
                "phase1_t0": None,
                "services_seen": {},
                "targets": [],
                "last_pending": (),
                "last_pending_emit_ts": 0.0,
            }
        self._gazebo_timing_emit(
            f"launch#{launch_id} start name={launch_key} pid={pid}"
        )
        return launch_id

    def _gazebo_launch_timing_snapshot(self) -> dict[str, Any] | None:
        with self._gazebo_launch_timing_lock:
            if self._gazebo_launch_timing is None:
                return None
            return dict(self._gazebo_launch_timing)

    def _gazebo_phase1_begin(self, targets: list[str]) -> None:
        now = time.monotonic()
        launch_id: int | None = None
        launch_elapsed: float = 0.0
        with self._gazebo_launch_timing_lock:
            meta = self._gazebo_launch_timing
            if meta is None:
                return
            meta["phase1_t0"] = now
            meta["targets"] = list(targets)
            launch_id = int(meta.get("id", 0))
            launch_elapsed = now - float(meta.get("t0", now))
        self._gazebo_timing_emit(
            "launch#"
            f"{launch_id} phase1 waiting for services={','.join(targets)} "
            f"launch_elapsed={launch_elapsed:.2f}s"
        )

    def _gazebo_note_service_ready(self, service_name: str, detected_ts: float) -> None:
        service = str(service_name or "").strip()
        if not service:
            return
        launch_id: int | None = None
        launch_elapsed: float = 0.0
        phase1_elapsed: float = 0.0
        with self._gazebo_launch_timing_lock:
            meta = self._gazebo_launch_timing
            if meta is None:
                return
            seen = meta.setdefault("services_seen", {})
            if service in seen:
                return
            seen[service] = float(detected_ts)
            launch_id = int(meta.get("id", 0))
            launch_t0 = float(meta.get("t0", detected_ts))
            phase1_t0 = float(meta.get("phase1_t0", launch_t0))
            launch_elapsed = float(detected_ts) - launch_t0
            phase1_elapsed = float(detected_ts) - phase1_t0
        self._gazebo_timing_emit(
            "launch#"
            f"{launch_id} service_ready name={service} "
            f"launch_elapsed={launch_elapsed:.2f}s phase1_elapsed={phase1_elapsed:.2f}s"
        )

    def _gazebo_note_services_pending(
        self,
        missing_services: list[str] | tuple[str, ...],
        detected_ts: float,
    ) -> None:
        missing = tuple(str(name).strip() for name in missing_services if str(name).strip())
        if not missing:
            return

        launch_id: int | None = None
        launch_elapsed: float = 0.0
        phase1_elapsed: float = 0.0
        should_emit = False
        with self._gazebo_launch_timing_lock:
            meta = self._gazebo_launch_timing
            if meta is None:
                return
            last_pending = tuple(meta.get("last_pending", ()))
            last_emit_ts = float(meta.get("last_pending_emit_ts", 0.0) or 0.0)
            if missing != last_pending or (float(detected_ts) - last_emit_ts) >= 5.0:
                meta["last_pending"] = missing
                meta["last_pending_emit_ts"] = float(detected_ts)
                should_emit = True
            launch_id = int(meta.get("id", 0))
            launch_t0 = float(meta.get("t0", detected_ts))
            phase1_t0 = float(meta.get("phase1_t0", launch_t0))
            launch_elapsed = float(detected_ts) - launch_t0
            phase1_elapsed = float(detected_ts) - phase1_t0
        if not should_emit:
            return
        self._gazebo_timing_emit(
            "launch#"
            f"{launch_id} waiting still_missing={','.join(missing)} "
            f"launch_elapsed={launch_elapsed:.2f}s phase1_elapsed={phase1_elapsed:.2f}s"
        )

    def _gazebo_phase1_complete(self) -> None:
        now = time.monotonic()
        launch_id: int | None = None
        launch_elapsed: float = 0.0
        phase1_elapsed: float = 0.0
        with self._gazebo_launch_timing_lock:
            meta = self._gazebo_launch_timing
            if meta is None:
                return
            launch_id = int(meta.get("id", 0))
            launch_t0 = float(meta.get("t0", now))
            phase1_t0 = float(meta.get("phase1_t0", launch_t0))
            launch_elapsed = now - launch_t0
            phase1_elapsed = now - phase1_t0
        self._gazebo_timing_emit(
            f"launch#{launch_id} phase1 complete launch_elapsed={launch_elapsed:.2f}s "
            f"phase1_elapsed={phase1_elapsed:.2f}s"
        )

    def _gazebo_phase1_failed(self, detail: str) -> None:
        now = time.monotonic()
        launch_id: int | None = None
        launch_elapsed: float = 0.0
        phase1_elapsed: float = 0.0
        with self._gazebo_launch_timing_lock:
            meta = self._gazebo_launch_timing
            if meta is None:
                return
            launch_id = int(meta.get("id", 0))
            launch_t0 = float(meta.get("t0", now))
            phase1_t0 = float(meta.get("phase1_t0", launch_t0))
            launch_elapsed = now - launch_t0
            phase1_elapsed = now - phase1_t0
        self._gazebo_timing_emit(
            f"launch#{launch_id} phase1 failed launch_elapsed={launch_elapsed:.2f}s "
            f"phase1_elapsed={phase1_elapsed:.2f}s detail={detail}"
        )

    def _gazebo_note_prewarm_start(self, robot: str) -> None:
        meta = self._gazebo_launch_timing_snapshot()
        if not meta:
            return
        launch_elapsed = time.monotonic() - float(meta.get("t0", time.monotonic()))
        self._gazebo_timing_emit(
            f"launch#{meta.get('id', 0)} prewarm start robot={robot} "
            f"launch_elapsed={launch_elapsed:.2f}s"
        )

    def _gazebo_note_prewarm_result(self, robot: str, ok: bool, elapsed: float, detail: str = "") -> None:
        meta = self._gazebo_launch_timing_snapshot()
        if not meta:
            return
        launch_elapsed = time.monotonic() - float(meta.get("t0", time.monotonic()))
        result = "ready" if ok else "failed"
        suffix = f" detail={detail}" if str(detail).strip() else ""
        self._gazebo_timing_emit(
            f"launch#{meta.get('id', 0)} prewarm {result} robot={robot} "
            f"elapsed={elapsed:.2f}s launch_elapsed={launch_elapsed:.2f}s{suffix}"
        )

    def _gazebo_launch_complete(self, ready_count: int, total_targets: int) -> None:
        meta = self._gazebo_launch_timing_snapshot()
        if not meta:
            return
        launch_elapsed = time.monotonic() - float(meta.get("t0", time.monotonic()))
        self._gazebo_timing_emit(
            f"launch#{meta.get('id', 0)} ready services_and_prewarm complete "
            f"targets_ready={ready_count}/{total_targets} launch_elapsed={launch_elapsed:.2f}s"
        )

    def _clear_cached_plan_safety_alerts(self) -> None:
        self._cached_plan_safety_alerts = []

    def _cache_plan_safety_alerts(self, alerts: list[dict[str, Any]]) -> None:
        self._cached_plan_safety_alerts = [dict(a) for a in alerts if isinstance(a, dict)]

    def consume_notice(self) -> str | None:
        notice = str(self.last_notice or "").strip()
        self.last_notice = None
        return notice or None

    def log_event_loop_lag(self, lag_sec: float) -> None:
        """Called by UI watchdog to diagnose websocket disconnect/freeze windows."""
        age = max(0.0, time.monotonic() - float(self._startup_phase_ts))
        self._diag_emit(
            "event-loop lag "
            f"{lag_sec:.3f}s (starting={self._starting} running={self.system_running} "
            f"phase={self._startup_phase} phase_age={age:.2f}s)"
        )

    def _maybe_start_agent_creator_prefetch(self) -> None:
        with self._agent_creator_prefetch_lock:
            if self._agent_creator_prefetch_started:
                return
            self._agent_creator_prefetch_started = True
            self._agent_creator_prefetch_thread = threading.Thread(
                target=self._prefetch_agent_creator_worker,
                name="agent_creator_prefetch",
                daemon=True,
            )
            self._agent_creator_prefetch_thread.start()

    def _prefetch_agent_creator_worker(self) -> None:
        t0 = time.monotonic()
        try:
            ac = self._import_agent_creator_module()
            self._agent_creator_cached = ac
            self._diag_emit(
                "agent_creator pre-import ready in "
                f"{time.monotonic() - t0:.2f}s"
            )
        except Exception as exc:
            # Keep startup resilient: fallback import happens on demand in start_system.
            self._diag_emit(f"agent_creator pre-import failed: {exc}")

    # ------------------------------------------------------------------
    # Product / resource discovery
    # ------------------------------------------------------------------
    def list_product_files(self) -> list[str]:
        if not _PRODUCT_DIR.exists():
            return []
        return sorted(str(p) for p in _PRODUCT_DIR.glob("*.json"))

    def list_resource_files(self) -> list[str]:
        if not _RESOURCE_DIR.exists():
            return []
        return sorted(str(p) for p in _RESOURCE_DIR.glob("*.json"))

    def load_config(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def save_config(self, path: str, data: dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Verified bundle management
    # ------------------------------------------------------------------
    @staticmethod
    def _first_manifest_entry(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if "name" in raw:
            name = str(raw.get("name", "product")).strip() or "product"
            return name, raw
        if not raw:
            raise ValueError("empty manifest")
        key = next(iter(raw.keys()))
        value = raw[key]
        if not isinstance(value, dict):
            raise ValueError("invalid manifest format")
        return str(key), value

    @staticmethod
    def _norm_path(path: str | Path) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        return str(Path(raw).resolve())

    @staticmethod
    def _abs_project_path(path_value: str | Path) -> Path:
        p = Path(str(path_value))
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p

    @staticmethod
    def _default_product_requirement_path(product_name: str) -> Path:
        return (_PRODUCT_REQUIREMENTS_DIR / f"{str(product_name).strip()}.txt").resolve()

    def _compute_safety_generation_hashes(self, safety_file: Path) -> dict[str, str]:
        if not safety_file.exists():
            raise FileNotFoundError(f"safety file missing: {safety_file}")

        safety_text = safety_file.read_text(encoding="utf-8").strip()
        if not safety_text:
            raise ValueError(f"safety file empty: {safety_file}")

        if not _TOOLS_OUT.exists():
            raise FileNotFoundError(f"tools catalogue missing: {_TOOLS_OUT}")
        prompts_path = _BASE / "prompts.py"
        if not prompts_path.exists():
            raise FileNotFoundError(f"prompts file missing: {prompts_path}")

        return {
            "safety_sha256": sha256_text(safety_text),
            "tools_sha256": sha256_file(_TOOLS_OUT),
            "prompts_sha256": sha256_file(prompts_path),
        }

    def _compute_source_hashes(self, requirement_file: Path, safety_file: Path) -> dict[str, str]:
        if not requirement_file.exists():
            raise FileNotFoundError(f"requirements file missing: {requirement_file}")

        requirement_text = requirement_file.read_text(encoding="utf-8").strip()
        if not requirement_text:
            raise ValueError(f"requirements file empty: {requirement_file}")
        hashes = self._compute_safety_generation_hashes(safety_file)
        return {
            "requirements_sha256": sha256_text(requirement_text),
            **hashes,
        }

    def _resolve_requirement_input(
        self,
        product_spec_file_or_requirement_file: str,
    ) -> tuple[Path, str | None]:
        """
        Resolve UI/runtime input into a requirements text file.

        Returns:
            (requirements_file_path, default_safety_file_path_if_known)
        """
        raw = str(product_spec_file_or_requirement_file or "").strip()
        if not raw:
            raise ValueError("product spec input is empty")

        p = Path(raw)
        if p.suffix.lower() == ".json":
            ctx = self._resolve_product_context(raw, include_hashes=False)
            return Path(str(ctx["product_spec_file"])), str(ctx.get("safety_file") or "")

        req_file = self._abs_project_path(raw)
        if not req_file.exists():
            raise FileNotFoundError(f"requirements file missing: {req_file}")
        return req_file.resolve(), None

    def _resolve_product_init_for_requirement(self, requirement_file: str) -> dict[str, Any]:
        req_norm = self._norm_path(self._abs_project_path(requirement_file))
        candidates: list[dict[str, Any]] = []
        for init_file in self.list_product_files():
            try:
                ctx = self._resolve_product_context(init_file, include_hashes=False)
            except Exception:
                continue
            out = dict(ctx)
            out["product_init_file"] = str(Path(init_file).resolve())
            candidates.append(out)
            if self._norm_path(ctx.get("product_spec_file", "")) == req_norm:
                return out

        # If there is only one product initialization manifest in the system,
        # use it as the product context and allow requirement-file override.
        if len(candidates) == 1:
            return candidates[0]

        raise ValueError(
            f"No product initialization manifest references requirements file: {requirement_file}"
        )

    def resolve_product_init_for_requirement(self, requirement_file: str) -> str:
        """Return product init JSON path that maps to the given requirement file."""
        ctx = self._resolve_product_init_for_requirement(requirement_file)
        return str(ctx["product_init_file"])

    def list_product_requirement_files(self, product_init_file: str | None = None) -> list[str]:
        options: set[str] = set()
        if _PRODUCT_REQUIREMENTS_DIR.is_dir():
            for p in _PRODUCT_REQUIREMENTS_DIR.iterdir():
                if p.is_file() and p.suffix == ".txt":
                    options.add(self._norm_path(str(p)))
        return sorted(options)

    def _load_safety_intent_approvals(self) -> dict[str, Any]:
        source = _SAFETY_INTENT_APPROVALS
        raw = self._read_json_dict(_SAFETY_INTENT_APPROVALS)
        if not raw and _legacy_safety_intent_approvals_path() != _SAFETY_INTENT_APPROVALS:
            legacy_raw = self._read_json_dict(_legacy_safety_intent_approvals_path())
            if legacy_raw:
                raw = legacy_raw
                source = _legacy_safety_intent_approvals_path()

        normalized = self._normalize_safety_intent_approvals_payload(raw)
        if source != _SAFETY_INTENT_APPROVALS or normalized != raw:
            self._save_safety_intent_approvals(normalized)
            if source != _SAFETY_INTENT_APPROVALS:
                _legacy_safety_intent_approvals_path().unlink(missing_ok=True)
        return normalized

    def _save_safety_intent_approvals(self, payload: dict[str, Any]) -> None:
        _USER_VERIFIED_SAFETY.mkdir(parents=True, exist_ok=True)
        normalized = self._normalize_safety_intent_approvals_payload(payload)
        atomic_json_write(_SAFETY_INTENT_APPROVALS, normalized)
        if _legacy_safety_intent_approvals_path() != _SAFETY_INTENT_APPROVALS:
            _legacy_safety_intent_approvals_path().unlink(missing_ok=True)

    def _load_safety_intent_previews(self) -> dict[str, Any]:
        source = _SAFETY_INTENT_PREVIEWS
        raw = self._read_json_dict(_SAFETY_INTENT_PREVIEWS)
        if not raw and _legacy_safety_intent_previews_path() != _SAFETY_INTENT_PREVIEWS:
            legacy_raw = self._read_json_dict(_legacy_safety_intent_previews_path())
            if legacy_raw:
                raw = legacy_raw
                source = _legacy_safety_intent_previews_path()
        normalized, referenced_dirs, removed_dirs = self._normalize_safety_intent_previews_payload(raw)
        if source != _SAFETY_INTENT_PREVIEWS or normalized != raw:
            _USER_VERIFIED_SAFETY.mkdir(parents=True, exist_ok=True)
            atomic_json_write(_SAFETY_INTENT_PREVIEWS, normalized)
            self._cleanup_safety_preview_dirs(referenced_dirs, removed_dirs)
            if source != _SAFETY_INTENT_PREVIEWS:
                _legacy_safety_intent_previews_path().unlink(missing_ok=True)
        return normalized

    def _save_safety_intent_previews(self, payload: dict[str, Any]) -> None:
        _USER_VERIFIED_SAFETY.mkdir(parents=True, exist_ok=True)
        normalized, referenced_dirs, removed_dirs = self._normalize_safety_intent_previews_payload(payload)
        atomic_json_write(_SAFETY_INTENT_PREVIEWS, normalized)
        self._cleanup_safety_preview_dirs(referenced_dirs, removed_dirs)
        if _legacy_safety_intent_previews_path() != _SAFETY_INTENT_PREVIEWS:
            _legacy_safety_intent_previews_path().unlink(missing_ok=True)

    @staticmethod
    def _normalize_approval_record(safety_key: str, record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record)
        out["approved"] = bool(record.get("approved", False))
        if not out["approved"]:
            out.pop("verified_file", None)
            return out

        safety_path = Path(str(safety_key)).resolve()
        if not safety_path.exists():
            out["approved"] = False
            out.pop("verified_file", None)
            return out

        raw_text = safety_path.read_text(encoding="utf-8")
        safety_text = raw_text.strip()
        if not safety_text:
            out["approved"] = False
            out.pop("verified_file", None)
            return out

        current_hash = sha256_text(safety_text)
        approved_hash = str(out.get("safety_sha256", "")).strip()
        if approved_hash != current_hash:
            out.pop("verified_file", None)
            return out

        _SAFETY_VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
        target = _SAFETY_VERIFIED_DIR / f"{slug(safety_path.stem)}__{approved_hash[:8]}.txt"
        if not target.exists() or target.read_text(encoding="utf-8") != raw_text:
            target.write_text(raw_text, encoding="utf-8")
        out["verified_file"] = str(target.resolve())
        return out

    def _normalize_safety_intent_approvals_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_approvals = payload.get("approvals", {}) if isinstance(payload, dict) else {}
        approvals = raw_approvals if isinstance(raw_approvals, dict) else {}
        normalized: dict[str, dict[str, Any]] = {}
        referenced_files: set[Path] = set()

        for raw_key, raw_record in approvals.items():
            safety_key = str(raw_key or "").strip()
            if not safety_key or not isinstance(raw_record, dict):
                continue
            record = self._normalize_approval_record(safety_key, raw_record)
            verified_file = str(record.get("verified_file", "")).strip()
            if verified_file:
                referenced_files.add(Path(verified_file).resolve())
            normalized[safety_key] = record

        self._cleanup_verified_safety_files(referenced_files)
        return {
            "schema_version": 1,
            "approvals": normalized,
        }

    @staticmethod
    def _preview_record_dir(record: dict[str, Any]) -> Path | None:
        raw = str(record.get("preview_dir", "")).strip()
        if not raw:
            return None
        try:
            candidate = Path(raw).resolve()
        except Exception:
            return None
        for root in (_SAFETY_PREVIEW_DIR, _legacy_safety_preview_dir()):
            try:
                candidate.relative_to(root.resolve())
                return candidate
            except Exception:
                continue
        return None

    def _normalize_preview_record(self, record: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
        out = dict(record)
        preview_dir = self._preview_record_dir(record)
        preview_id = str(out.get("preview_id", "")).strip()
        out["tools_sha256"] = str(out.get("tools_sha256", "") or "").strip()
        out["prompts_sha256"] = str(out.get("prompts_sha256", "") or "").strip()

        if preview_dir is not None:
            legacy_root = _legacy_safety_preview_dir().resolve()
            try:
                preview_dir.relative_to(legacy_root)
                target_dir = (_SAFETY_PREVIEW_DIR / preview_dir.name).resolve()
                _SAFETY_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                if preview_dir.exists():
                    if target_dir.exists():
                        shutil.rmtree(preview_dir, ignore_errors=True)
                    else:
                        shutil.move(str(preview_dir), str(target_dir))
                preview_dir = target_dir
            except Exception:
                pass
        elif preview_id:
            preview_dir = (_SAFETY_PREVIEW_DIR / preview_id).resolve()

        if preview_dir is not None:
            out["preview_dir"] = str(preview_dir)
            logic_path = preview_dir / "cca_safety_logic.json"
            out["safety_logic_json"] = str(logic_path.resolve())
            out["dfa_dot_files"] = sorted(str(p.resolve()) for p in preview_dir.glob("SAFE_*_dfa.dot"))
            out["dfa_png_files"] = sorted(str(p.resolve()) for p in preview_dir.glob("SAFE_*_dfa.png"))
            logic_payload = self._read_json_dict(logic_path)
            rules = logic_payload.get("rules", []) if isinstance(logic_payload, dict) else []
            if isinstance(rules, list):
                out["rules_count"] = len([rule for rule in rules if isinstance(rule, dict)])

        return out, preview_dir

    @staticmethod
    def _rule_ids_from_safety_logic_payload(payload: dict[str, Any]) -> list[str]:
        raw_rules = payload.get("rules", []) if isinstance(payload, dict) else []
        rules = raw_rules if isinstance(raw_rules, list) else []
        out: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", "")).strip()
            if rid:
                out.append(rid)
        return out

    @staticmethod
    def _dfa_rule_id_from_path(path: Path) -> str:
        rid = str(path.stem or "").strip()
        if rid.endswith("_dfa"):
            rid = rid[: -len("_dfa")]
        return rid

    def _build_precomputed_safety_descriptor(
        self,
        safety_path: Path,
        record: dict[str, Any],
        current_hashes: dict[str, str],
    ) -> tuple[dict[str, Any] | None, str]:
        approved_tools_hash = str(record.get("tools_sha256", "") or "").strip()
        approved_prompts_hash = str(record.get("prompts_sha256", "") or "").strip()
        preview_id = str(record.get("preview_id", "") or "").strip()
        if not preview_id or not approved_tools_hash or not approved_prompts_hash:
            return None, "approved_preview_provenance_missing"
        if approved_tools_hash != current_hashes["tools_sha256"]:
            return None, "tools_changed_since_approval"
        if approved_prompts_hash != current_hashes["prompts_sha256"]:
            return None, "prompts_changed_since_approval"

        safety_key = self._norm_path(safety_path)
        payload = self._load_safety_intent_previews()
        previews = payload.get("previews", {})
        entries = self._preview_history_entries(
            previews.get(safety_key, []) if isinstance(previews, dict) else []
        )
        preview_record = self._find_preview_record(entries, preview_id)
        if not preview_record:
            return None, "approved_preview_missing"

        logic_path_raw = str(preview_record.get("safety_logic_json", "")).strip()
        logic_path = Path(logic_path_raw) if logic_path_raw else Path()
        if not logic_path.exists():
            return None, "approved_preview_artifacts_missing"

        logic_payload = self._read_json_dict(logic_path)
        rule_ids = self._rule_ids_from_safety_logic_payload(logic_payload)

        dot_paths: list[str] = []
        dot_rule_ids: set[str] = set()
        for raw_dot in preview_record.get("dfa_dot_files", []):
            dot_path = Path(str(raw_dot or "").strip())
            if not dot_path.exists():
                return None, "approved_preview_artifacts_missing"
            resolved = str(dot_path.resolve())
            dot_paths.append(resolved)
            rid = self._dfa_rule_id_from_path(dot_path)
            if rid:
                dot_rule_ids.add(rid)
        if any(rid not in dot_rule_ids for rid in rule_ids):
            return None, "approved_preview_artifacts_missing"

        png_paths: list[str] = []
        for raw_png in preview_record.get("dfa_png_files", []):
            png_path = Path(str(raw_png or "").strip())
            if not png_path.exists():
                return None, "approved_preview_artifacts_missing"
            png_paths.append(str(png_path.resolve()))

        return (
            {
                "mode": "approved_preview",
                "preview_id": preview_id,
                "preview_generated_at_utc": str(
                    preview_record.get("generated_at_utc", "") or ""
                ).strip(),
                "safety_logic_json": str(logic_path.resolve()),
                "dfa_dot_files": sorted(dot_paths),
                "dfa_png_files": sorted(png_paths),
                "safety_sha256": current_hashes["safety_sha256"],
                "tools_sha256": current_hashes["tools_sha256"],
                "prompts_sha256": current_hashes["prompts_sha256"],
            },
            "approved",
        )

    @staticmethod
    def _approval_requires_preview_refresh(reason: str, record: dict[str, Any]) -> bool:
        if not isinstance(record, dict) or not bool(record.get("approved", False)):
            return False
        return str(reason or "").strip() in {
            "content_changed_since_approval",
            "tools_changed_since_approval",
            "prompts_changed_since_approval",
            "approved_preview_provenance_missing",
            "approved_preview_missing",
            "approved_preview_artifacts_missing",
        }

    @staticmethod
    def _approval_refresh_error(reason: str) -> str:
        mapping = {
            "content_changed_since_approval": (
                "Safety approval is stale because the safety file changed. "
                "Regenerate the safety preview and approve it again before generating a plan set."
            ),
            "tools_changed_since_approval": (
                "Safety approval is stale because the tools catalog changed. "
                "Regenerate the safety preview and approve it again before generating a plan set."
            ),
            "prompts_changed_since_approval": (
                "Safety approval is stale because prompts.py changed. "
                "Regenerate the safety preview and approve it again before generating a plan set."
            ),
            "approved_preview_provenance_missing": (
                "Approved safety preview metadata is incomplete. "
                "Regenerate the safety preview and approve it again before generating a plan set."
            ),
            "approved_preview_missing": (
                "The approved safety preview could not be found. "
                "Regenerate the safety preview and approve it again before generating a plan set."
            ),
            "approved_preview_artifacts_missing": (
                "The approved safety preview artifacts are incomplete or missing. "
                "Regenerate the safety preview and approve it again before generating a plan set."
            ),
        }
        return mapping.get(
            str(reason or "").strip(),
            "Approved safety preview is stale. Regenerate the safety preview and approve it again before generating a plan set.",
        )

    def _normalize_safety_intent_previews_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], set[Path], set[Path]]:
        raw_previews = payload.get("previews", {}) if isinstance(payload, dict) else {}
        previews = raw_previews if isinstance(raw_previews, dict) else {}
        normalized_previews: dict[str, list[dict[str, Any]]] = {}
        referenced_dirs: set[Path] = set()
        removed_dirs: set[Path] = set()

        for raw_key, raw_entries in previews.items():
            safety_key = str(raw_key or "").strip()
            if not safety_key:
                continue
            entries = self._preview_history_entries(raw_entries)
            normalized_entries: list[tuple[dict[str, Any], Path | None]] = []
            for entry in entries:
                normalized_entry, preview_dir = self._normalize_preview_record(entry)
                logic_path = Path(str(normalized_entry.get("safety_logic_json", "")).strip())
                if (
                    preview_dir is None
                    or not preview_dir.exists()
                    or not logic_path.exists()
                ):
                    continue
                normalized_entries.append((normalized_entry, preview_dir.resolve()))
            kept_entries = normalized_entries[:_SAFETY_PREVIEW_HISTORY_LIMIT]
            if kept_entries:
                normalized_previews[safety_key] = [entry for entry, _ in kept_entries]
            for _, preview_dir in kept_entries:
                if preview_dir is not None:
                    referenced_dirs.add(preview_dir)
            for _, preview_dir in normalized_entries[_SAFETY_PREVIEW_HISTORY_LIMIT:]:
                if preview_dir is not None:
                    removed_dirs.add(preview_dir)

        return (
            {
                "schema_version": 1,
                "previews": normalized_previews,
            },
            referenced_dirs,
            removed_dirs,
        )

    @staticmethod
    def _cleanup_safety_preview_dirs(referenced_dirs: set[Path], removed_dirs: set[Path]) -> None:
        for preview_dir in sorted(removed_dirs):
            shutil.rmtree(preview_dir, ignore_errors=True)

        for root in (_SAFETY_PREVIEW_DIR, _legacy_safety_preview_dir()):
            if not root.exists():
                continue
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    resolved = child.resolve()
                    resolved.relative_to(root.resolve())
                except Exception:
                    continue
                if resolved in referenced_dirs:
                    continue
                shutil.rmtree(resolved, ignore_errors=True)
            if root == _legacy_safety_preview_dir():
                try:
                    next(root.iterdir())
                except StopIteration:
                    root.rmdir()
                except Exception:
                    pass

    @staticmethod
    def _cleanup_verified_safety_files(referenced_files: set[Path]) -> None:
        if not _SAFETY_VERIFIED_DIR.exists():
            return
        for child in _SAFETY_VERIFIED_DIR.iterdir():
            if not child.is_file():
                continue
            try:
                resolved = child.resolve()
                resolved.relative_to(_SAFETY_VERIFIED_DIR.resolve())
            except Exception:
                continue
            if resolved in referenced_files:
                continue
            child.unlink(missing_ok=True)

    @staticmethod
    def _preview_history_entries(entries: Any) -> list[dict[str, Any]]:
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    @staticmethod
    def _parse_inline_debug_list(raw: str) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(text)
            except Exception:
                continue
            if isinstance(value, (list, tuple, set)):
                out: list[str] = []
                for item in value:
                    token = str(item or "").strip()
                    if token:
                        out.append(token)
                return out
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1]
            out = []
            for part in inner.split(","):
                token = part.strip().strip("'\"")
                if token:
                    out.append(token)
            return out
        return [text]

    @classmethod
    def _extract_debug_list(cls, message: str, pattern: str) -> list[str]:
        match = re.search(pattern, str(message or ""), flags=re.IGNORECASE)
        if not match:
            return []
        return cls._parse_inline_debug_list(match.group(1))

    @classmethod
    def _classify_safety_preview_failure(cls, error_message: str) -> dict[str, Any]:
        message = str(error_message or "").strip() or "unknown error"
        lower = message.lower()
        title = "Safety preview generation failed"
        summary = (
            "The generated safety logic could not be grounded to the current tool, "
            "state, and resource catalog."
        )
        details = [
            "The compiler rejected the generated safety rule because at least one selector "
            "or formula node could not be mapped to supported catalog entries."
        ]
        suggestions = [
            "Reuse exact context keys, function names, and state names that exist in the tool catalog.",
            "Keep each selector aligned to one consistent context role and one consistent state slice.",
            "If the rule already names concrete resources, keep those resources explicit in the logic.",
        ]
        category = "grounding_failure"

        if "uses unresolved context keys" in lower:
            bad_keys = cls._extract_debug_list(message, r"unresolved context keys (\[[^\]]*\])")
            canonical_roles = cls._extract_debug_list(
                message,
                r"candidate canonical context roles are (\[[^\]]*\])",
            )
            title = "Context grounding is ambiguous"
            summary = (
                "The generated selector used context keys that are not canonical in the tool catalog, "
                "and the compiler could not infer one unique replacement."
            )
            details = []
            if bad_keys:
                details.append(
                    "Unsupported selector context keys: " + ", ".join(bad_keys)
                )
            if canonical_roles:
                details.append(
                    "Matching tool rows still pointed to multiple canonical context roles: "
                    + ", ".join(canonical_roles)
                )
            details.append(
                "This usually means the selector used generic wording such as station/location or mixed "
                "different action families in one occupancy condition."
            )
            suggestions = [
                "Refine the requirement or feedback so the condition maps to one canonical context role only.",
                "Avoid generic context names like station/location and use the tool family's exact role names.",
                "Do not mix pick-side and place-side states or functions in one selector branch.",
            ]
            category = "unresolved_context_keys"
        elif "mixes incompatible tool families" in lower:
            families = cls._extract_debug_list(message, r"families=(\[[^\]]*\])")
            title = "The generated selector mixes incompatible tool families"
            summary = (
                "The selector grounded to rows that belong to different context/state families, "
                "so the compiler refused to merge them into one condition."
            )
            details = []
            if families:
                details.append("Incompatible grounded families: " + ", ".join(families))
            details.append(
                "This usually happens when one selector combines origin-side and destination-side behavior."
            )
            suggestions = [
                "Split the condition so each selector covers only one family of tool rows.",
                "Keep pick-side functions/states separate from place-side functions/states.",
                "Use refinement feedback to say that one station/occupancy condition should map to one side only.",
            ]
            category = "family_mismatch"
        elif "uses multiple distinct resource_var names" in lower:
            resource_vars = cls._extract_debug_list(
                message,
                r"resource_var names (\[[^\]]*\])",
            )
            title = "Resource binding is inconsistent"
            summary = (
                "The generated formula used multiple different resource variables in one rule, "
                "which this compiler does not support."
            )
            details = []
            if resource_vars:
                details.append("Distinct resource variables used: " + ", ".join(resource_vars))
            details.append(
                "This can collapse or distort multi-resource mutex logic, so the compiler fails early."
            )
            suggestions = [
                "If the rule already names specific resources, keep them as concrete resources in the AST.",
                "If the pattern should repeat generically, use one shared resource_var instead of several.",
                "Avoid introducing separate placeholders like $r1 and $r2 in a single mutex rule.",
            ]
            category = "resource_binding"
        elif "references fewer than two concrete resources" in lower:
            used_resources = cls._extract_debug_list(
                message,
                r"used_resources=(\[[^\]]*\])",
            )
            expected_resources = cls._extract_debug_list(
                message,
                r"expected_resources=(\[[^\]]*\])",
            )
            title = "The generated mutex does not cover both resources"
            summary = (
                "The rule is a multi-resource mutex, but the compiled formula ended up referring to too few "
                "concrete resources."
            )
            details = []
            if used_resources:
                details.append("Resources actually referenced in the compiled formula: " + ", ".join(used_resources))
            if expected_resources:
                details.append("Resources expected from the rule: " + ", ".join(expected_resources))
            suggestions = [
                "Describe the safety intent as a cross-resource exclusion, not a one-resource condition.",
                "Keep both named resources explicit when the requirement already names them.",
                "Use refinement feedback to say that the two resources must not satisfy the condition simultaneously.",
            ]
            category = "degenerate_mutex"
        elif "degenerates into independent single-resource conjuncts" in lower:
            title = "The generated mutex collapsed into self-constraints"
            summary = (
                "The compiled formula split into separate single-resource clauses instead of one real cross-resource mutex."
            )
            details = [
                "This means the generated logic prevented each resource from conflicting with itself, "
                "rather than preventing the two resources from conflicting with each other."
            ]
            suggestions = [
                "Refine the requirement or feedback to emphasize that the exclusion is between the two resources.",
                "Avoid logic that expands into one conjunct per resource.",
                "Keep the two sides of the mutex in one shared conflict condition.",
            ]
            category = "degenerate_mutex"
        elif "requested states" in lower and "before context grounding" in lower:
            requested_states = cls._extract_debug_list(
                message,
                r"requested states (\[[^\]]*\])",
            )
            available_states = cls._extract_debug_list(
                message,
                r"out_states (\[[^\]]*\])",
            )
            title = "The generated state slice does not exist in the matched tools"
            summary = (
                "The selector asked for persistent states that are not exposed by the matching tool rows."
            )
            details = []
            if requested_states:
                details.append("Requested states: " + ", ".join(requested_states))
            if available_states:
                details.append("Available persistent states from matching rows: " + ", ".join(available_states))
            suggestions = [
                "Use only persistent states that are actually produced by the matching tool family.",
                "Do not mix generic idle states with action-specific occupancy states unless the catalog supports that slice.",
                "Refine the requirement so it describes one narrower operational state family.",
            ]
            category = "state_mismatch"
        elif "did not match any tool rows before context grounding" in lower:
            title = "The generated selector does not match any supported tool rows"
            summary = (
                "Even before applying context filters, the selector's process/resource/function constraints did not "
                "match the current tool catalog."
            )
            details = [
                "This usually means the generated logic used unsupported function names, resource types, or process labels."
            ]
            suggestions = [
                "Reuse exact function names and process names from the tool catalog.",
                "Avoid invented events or tool families.",
                "If the requirement is more abstract, use a selector that lets the compiler infer the concrete APs.",
            ]
            category = "no_matching_rows"
        elif "could not ground context" in lower:
            title = "The selector context could not be grounded"
            summary = (
                "The selector matched a tool family, but the context object still did not map to any concrete tool rows."
            )
            details = [
                "This usually means the context value or key does not line up with the matched tool family's required context."
            ]
            suggestions = [
                "Use exact catalog context keys and identifiers.",
                "Keep the selector narrow enough that one context family is implied.",
                "Avoid mixing unrelated states/functions that force the selector across multiple families.",
            ]
            category = "context_grounding"
        elif "did not expand to any aps" in lower:
            title = "The selector over-constrained the safety condition"
            summary = (
                "The selector compiled successfully enough to resolve its shape, but no concrete APs survived expansion."
            )
            details = [
                "This usually means the selector combined context, functions, or states too narrowly for any real APs to remain."
            ]
            suggestions = [
                "Remove unsupported or overly narrow function/state filters.",
                "Check that the rule's context and state slice match the same tool family.",
                "Use refinement feedback to describe the intended operational condition more directly.",
            ]
            category = "empty_selector"

        return {
            "category": category,
            "title": title,
            "summary": summary,
            "details": details,
            "suggestions": suggestions,
            "raw_error": message,
        }

    def _record_safety_preview_failure(
        self,
        safety_key: str,
        error_message: str,
        *,
        safety_sha256: str = "",
        refinement_feedback: str = "",
        parent_preview_id: str = "",
    ) -> None:
        key = str(safety_key or "").strip()
        if not key:
            return
        payload = self._classify_safety_preview_failure(error_message)
        payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload["safety_sha256"] = str(safety_sha256 or "").strip()
        payload["refinement_feedback"] = str(refinement_feedback or "").strip()
        payload["parent_preview_id"] = str(parent_preview_id or "").strip()
        self._safety_preview_failures[key] = payload

    def _clear_safety_preview_failure(self, safety_key: str) -> None:
        key = str(safety_key or "").strip()
        if not key:
            return
        self._safety_preview_failures.pop(key, None)

    def _get_safety_preview_failure(
        self,
        safety_key: str,
        *,
        current_hash: str = "",
    ) -> dict[str, Any]:
        key = str(safety_key or "").strip()
        failure = self._safety_preview_failures.get(key)
        if not isinstance(failure, dict):
            return {}
        payload = dict(failure)
        failure_hash = str(payload.get("safety_sha256", "")).strip()
        payload["hash_matches_current"] = bool(
            current_hash and failure_hash and current_hash == failure_hash
        )
        return payload

    @staticmethod
    def _find_preview_record(
        entries: list[dict[str, Any]], preview_id: str
    ) -> dict[str, Any]:
        target = str(preview_id or "").strip()
        if not target:
            return {}
        for entry in entries:
            if str(entry.get("preview_id", "")).strip() == target:
                return entry
        return {}

    def _preview_record_rules(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(record, dict):
            return []
        preview_dir = Path(str(record.get("preview_dir", "")).strip())
        logic_path_raw = str(record.get("safety_logic_json", "")).strip()
        logic_path = Path(logic_path_raw) if logic_path_raw else (preview_dir / "cca_safety_logic.json")
        if not logic_path.exists():
            return []
        logic_payload = self._read_json_dict(logic_path)
        raw_rules = logic_payload.get("rules", [])
        return raw_rules if isinstance(raw_rules, list) else []

    @staticmethod
    def _preview_prompt_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt_rules: list[dict[str, Any]] = []
        for idx, rule in enumerate(rules, start=1):
            if not isinstance(rule, dict):
                continue
            prompt_rules.append(
                {
                    "id": str(rule.get("id", "")).strip() or f"SAFE_{idx}",
                    "raw_text": str(rule.get("raw_text", "")).strip(),
                    "constraint_type": str(rule.get("constraint_type", "")).strip(),
                    "process": rule.get("process"),
                    "product": rule.get("product"),
                    "resources": rule.get("resources"),
                    "event": rule.get("event"),
                    "context": rule.get("context"),
                    "aps": [
                        str(ap.get("full", "")).strip()
                        for ap in (rule.get("aps") or [])
                        if isinstance(ap, dict) and str(ap.get("full", "")).strip()
                    ],
                    "ltlf": str(rule.get("ltlf", "")).strip(),
                }
            )
        return prompt_rules

    @staticmethod
    def _rule_aps_signature(rule: dict[str, Any]) -> list[str]:
        aps = rule.get("aps", []) if isinstance(rule, dict) else []
        out: list[str] = []
        for ap in aps if isinstance(aps, list) else []:
            if isinstance(ap, dict):
                full = str(ap.get("full", "")).strip()
                if full:
                    out.append(full)
            else:
                text = str(ap or "").strip()
                if text:
                    out.append(text)
        return sorted(set(out))

    @classmethod
    def _summarize_safety_preview_diff(
        cls,
        previous_rules: list[dict[str, Any]],
        current_rules: list[dict[str, Any]],
    ) -> str:
        if not previous_rules:
            return "No previous preview comparison available."

        previous_map = {
            str(rule.get("id", "")).strip(): rule
            for rule in previous_rules
            if isinstance(rule, dict) and str(rule.get("id", "")).strip()
        }
        current_map = {
            str(rule.get("id", "")).strip(): rule
            for rule in current_rules
            if isinstance(rule, dict) and str(rule.get("id", "")).strip()
        }

        lines: list[str] = []
        previous_ids = set(previous_map)
        current_ids = set(current_map)
        for rid in sorted(current_ids - previous_ids):
            lines.append(f"{rid}: new rule in current preview.")
        for rid in sorted(previous_ids - current_ids):
            lines.append(f"{rid}: removed from current preview.")

        shared_ids = sorted(previous_ids & current_ids)
        for rid in shared_ids:
            old = previous_map[rid]
            new = current_map[rid]
            changed_fields: list[str] = []
            for field in ("raw_text", "constraint_type", "ltlf"):
                if str(old.get(field, "")).strip() != str(new.get(field, "")).strip():
                    changed_fields.append(field)
            if cls._rule_aps_signature(old) != cls._rule_aps_signature(new):
                changed_fields.append("aps")
            if changed_fields:
                lines.append(f"{rid}: changed {', '.join(changed_fields)}.")

        if not lines:
            return "Current preview matches the previous preview on rule text, APs, and LTLf."
        return "\n".join(lines)

    @staticmethod
    def _preview_history_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for entry in entries[:_SAFETY_PREVIEW_HISTORY_LIMIT]:
            summary.append(
                {
                    "preview_id": str(entry.get("preview_id", "")).strip(),
                    "generated_at_utc": str(entry.get("generated_at_utc", "")).strip(),
                    "parent_preview_id": str(entry.get("parent_preview_id", "")).strip(),
                    "refinement_feedback": str(entry.get("refinement_feedback", "")).strip(),
                    "rules_count": int(entry.get("rules_count", 0) or 0),
                }
            )
        return summary

    @staticmethod
    def _flatten_dfa_transitions(parsed_dfa: dict[str, Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        transitions = parsed_dfa.get("transitions", {}) if isinstance(parsed_dfa, dict) else {}
        if not isinstance(transitions, dict):
            return out
        for src, arcs in transitions.items():
            if not isinstance(arcs, list):
                continue
            for arc in arcs:
                if not isinstance(arc, (list, tuple)) or len(arc) < 2:
                    continue
                label = str(arc[0])
                dst = str(arc[1])
                out.append(
                    {
                        "from": str(src),
                        "condition": label,
                        "to": dst,
                    }
                )
        return out

    @staticmethod
    def _dfa_plain_meaning(rule: dict[str, Any], parsed_dfa: dict[str, Any]) -> str:
        rid = str(rule.get("id", "")).strip() or "rule"
        initial = str(parsed_dfa.get("initial", "")).strip() or "unknown"
        violation = str(parsed_dfa.get("violation_state", "")).strip()
        aps = rule.get("aps", []) if isinstance(rule.get("aps"), list) else []
        ap_parts: list[str] = []
        for ap in aps:
            if not isinstance(ap, dict):
                continue
            label = str(ap.get("label", "")).strip()
            full = str(ap.get("full", "")).strip()
            if label and full:
                ap_parts.append(f"{label}={full}")
        ap_desc = ", ".join(ap_parts) if ap_parts else "no AP mapping"
        if violation:
            return (
                f"{rid}: DFA starts at state {initial}. Transitions follow edge conditions over AP labels. "
                f"If it reaches state {violation}, the rule is violated. AP map: {ap_desc}."
            )
        return (
            f"{rid}: DFA starts at state {initial}. Follow transitions using AP labels and edge conditions. "
            f"No explicit violation sink was detected. AP map: {ap_desc}."
        )

    @staticmethod
    def _ltlf_plain_feedback(rule: dict[str, Any]) -> str:
        """Fallback operator-facing interpretation of the generated AP + LTLf."""
        ltlf = str(rule.get("ltlf", "")).strip()
        aps = rule.get("aps", []) if isinstance(rule.get("aps"), list) else []

        ap_map: dict[str, str] = {}
        for ap in aps:
            if not isinstance(ap, dict):
                continue
            label = str(ap.get("label", "")).strip()
            full = str(ap.get("full", "")).strip()
            if not label or not full:
                continue
            ap_map[label] = full
        normalized = re.sub(r"\s+", "", ltlf)
        m_order = (
            re.fullmatch(r"\(\(!?(ap\d+)\)U(ap\d+)\)", normalized)
            or re.fullmatch(r"\(!?(ap\d+)\)U(ap\d+)", normalized)
        )
        m_response = re.fullmatch(r"G\((ap\d+)->F(ap\d+)\)", normalized)

        if ltlf.startswith("G !(") or ltlf.startswith("G!("):
            events = [full for _, full in sorted(ap_map.items())]
            if events:
                return (
                    "This generated rule forbids simultaneous overlap among these grounded events: "
                    + "; ".join(events)
                    + "."
                )
            return "This generated rule is a global mutual-exclusion constraint over the generated AP events."
        if m_response:
            trigger = ap_map.get(m_response.group(1), m_response.group(1))
            response = ap_map.get(m_response.group(2), m_response.group(2))
            return f"Whenever {trigger} occurs, {response} must eventually follow."
        if m_order:
            earlier = ap_map.get(m_order.group(1), m_order.group(1))
            later = ap_map.get(m_order.group(2), m_order.group(2))
            return f"The generated ordering requires {earlier} to happen before {later}."
        if " U " in ltlf:
            return (
                "This generated rule uses an until-condition: the left-hand condition must hold "
                "until the right-hand condition becomes true."
            )
        if ltlf.startswith("G(") or ltlf.startswith("G "):
            return "This generated rule is a global constraint that must hold throughout execution."
        if ltlf:
            return f"This generated rule constrains execution according to the formula {ltlf}."
        return "No generated interpretation is available for this rule."

    @classmethod
    def _preview_interpretation_summary(cls, rules: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for idx, rule in enumerate(rules, start=1):
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", "")).strip() or f"SAFE_{idx}"
            interpretation = str(rule.get("generated_interpretation", "")).strip() or cls._ltlf_plain_feedback(rule)
            lines.append(f"- {rid}: {interpretation}")
        return "\n".join(lines) if lines else "No generated rule interpretation available."

    @staticmethod
    def _diagnose_dfa_artifact(dot_text: str, parsed_dfa: dict[str, Any]) -> tuple[str, str]:
        raw = str(dot_text or "").strip()
        if not raw:
            return ("missing", "No DFA artifact was generated for this rule.")

        transitions = parsed_dfa.get("transitions", {}) if isinstance(parsed_dfa, dict) else {}
        has_labeled_transitions = bool(re.search(r"\[label=\".+?\"\]", raw))
        has_parsed_transitions = False
        if isinstance(transitions, dict):
            has_parsed_transitions = any(bool(edges) for edges in transitions.values())

        if has_labeled_transitions and has_parsed_transitions:
            return ("ok", "")

        if shutil.which("mona") is None:
            return (
                "backend_missing",
                "DFA graph unavailable: the MONA executable is not installed, so ltlf2dfa returned placeholder output.",
            )
        return (
            "invalid",
            "DFA graph unavailable: ltlf2dfa returned placeholder DOT with no labeled transitions.",
        )

    def _parse_rule_dfas(
        self,
        dfa_map: dict[str, str],
        rules: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not dfa_map:
            return {}
        try:
            from cais_spade_llm.agents.central_controller.base_safety_checker import (
                BaseSafetyChecker,
            )
        except ImportError:
            from agents.central_controller.base_safety_checker import BaseSafetyChecker
        try:
            checker = BaseSafetyChecker(dfa_dots=dfa_map, safety_rules=rules)
            parsed = getattr(checker, "dfas", {})
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def get_safety_rule_preview(self, safety_requirement_file: str) -> dict[str, Any]:
        raw = str(safety_requirement_file or "").strip()
        if not raw:
            return {
                "available": False,
                "reason": "safety_file_missing",
                "safety_file": "",
                "failure": {},
                "rules": [],
            }

        safety_path = self._abs_project_path(raw).resolve()
        safety_key = self._norm_path(safety_path)
        exists = safety_path.exists()
        safety_text = safety_path.read_text(encoding="utf-8").strip() if exists else ""
        current_hash = sha256_text(safety_text) if safety_text else ""
        failure = self._get_safety_preview_failure(safety_key, current_hash=current_hash)

        payload = self._load_safety_intent_previews()
        previews = payload.get("previews", {})
        entries = self._preview_history_entries(
            previews.get(safety_key, []) if isinstance(previews, dict) else []
        )
        if not entries:
            return {
                "available": False,
                "reason": "not_generated",
                "safety_file": safety_key,
                "current_hash": current_hash,
                "hash_matches_current": False,
                "failure": failure,
                "rules": [],
            }

        latest = entries[0] if isinstance(entries[0], dict) else {}
        preview_dir = Path(str(latest.get("preview_dir", "")).strip())
        logic_path_raw = str(latest.get("safety_logic_json", "")).strip()
        logic_path = Path(logic_path_raw) if logic_path_raw else (preview_dir / "cca_safety_logic.json")
        if not logic_path.exists():
            return {
                "available": False,
                "reason": "preview_artifacts_missing",
                "safety_file": safety_key,
                "current_hash": current_hash,
                "hash_matches_current": False,
                "record": latest,
                "failure": failure,
                "rules": [],
            }

        logic_payload = self._read_json_dict(logic_path)
        raw_rules = logic_payload.get("rules", [])
        rules: list[dict[str, Any]] = raw_rules if isinstance(raw_rules, list) else []
        preview_interpretation_summary = str(
            logic_payload.get("preview_interpretation_summary", "") or ""
        ).strip()
        parent_preview_id = str(latest.get("parent_preview_id", "")).strip()
        parent_record = self._find_preview_record(entries[1:], parent_preview_id)
        if not parent_record and len(entries) > 1:
            parent_record = entries[1]
        parent_rules = self._preview_record_rules(parent_record)
        diff_summary = self._summarize_safety_preview_diff(parent_rules, rules)

        dfa_map: dict[str, str] = {}
        for rule in rules:
            rid = str(rule.get("id", "")).strip()
            if not rid:
                continue
            dot_path = preview_dir / f"{rid}_dfa.dot"
            if not dot_path.exists():
                continue
            try:
                dfa_map[rid] = dot_path.read_text(encoding="utf-8")
            except Exception:
                continue

        parsed_dfas = self._parse_rule_dfas(dfa_map, rules)
        preview_rules: list[dict[str, Any]] = []
        for idx, rule in enumerate(rules, start=1):
            rid = str(rule.get("id", "")).strip() or f"SAFE_{idx}"
            parsed = parsed_dfas.get(rid, {})
            dot_path = preview_dir / f"{rid}_dfa.dot"
            png_path = preview_dir / f"{rid}_dfa.png"
            generated_interpretation = str(rule.get("generated_interpretation", "")).strip() or self._ltlf_plain_feedback(rule)
            dfa_status, dfa_diagnostic = self._diagnose_dfa_artifact(dfa_map.get(rid, ""), parsed)
            preview_rules.append(
                {
                    "id": rid,
                    "raw_text": str(rule.get("raw_text", "")),
                    "constraint_type": str(rule.get("constraint_type", "")),
                    "ltlf": str(rule.get("ltlf", "")),
                    "aps": rule.get("aps", []) if isinstance(rule.get("aps"), list) else [],
                    "generated_interpretation": generated_interpretation,
                    "ltlf_plain_feedback": generated_interpretation,
                    "dfa_dot": dfa_map.get(rid, ""),
                    "dfa_dot_path": str(dot_path) if dot_path.exists() else "",
                    "dfa_png_path": str(png_path) if png_path.exists() else "",
                    "dfa_initial_state": str(parsed.get("initial", "")),
                    "dfa_violation_state": str(parsed.get("violation_state", "")),
                    "dfa_ap_symbols": parsed.get("ap_symbols", []) if isinstance(parsed.get("ap_symbols"), list) else [],
                    "dfa_transitions": self._flatten_dfa_transitions(parsed),
                    "dfa_meaning": self._dfa_plain_meaning(rule, parsed),
                    "dfa_status": dfa_status,
                    "dfa_diagnostic": dfa_diagnostic,
                }
            )

        if not preview_interpretation_summary:
            preview_interpretation_summary = self._preview_interpretation_summary(preview_rules)

        preview_hash = str(latest.get("safety_sha256", "")).strip()
        hash_matches = bool(current_hash and preview_hash and current_hash == preview_hash)
        return {
            "available": True,
            "reason": "ok",
            "safety_file": safety_key,
            "current_hash": current_hash,
            "hash_matches_current": hash_matches,
            "record": latest,
            "failure": failure,
            "refinement_feedback": str(latest.get("refinement_feedback", "")).strip(),
            "parent_record": {
                "preview_id": str(parent_record.get("preview_id", "")).strip(),
                "generated_at_utc": str(parent_record.get("generated_at_utc", "")).strip(),
            }
            if parent_record
            else {},
            "history": self._preview_history_summary(entries),
            "diff_summary": diff_summary,
            "preview_interpretation_summary": preview_interpretation_summary,
            "rules": preview_rules,
        }

    def generate_safety_rule_preview(
        self,
        safety_requirement_file: str,
        *,
        refinement_feedback: str = "",
        parent_preview_id: str = "",
    ) -> dict[str, Any]:
        if self.system_running or self._starting or self._stopping:
            raise RuntimeError("cannot generate safety preview while system lifecycle is active")
        self._ensure_called_from_worker_thread("generate_safety_rule_preview")

        raw = str(safety_requirement_file or "").strip()
        if not raw:
            raise ValueError("safety requirement file is required")

        safety_path = self._abs_project_path(raw).resolve()
        if not safety_path.exists():
            raise FileNotFoundError(f"safety requirement file missing: {safety_path}")
        safety_text = safety_path.read_text(encoding="utf-8").strip()
        if not safety_text:
            raise ValueError(f"safety requirement file is empty: {safety_path}")

        safety_hashes = self._compute_safety_generation_hashes(safety_path)
        safety_hash = safety_hashes["safety_sha256"]
        safety_key = self._norm_path(safety_path)
        payload = self._load_safety_intent_previews()
        previews = payload.get("previews", {})
        if not isinstance(previews, dict):
            previews = {}
        history = self._preview_history_entries(previews.get(safety_key, []))
        requested_parent_id = str(parent_preview_id or "").strip()
        parent_record: dict[str, Any] = {}
        if requested_parent_id:
            parent_record = self._find_preview_record(history, requested_parent_id)
            if not parent_record:
                raise ValueError(f"parent preview not found: {requested_parent_id}")
        elif str(refinement_feedback or "").strip() and history:
            parent_record = history[0]
        prompt_preview_rules = self._preview_prompt_rules(
            self._preview_record_rules(parent_record)
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        preview_id = f"{stamp}__{slug(safety_path.stem)}__{safety_hash[:8]}"
        preview_dir = _SAFETY_PREVIEW_DIR / preview_id
        suffix = 1
        while preview_dir.exists():
            suffix += 1
            preview_id = f"{stamp}__{slug(safety_path.stem)}__{safety_hash[:8]}_{suffix}"
            preview_dir = _SAFETY_PREVIEW_DIR / preview_id
        preview_dir.mkdir(parents=True, exist_ok=False)

        _, CentralControllerAgent, _, _, _ = self.bundle_compiler._import_runtime_classes()
        resources = self.bundle_compiler._collect_resource_refs(
            robot_env=str(self.robot_env or "gazebo")
        )
        cca_agent = CentralControllerAgent(
            "cca_preview@localhost",
            "none",
            name="cca_preview",
            resource_agents=resources,
            safety_file=str(safety_path),
        )
        safety_logic = getattr(cca_agent, "safety_logic", None)
        if safety_logic is None:
            raise RuntimeError("failed to initialize SafetyLogic for preview generation")

        try:
            async def _run_preview() -> dict[str, str]:
                await safety_logic.build_safety_rules_and_logic(
                    safety_text,
                    refinement_feedback=str(refinement_feedback or "").strip(),
                    previous_preview_rules=prompt_preview_rules,
                )
                await safety_logic.build_preview_interpretations()
                await asyncio.to_thread(safety_logic.save, preview_dir / "cca_safety_logic.json")
                return await asyncio.to_thread(safety_logic.build_dfas_per_rule, preview_dir)

            asyncio.run(_run_preview())
        except Exception as exc:
            self._record_safety_preview_failure(
                safety_key,
                str(exc),
                safety_sha256=safety_hash,
                refinement_feedback=str(refinement_feedback or "").strip(),
                parent_preview_id=str(parent_record.get("preview_id", "")).strip(),
            )
            shutil.rmtree(preview_dir, ignore_errors=True)
            raise

        self._clear_safety_preview_failure(safety_key)

        dot_files = sorted(str(p.resolve()) for p in preview_dir.glob("SAFE_*_dfa.dot"))
        png_files = sorted(str(p.resolve()) for p in preview_dir.glob("SAFE_*_dfa.png"))
        record = {
            "preview_id": preview_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "safety_file": safety_key,
            "safety_sha256": safety_hash,
            "tools_sha256": safety_hashes["tools_sha256"],
            "prompts_sha256": safety_hashes["prompts_sha256"],
            "refinement_feedback": str(refinement_feedback or "").strip(),
            "parent_preview_id": str(parent_record.get("preview_id", "")).strip(),
            "preview_dir": str(preview_dir.resolve()),
            "safety_logic_json": str((preview_dir / "cca_safety_logic.json").resolve()),
            "dfa_dot_files": dot_files,
            "dfa_png_files": png_files,
            "rules_count": len(getattr(safety_logic, "rules", []) or []),
            "prompt_context_rules_count": len(prompt_preview_rules),
        }

        history.insert(0, record)
        previews[safety_key] = history[:_SAFETY_PREVIEW_HISTORY_LIMIT]
        payload["previews"] = previews
        self._save_safety_intent_previews(payload)

        return self.get_safety_rule_preview(safety_key)

    def delete_safety_intent_previews(self, safety_requirement_file: str) -> None:
        raw = str(safety_requirement_file or "").strip()
        if not raw:
            return
        safety_path = self._abs_project_path(raw).resolve()
        safety_key = self._norm_path(safety_path)
        self._clear_safety_preview_failure(safety_key)
        payload = self._load_safety_intent_previews()
        previews = payload.get("previews", {}) if isinstance(payload, dict) else {}
        if not isinstance(previews, dict):
            previews = {}
        previews.pop(safety_key, None)
        payload["previews"] = previews
        self._save_safety_intent_previews(payload)

    def delete_safety_intent_state(self, safety_requirement_file: str) -> None:
        raw = str(safety_requirement_file or "").strip()
        if not raw:
            return
        safety_path = self._abs_project_path(raw).resolve()
        safety_key = self._norm_path(safety_path)
        self._clear_safety_preview_failure(safety_key)
        self.delete_safety_intent_previews(safety_key)

        payload = self._load_safety_intent_approvals()
        approvals = payload.get("approvals", {}) if isinstance(payload, dict) else {}
        if not isinstance(approvals, dict):
            approvals = {}
        approvals.pop(safety_key, None)
        payload["approvals"] = approvals
        self._save_safety_intent_approvals(payload)

    def evaluate_safety_intent_approval(self, safety_requirement_file: str) -> dict[str, Any]:
        raw = str(safety_requirement_file or "").strip()
        if not raw:
            return {"approved": False, "reason": "safety_file_missing", "record": {}, "safety_file": ""}

        safety_path = self._abs_project_path(raw).resolve()
        if not safety_path.exists():
            return {
                "approved": False,
                "reason": "safety_file_missing",
                "record": {},
                "safety_file": str(safety_path),
            }

        safety_text = safety_path.read_text(encoding="utf-8").strip()
        if not safety_text:
            return {
                "approved": False,
                "reason": "safety_file_empty",
                "record": {},
                "safety_file": str(safety_path),
            }
        current_hashes = self._compute_safety_generation_hashes(safety_path)
        current_hash = current_hashes["safety_sha256"]
        safety_key = self._norm_path(safety_path)

        payload = self._load_safety_intent_approvals()
        approvals = payload.get("approvals", {})
        record = approvals.get(safety_key, {}) if isinstance(approvals, dict) else {}
        if not isinstance(record, dict):
            record = {}

        if not record:
            return {
                "approved": False,
                "reason": "not_approved",
                "record": {},
                "safety_file": safety_key,
                "current_hash": current_hash,
            }

        if not bool(record.get("approved", False)):
            return {
                "approved": False,
                "reason": "revoked",
                "record": record,
                "safety_file": safety_key,
                "current_hash": current_hash,
            }

        approved_hash = str(record.get("safety_sha256", "")).strip()
        if approved_hash != current_hash:
            return {
                "approved": False,
                "reason": "content_changed_since_approval",
                "record": record,
                "safety_file": safety_key,
                "current_hash": current_hash,
            }

        descriptor, reason = self._build_precomputed_safety_descriptor(
            safety_path,
            record,
            current_hashes,
        )
        if descriptor is None:
            return {
                "approved": False,
                "reason": reason,
                "record": record,
                "safety_file": safety_key,
                "current_hash": current_hash,
            }

        return {
            "approved": True,
            "reason": "approved",
            "record": record,
            "safety_file": safety_key,
            "current_hash": current_hash,
            "precomputed_safety_artifacts": descriptor,
        }

    def approve_safety_intent(self, safety_requirement_file: str, note: str = "") -> dict[str, Any]:
        raw = str(safety_requirement_file or "").strip()
        if not raw:
            raise ValueError("safety requirement file is required")

        safety_path = self._abs_project_path(raw).resolve()
        if not safety_path.exists():
            raise FileNotFoundError(f"safety requirement file missing: {safety_path}")

        safety_text = safety_path.read_text(encoding="utf-8").strip()
        if not safety_text:
            raise ValueError(f"safety requirement file is empty: {safety_path}")

        preview = self.get_safety_rule_preview(str(safety_path))
        if not bool(preview.get("available", False)):
            raise ValueError(
                "generate safety rule preview first before approving intent"
            )
        if not bool(preview.get("hash_matches_current", False)):
            raise ValueError(
                "safety file changed after preview generation; regenerate preview before approval"
            )
        current_hashes = self._compute_safety_generation_hashes(safety_path)
        preview_record = preview.get("record", {}) if isinstance(preview.get("record"), dict) else {}
        preview_id = str(preview_record.get("preview_id", "") or "").strip()
        preview_tools_hash = str(preview_record.get("tools_sha256", "") or "").strip()
        preview_prompts_hash = str(preview_record.get("prompts_sha256", "") or "").strip()
        if not preview_id or not preview_tools_hash or not preview_prompts_hash:
            raise ValueError(
                "approved preview metadata is incomplete; regenerate preview before approval"
            )
        if preview_tools_hash != current_hashes["tools_sha256"]:
            raise ValueError(
                "tools catalog changed after preview generation; regenerate preview before approval"
            )
        if preview_prompts_hash != current_hashes["prompts_sha256"]:
            raise ValueError(
                "prompts.py changed after preview generation; regenerate preview before approval"
            )

        safety_key = self._norm_path(safety_path)
        now_utc = datetime.now(timezone.utc).isoformat()
        record = {
            "approved": True,
            "approved_at_utc": now_utc,
            "safety_sha256": current_hashes["safety_sha256"],
            "tools_sha256": current_hashes["tools_sha256"],
            "prompts_sha256": current_hashes["prompts_sha256"],
            "note": str(note or "").strip(),
            "preview_id": preview_id,
            "preview_generated_at_utc": str(preview_record.get("generated_at_utc", "")).strip(),
        }

        payload = self._load_safety_intent_approvals()
        approvals = payload.get("approvals", {})
        if not isinstance(approvals, dict):
            approvals = {}
        approvals[safety_key] = record
        payload["approvals"] = approvals
        self._save_safety_intent_approvals(payload)
        saved_payload = self._load_safety_intent_approvals()
        saved_record = (
            saved_payload.get("approvals", {}).get(safety_key, {})
            if isinstance(saved_payload.get("approvals", {}), dict)
            else {}
        )
        return {
            "approved": True,
            "record": saved_record if isinstance(saved_record, dict) else record,
            "safety_file": safety_key,
        }

    def revoke_safety_intent_approval(self, safety_requirement_file: str, note: str = "") -> dict[str, Any]:
        raw = str(safety_requirement_file or "").strip()
        if not raw:
            raise ValueError("safety requirement file is required")

        safety_path = self._abs_project_path(raw).resolve()
        safety_key = self._norm_path(safety_path)
        existing = self.evaluate_safety_intent_approval(safety_key)
        prior_record = dict(existing.get("record") or {})

        if safety_path.exists():
            safety_text = safety_path.read_text(encoding="utf-8").strip()
            safety_hash = sha256_text(safety_text) if safety_text else ""
        else:
            safety_hash = str(prior_record.get("safety_sha256", "")).strip()

        record = {
            **prior_record,
            "approved": False,
            "revoked_at_utc": datetime.now(timezone.utc).isoformat(),
            "safety_sha256": safety_hash,
            "note": str(note or "").strip(),
        }
        record.pop("verified_file", None)

        payload = self._load_safety_intent_approvals()
        approvals = payload.get("approvals", {})
        if not isinstance(approvals, dict):
            approvals = {}
        approvals[safety_key] = record
        payload["approvals"] = approvals
        self._save_safety_intent_approvals(payload)
        saved_payload = self._load_safety_intent_approvals()
        saved_record = (
            saved_payload.get("approvals", {}).get(safety_key, {})
            if isinstance(saved_payload.get("approvals", {}), dict)
            else {}
        )
        return {
            "approved": False,
            "record": saved_record if isinstance(saved_record, dict) else record,
            "safety_file": safety_key,
        }

    def list_safety_requirement_files(self, *, approved_only: bool = False) -> list[str]:
        files: list[str] = []
        if _SAFETY_REQUIREMENTS_DIR.exists():
            files.extend(self._norm_path(p) for p in sorted(_SAFETY_REQUIREMENTS_DIR.glob("*.txt")))
        if not approved_only:
            return files
        approved_files: list[str] = []
        for path in files:
            try:
                eval_out = self.evaluate_safety_intent_approval(path)
            except Exception:
                continue
            if bool(eval_out.get("approved", False)):
                approved_files.append(path)
        return approved_files

    def _resolve_product_context(
        self,
        product_spec_file: str,
        *,
        include_hashes: bool = True,
    ) -> dict[str, Any]:
        p = Path(str(product_spec_file))
        if not p.exists():
            raise FileNotFoundError(f"product init file missing: {p}")
        raw = self.load_config(str(p))
        product_name, product_meta = self._first_manifest_entry(raw)
        product_spec_path_raw = str(product_meta.get("product_specification_file", "")).strip()

        cca_raw = self.load_config(str(_CCA_INIT))
        if "cca" in cca_raw and isinstance(cca_raw["cca"], dict):
            cca_meta = cca_raw["cca"]
        else:
            _, cca_meta = self._first_manifest_entry(cca_raw)
        safety_path = str(cca_meta.get("safety_file", "")).strip()
        if not safety_path:
            safety_path = str(product_meta.get("safety_file", "")).strip()

        req_file_str = ""
        if product_spec_path_raw:
            req_file_str = str(self._abs_project_path(product_spec_path_raw).resolve())
        elif product_name:
            req_file_str = str(self._default_product_requirement_path(product_name))
        safe_file_str = ""
        if safety_path:
            safe_file_str = str(self._abs_project_path(safety_path).resolve())
        source_hashes = {}
        if include_hashes and req_file_str and safe_file_str and Path(req_file_str).exists():
            source_hashes = self._compute_source_hashes(
                Path(req_file_str), Path(safe_file_str)
            )
        return {
            "product_name": product_name,
            "product_spec_file": req_file_str,
            "safety_file": safe_file_str,
            "product_init_file": str(p.resolve()),
            "source_hashes": source_hashes,
        }

    def list_bundles(self) -> list[dict[str, Any]]:
        active_id = self.bundle_store.get_active_bundle_id()
        rows = []
        for row in self.bundle_store.list_bundles():
            out = dict(row)
            out["active"] = bool(active_id and str(row.get("bundle_id")) == str(active_id))
            rows.append(out)
        return rows

    def get_bundle_source_files(self, product_spec_file: str) -> dict[str, Any]:
        """
        Resolve which requirement/safety files will be used for offline bundle generation.
        """
        ctx = self._resolve_product_context(product_spec_file, include_hashes=False)
        req_file = Path(str(ctx.get("product_spec_file", "")))
        safety_file = Path(str(ctx.get("safety_file", "")))
        if not req_file.is_absolute():
            req_file = _PROJECT_ROOT / req_file
        if not safety_file.is_absolute():
            safety_file = _PROJECT_ROOT / safety_file
        return {
            "product_name": ctx.get("product_name", ""),
            "product_requirement_file": str(req_file.resolve()),
            "safety_requirement_file": str(safety_file.resolve()),
        }

    def get_active_bundle(self) -> dict[str, Any] | None:
        active_id = self.bundle_store.get_active_bundle_id()
        if not active_id:
            return None
        summary = self.bundle_store.get_bundle_summary(active_id)
        manifest = self.bundle_store.load_manifest(active_id)
        if not summary and not manifest:
            return None
        return {
            "bundle_id": active_id,
            "summary": summary,
            "manifest": manifest,
        }

    @staticmethod
    def _plan_validation_artifact_rel(artifacts: dict[str, Any] | None) -> str:
        if not isinstance(artifacts, dict):
            return ""
        for key in ("plan_validation_json", "offline_validation_json"):
            value = str(artifacts.get(key, "")).strip()
            if value:
                return value
        return ""

    def set_active_bundle(self, bundle_id: str | None) -> None:
        if bundle_id is None:
            self.bundle_store.set_active_bundle_id(None)
            return
        bid = str(bundle_id).strip()
        if not bid:
            self.bundle_store.set_active_bundle_id(None)
            return
        summary = self.bundle_store.get_bundle_summary(bid)
        manifest = self.bundle_store.load_manifest(bid)
        status = str((manifest or {}).get("status") or (summary or {}).get("status") or "")
        if status != BUNDLE_STATUS_VERIFIED:
            raise ValueError(f"plan set {bid} is not verified (status={status or 'unknown'})")
        self.bundle_store.set_active_bundle_id(bid)

    def _bundle_missing_linked_files(self, manifest: dict[str, Any] | None) -> list[str]:
        data = manifest if isinstance(manifest, dict) else {}
        if not data:
            return ["manifest_missing"]

        missing: list[str] = []
        refs = (
            ("product_requirement_file_missing", str(data.get("product_spec_file", "")).strip()),
            ("safety_requirement_file_missing", str(data.get("safety_file", "")).strip()),
        )
        for reason, raw_path in refs:
            if not raw_path:
                missing.append(reason)
                continue
            candidate = self._abs_project_path(raw_path).resolve()
            if not candidate.exists():
                missing.append(reason)
        return missing

    def get_bundle_delete_policy(self, bundle_id: str) -> dict[str, Any]:
        bid = str(bundle_id or "").strip()
        if not bid:
            return {
                "bundle_id": "",
                "exists": False,
                "can_delete": False,
                "status": "unknown",
                "missing_links": [],
                "reason": "plan set id is missing",
            }

        summary = self.bundle_store.get_bundle_summary(bid) or {}
        manifest = self.bundle_store.load_manifest(bid) or {}
        bundle_dir = self.bundle_store.bundle_dir(bid)
        if not summary and not manifest and not bundle_dir.exists():
            return {
                "bundle_id": bid,
                "exists": False,
                "can_delete": False,
                "status": "unknown",
                "missing_links": [],
                "reason": "plan set not found",
            }

        status = str(manifest.get("status") or summary.get("status") or "").strip().lower() or "unknown"
        missing_links = self._bundle_missing_linked_files(manifest)
        manifest_missing = "manifest_missing" in missing_links
        can_delete = status != BUNDLE_STATUS_VERIFIED or manifest_missing or bool(
            [reason for reason in missing_links if reason != "manifest_missing"]
        )
        if can_delete:
            reason = "ok"
        elif status == BUNDLE_STATUS_VERIFIED:
            reason = "verified_plan_set_requires_unverify"
        else:
            reason = "ok"
        return {
            "bundle_id": bid,
            "exists": True,
            "can_delete": can_delete,
            "status": status,
            "missing_links": missing_links,
            "reason": reason,
        }

    def check_bundle_compatibility(
        self,
        bundle_id: str,
        product_spec_file: str,
        execution_mode: str,
        robot_env: str,
        safety_requirement_file: str | None = None,
    ) -> tuple[bool, list[str]]:
        bid = str(bundle_id).strip()
        if not bid:
            return False, ["bundle_id_missing"]

        manifest = self.bundle_store.load_manifest(bid)
        if not manifest:
            return False, ["manifest_missing"]

        reasons: list[str] = []
        status = str(manifest.get("status", "")).strip()
        if status != BUNDLE_STATUS_VERIFIED:
            reasons.append("bundle_not_verified")

        try:
            req_file, default_safety = self._resolve_requirement_input(product_spec_file)
        except Exception as exc:
            return False, [f"context_error:{exc}"]

        manifest_req_file = str(manifest.get("product_spec_file", "")).strip()
        if self._norm_path(manifest_req_file) != self._norm_path(req_file):
            reasons.append("product_spec_file")

        if str(manifest.get("execution_mode", "")) != str(execution_mode):
            reasons.append("execution_mode")
        if str(manifest.get("robot_env", "")) != str(robot_env):
            reasons.append("robot_env")

        manifest_safety = str(manifest.get("safety_file", "")).strip()
        selected_safety = str(safety_requirement_file or "").strip()
        if selected_safety.upper() == "__NONE__":
            reasons.append("safety_file_none")
            ok = len(reasons) == 0
            return ok, reasons
        if selected_safety:
            selected_safety_norm = self._norm_path(self._abs_project_path(selected_safety))
            if manifest_safety and self._norm_path(manifest_safety) != selected_safety_norm:
                reasons.append("safety_file")
            safety_for_hash = Path(selected_safety_norm)
        elif manifest_safety:
            safety_for_hash = Path(self._norm_path(self._abs_project_path(manifest_safety)))
        elif default_safety:
            safety_for_hash = Path(self._norm_path(self._abs_project_path(default_safety)))
        else:
            return False, ["context_error:missing safety_file reference"]

        try:
            expected_hashes = self._compute_source_hashes(
                Path(self._norm_path(req_file)),
                safety_for_hash,
            )
        except Exception as exc:
            return False, [f"context_error:{exc}"]

        got_hashes = manifest.get("source_hashes", {}) if isinstance(manifest.get("source_hashes"), dict) else {}
        tools_snapshot_ok = self._bundle_tools_snapshot_matches_manifest(bid, manifest)
        for key, expected in expected_hashes.items():
            # Prompt text changes affect future offline generation, but a
            # user-verified bundle should still be reusable at startup.
            if key == "prompts_sha256":
                continue
            if key == "tools_sha256":
                if tools_snapshot_ok is False:
                    reasons.append(key)
                continue
            if str(got_hashes.get(key, "")) != str(expected):
                reasons.append(key)

        ok = len(reasons) == 0
        return ok, reasons

    def _bundle_tools_snapshot_matches_manifest(
        self,
        bundle_id: str,
        manifest: dict[str, Any] | None = None,
    ) -> bool | None:
        """Return True when a bundled tools snapshot exists and matches the manifest hash."""
        bid = str(bundle_id or "").strip()
        if not bid:
            return None
        data = manifest if isinstance(manifest, dict) else self.bundle_store.load_manifest(bid)
        if not isinstance(data, dict):
            return None
        artifacts = data.get("artifacts", {}) if isinstance(data.get("artifacts"), dict) else {}
        rel = str(artifacts.get("tools_json", "")).strip()
        if not rel:
            return None

        snapshot_path = (self.bundle_store.bundle_dir(bid) / rel).resolve()
        if not snapshot_path.exists():
            return False

        got_hashes = data.get("source_hashes", {}) if isinstance(data.get("source_hashes"), dict) else {}
        expected = str(got_hashes.get("tools_sha256", "")).strip()
        if not expected:
            return False
        return sha256_file(snapshot_path) == expected

    @staticmethod
    def _describe_bundle_incompatibility_reasons(reasons: list[str]) -> str:
        labels = {
            "bundle_id_missing": "active plan set id is missing",
            "manifest_missing": "plan set manifest is missing",
            "bundle_not_verified": "plan set is not verified",
            "product_spec_file": "selected product requirement file changed",
            "execution_mode": "execution mode changed",
            "robot_env": "robot environment changed",
            "safety_file_none": "safety was disabled",
            "safety_file": "selected safety file changed",
            "requirements_sha256": "product requirement file contents changed",
            "safety_sha256": "safety file contents changed",
            "tools_sha256": "bundled tools snapshot changed or is missing",
            "prompts_sha256": "prompts.py changed",
        }
        details: list[str] = []
        for reason in reasons:
            raw = str(reason or "").strip()
            if not raw:
                continue
            if raw.startswith("context_error:"):
                details.append(raw.split(":", 1)[1].strip() or "startup context changed")
                continue
            details.append(labels.get(raw, raw.replace("_", " ")))
        seen: set[str] = set()
        ordered = []
        for item in details:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ", ".join(ordered) if ordered else "current selection no longer matches the plan set"

    def _resolve_startup_bundle_context(
        self,
        *,
        product_spec_file: str,
        execution_mode: str,
        robot_env: str,
        safety_requirement_file: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        active_id = self.bundle_store.get_active_bundle_id()
        if not active_id:
            return None, None

        ok, reasons = self.check_bundle_compatibility(
            active_id,
            product_spec_file,
            execution_mode,
            robot_env,
            safety_requirement_file,
        )
        if not ok:
            detail = self._describe_bundle_incompatibility_reasons(reasons)
            self.bundle_store.set_active_bundle_id(None)
            notice = (
                f"System started without verified plan set '{active_id}': it was deactivated "
                f"because {detail}."
            )
            self._diag_emit(f"[Bundle] Auto-deactivated incompatible active bundle {active_id}: {detail}")
            log.warning("Auto-deactivated incompatible active plan set %s: %s", active_id, detail)
            return None, notice

        return (
            self._resolve_active_bundle_context(
                product_spec_file=product_spec_file,
                execution_mode=execution_mode,
                robot_env=robot_env,
                safety_requirement_file=safety_requirement_file,
            ),
            None,
        )

    @staticmethod
    def _startup_bundle_scope_input(
        selected_product_file: str,
        selected_requirement_file: str | None = None,
    ) -> str:
        """Prefer the explicit requirement override when matching startup bundles."""
        requirement_file = str(selected_requirement_file or "").strip()
        if requirement_file:
            return requirement_file
        return str(selected_product_file or "").strip()

    def list_compatible_bundles(
        self,
        product_spec_file: str,
        execution_mode: str,
        robot_env: str,
        safety_requirement_file: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.list_bundles():
            bid = str(row.get("bundle_id", "")).strip()
            if not bid:
                continue
            ok, reasons = self.check_bundle_compatibility(
                bid,
                product_spec_file,
                execution_mode,
                robot_env,
                safety_requirement_file,
            )
            if not ok:
                continue
            out = dict(row)
            out["compatibility_ok"] = True
            out["compatibility_reasons"] = []
            rows.append(out)
        return rows

    def evaluate_bundle(
        self,
        bundle_id: str,
        product_spec_file: str,
        execution_mode: str,
        robot_env: str,
        safety_requirement_file: str | None = None,
    ) -> dict[str, Any]:
        bid = str(bundle_id or "").strip()
        if not bid:
            return {"ok": False, "reasons": ["bundle_id_missing"], "status": BUNDLE_STATUS_INVALID}
        summary = self.bundle_store.get_bundle_summary(bid) or {}
        manifest = self.bundle_store.load_manifest(bid) or {}
        ok, reasons = self.check_bundle_compatibility(
            bid,
            product_spec_file,
            execution_mode,
            robot_env,
            safety_requirement_file,
        )
        status = str(manifest.get("status") or summary.get("status") or "")
        if status == BUNDLE_STATUS_VERIFIED and not ok:
            status = BUNDLE_STATUS_STALE
        return {
            "ok": ok,
            "reasons": reasons,
            "status": status or BUNDLE_STATUS_INVALID,
            "summary": summary,
            "manifest": manifest,
        }

    def evaluate_bundle_for_files(
        self,
        bundle_id: str,
        product_requirement_file: str,
        safety_requirement_file: str | None,
        execution_mode: str,
        robot_env: str,
    ) -> dict[str, Any]:
        return self.evaluate_bundle(
            bundle_id=bundle_id,
            product_spec_file=product_requirement_file,
            execution_mode=execution_mode,
            robot_env=robot_env,
            safety_requirement_file=safety_requirement_file,
        )

    @staticmethod
    def _ensure_called_from_worker_thread(method_name: str) -> None:
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                f"{method_name}() must be called from a worker thread "
                "(use asyncio.to_thread in UI callbacks)."
            )
        except RuntimeError as exc:
            if "no running event loop" not in str(exc).lower():
                raise

    def generate_verified_bundle(
        self,
        product_spec_file: str | None = None,
        execution_mode: str | None = None,
        robot_env: str | None = None,
        *,
        product_requirement_file: str | None = None,
        safety_requirement_file: str | None = None,
        auto_replan_max_attempts: int | None = None,
        refinement_feedback: str = "",
        parent_bundle_id: str = "",
    ) -> dict[str, Any]:
        if self.system_running or self._starting or self._stopping:
            raise RuntimeError("cannot generate plan set while system lifecycle is active")

        self._ensure_called_from_worker_thread("generate_verified_bundle")

        resolved_execution_mode = str(execution_mode or self.execution_mode or "simulation").strip()
        resolved_robot_env = str(robot_env or self.robot_env or "gazebo").strip()

        req_file = str(product_requirement_file or "").strip()
        input_spec = str(product_spec_file or "").strip()
        if not req_file and input_spec and Path(input_spec).suffix.lower() != ".json":
            req_file = input_spec

        if req_file:
            product_ctx = self._resolve_product_init_for_requirement(req_file)
            product_init_file = str(product_ctx["product_init_file"])
            req_file = self._norm_path(self._abs_project_path(req_file))
        elif input_spec:
            p_input = Path(input_spec)
            if p_input.suffix.lower() == ".json":
                product_init_file = self._norm_path(self._abs_project_path(p_input))
                ctx = self._resolve_product_context(product_init_file, include_hashes=False)
                req_file = str(ctx.get("product_spec_file", ""))
            else:
                product_ctx = self._resolve_product_init_for_requirement(input_spec)
                product_init_file = str(product_ctx["product_init_file"])
                req_file = self._norm_path(self._abs_project_path(input_spec))
        else:
            raise ValueError("product specification input is required")

        product_ctx = self._resolve_product_context(product_init_file, include_hashes=False)
        safety_override = (
            self._norm_path(self._abs_project_path(safety_requirement_file))
            if str(safety_requirement_file or "").strip()
            else None
        )
        selected_safety_file = str(
            safety_override or product_ctx.get("safety_file") or ""
        ).strip() or None
        precomputed_safety_artifacts: dict[str, Any] | None = None
        if selected_safety_file:
            safety_eval = self.evaluate_safety_intent_approval(selected_safety_file)
            record = safety_eval.get("record", {}) if isinstance(safety_eval.get("record"), dict) else {}
            if bool(safety_eval.get("approved", False)):
                raw_precomputed = safety_eval.get("precomputed_safety_artifacts")
                if not isinstance(raw_precomputed, dict) or not raw_precomputed:
                    raise ValueError(
                        "approved safety preview could not be resolved; regenerate preview and approve it again"
                    )
                precomputed_safety_artifacts = dict(raw_precomputed)
            elif self._approval_requires_preview_refresh(
                str(safety_eval.get("reason", "") or ""),
                record,
            ):
                raise ValueError(
                    self._approval_refresh_error(str(safety_eval.get("reason", "") or ""))
                )
        resolved_auto_replan_max_attempts = 3 if auto_replan_max_attempts is None else auto_replan_max_attempts

        return asyncio.run(
            self.bundle_compiler.compile_bundle(
                product_init_file=product_init_file,
                execution_mode=resolved_execution_mode,
                robot_env=resolved_robot_env,
                product_requirement_file=req_file or None,
                safety_requirement_file=safety_override,
                precomputed_safety_artifacts=precomputed_safety_artifacts,
                auto_replan_max_attempts=resolved_auto_replan_max_attempts,
                refinement_feedback=str(refinement_feedback or "").strip(),
                parent_bundle_id=str(parent_bundle_id or "").strip(),
            )
        )

    def verify_bundle(self, bundle_id: str) -> dict[str, Any]:
        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("plan_set_id is required")

        manifest = self.bundle_store.load_manifest(bid)
        if not manifest:
            raise ValueError(f"plan-set manifest not found: {bid}")

        manifest["status"] = BUNDLE_STATUS_VERIFIED
        manifest["verified"] = True
        manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.bundle_store.overwrite_manifest(bid, manifest)

        summary = self.bundle_store.update_bundle_summary(
            bid,
            {
                "status": BUNDLE_STATUS_VERIFIED,
                "verified": True,
            },
        )
        return {"bundle_id": bid, "summary": summary, "manifest": manifest}

    def unverify_bundle(self, bundle_id: str) -> dict[str, Any]:
        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("plan_set_id is required")

        manifest = self.bundle_store.load_manifest(bid)
        if not manifest:
            raise ValueError(f"plan-set manifest not found: {bid}")

        current_status = str(manifest.get("status", "")).strip().lower()
        if current_status == BUNDLE_STATUS_INVALID:
            raise ValueError("invalid plan set cannot be unverified")

        manifest["status"] = BUNDLE_STATUS_DRAFT
        manifest["verified"] = False
        manifest["unverified_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.bundle_store.overwrite_manifest(bid, manifest)

        summary = self.bundle_store.update_bundle_summary(
            bid,
            {
                "status": BUNDLE_STATUS_DRAFT,
                "verified": False,
            },
        )
        if self.bundle_store.get_active_bundle_id() == bid:
            self.bundle_store.set_active_bundle_id(None)
        return {"bundle_id": bid, "summary": summary, "manifest": manifest}

    def delete_bundle(self, bundle_id: str) -> dict[str, Any]:
        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("plan_set_id is required")
        if self.system_running or self._starting or self._stopping:
            raise RuntimeError("cannot delete plan set while system lifecycle is active")

        summary = self.bundle_store.get_bundle_summary(bid) or {}
        manifest = self.bundle_store.load_manifest(bid) or {}
        bundle_dir = self.bundle_store.bundle_dir(bid)
        if not summary and not manifest and not bundle_dir.exists():
            raise ValueError(f"plan-set not found: {bid}")

        status = str(manifest.get("status") or summary.get("status") or "").strip().lower()
        delete_policy = self.get_bundle_delete_policy(bid)
        if not delete_policy.get("can_delete", False):
            raise ValueError("verified plan set cannot be deleted; unverify it first")

        bundle_dir_existed = bundle_dir.exists()
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir, ignore_errors=True)

        removed = self.bundle_store.delete_bundle_summary(bid) or bundle_dir_existed or bool(summary) or bool(manifest)
        if self.bundle_store.get_active_bundle_id() == bid:
            self.bundle_store.set_active_bundle_id(None)

        return {
            "bundle_id": bid,
            "removed": bool(removed),
            "status": status or "unknown",
        }

    def get_bundle_artifacts(self, bundle_id: str) -> dict[str, Any]:
        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("plan_set_id is required")

        manifest = self.bundle_store.load_manifest(bid)
        if not manifest:
            raise ValueError(f"plan-set manifest not found: {bid}")
        root = self.bundle_store.bundle_dir(bid)
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}

        def _artifact_path(key: str) -> Path | None:
            rel = artifacts.get(key)
            if not rel:
                return None
            return (root / str(rel)).resolve()

        plan_path = _artifact_path("plan_json")
        fsa_path = _artifact_path("global_fsa_json")
        validation_rel = self._plan_validation_artifact_rel(artifacts)
        validation_path = (root / validation_rel).resolve() if validation_rel else None

        plan_payload = self._read_json_dict(plan_path) if plan_path else {}
        fsa_payload = self._read_json_dict(fsa_path) if fsa_path else {}
        validation_payload = self._read_json_dict(validation_path) if validation_path else {}

        return {
            "bundle_id": bid,
            "manifest": manifest,
            "summary": self.bundle_store.get_bundle_summary(bid) or {},
            "plan_nodes": plan_payload.get("nodes", []) if isinstance(plan_payload.get("nodes", []), list) else [],
            "global_fsa": fsa_payload,
            "validation": validation_payload,
            "paths": {
                "root": str(root),
                "plan_json": str(plan_path) if plan_path else "",
                "global_fsa_json": str(fsa_path) if fsa_path else "",
                "plan_validation_json": str(validation_path) if validation_path else "",
                "offline_validation_json": str(validation_path) if validation_path else "",
            },
        }

    def get_bundle_safety_rules(self, bundle_id: str) -> list[dict[str, Any]]:
        """Load compiled safety rules from a stored plan set without starting the system."""
        bid = str(bundle_id or "").strip()
        if not bid:
            return []
        manifest = self.bundle_store.load_manifest(bid)
        if not manifest:
            return []
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
        logic_rel = str(artifacts.get("safety_logic_json", "")).strip()
        if not logic_rel:
            return []
        logic_path = (self.bundle_store.bundle_dir(bid) / logic_rel).resolve()
        logic_payload = self._read_json_dict(logic_path)
        rules = logic_payload.get("rules", [])
        if not isinstance(rules, list):
            return []
        normalized: list[dict[str, Any]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            item = dict(rule)
            interpretation = str(item.get("generated_interpretation", "")).strip()
            if not interpretation:
                interpretation = self._ltlf_plain_feedback(item)
            item["generated_interpretation"] = interpretation
            normalized.append(item)
        return normalized

    def get_bundle_plan_nodes(self, bundle_id: str) -> list[dict[str, Any]]:
        """Load DAG nodes from a stored plan set without starting the system."""
        bid = str(bundle_id or "").strip()
        if not bid:
            return []
        manifest = self.bundle_store.load_manifest(bid)
        if not manifest:
            return []
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
        plan_rel = str(artifacts.get("plan_json", "")).strip()
        if not plan_rel:
            return []
        plan_path = (self.bundle_store.bundle_dir(bid) / plan_rel).resolve()
        plan_payload = self._read_json_dict(plan_path)
        nodes = plan_payload.get("nodes", [])
        return nodes if isinstance(nodes, list) else []

    def apply_bundle_chat_edit(self, bundle_id: str, user_message: str) -> dict[str, Any]:
        if self.system_running or self._starting or self._stopping:
            raise RuntimeError("stop the system before editing plan sets")
        self._ensure_called_from_worker_thread("apply_bundle_chat_edit")

        bid = str(bundle_id or "").strip()
        if not bid:
            raise ValueError("plan_set_id is required")
        message = str(user_message or "").strip()
        if not message:
            raise ValueError("message is empty")

        manifest = self.bundle_store.load_manifest(bid)
        if not manifest:
            raise ValueError(f"plan-set manifest not found: {bid}")
        current_status = str(manifest.get("status", "")).strip().lower()
        if current_status == BUNDLE_STATUS_VERIFIED:
            raise ValueError("verified plan set is locked; unverify it before editing")
        root = self.bundle_store.bundle_dir(bid)
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}

        plan_rel = str(artifacts.get("plan_json", "")).strip()
        fsa_rel = str(artifacts.get("global_fsa_json", "")).strip()
        safety_logic_rel = str(artifacts.get("safety_logic_json", "")).strip()
        validation_rel = self._plan_validation_artifact_rel(artifacts)
        if not plan_rel or not fsa_rel or not safety_logic_rel or not validation_rel:
            raise ValueError("plan set is missing required plan/safety artifacts")

        plan_path = (root / plan_rel).resolve()
        fsa_path = (root / fsa_rel).resolve()
        safety_logic_path = (root / safety_logic_rel).resolve()
        validation_path = (root / validation_rel).resolve()
        if not plan_path.exists():
            raise FileNotFoundError(f"plan artifact missing: {plan_path}")
        if not safety_logic_path.exists():
            raise FileNotFoundError(f"safety logic artifact missing: {safety_logic_path}")

        safety_payload = self._read_json_dict(safety_logic_path)
        rules = safety_payload.get("rules", []) if isinstance(safety_payload.get("rules", []), list) else []
        if not rules:
            raise ValueError("safety logic has no rules")

        dfa_map: dict[str, str] = {}
        dot_rels = artifacts.get("safety_dfa_dot_files", [])
        if isinstance(dot_rels, list):
            for rel in dot_rels:
                dot_path = (root / str(rel)).resolve()
                if not dot_path.exists():
                    continue
                rid = dot_path.stem
                if rid.endswith("_dfa"):
                    rid = rid[: -len("_dfa")]
                try:
                    dfa_map[rid] = dot_path.read_text(encoding="utf-8")
                except Exception:
                    continue

        if not dfa_map:
            for rule in rules:
                rid = str(rule.get("id", "")).strip()
                if not rid:
                    continue
                dot_path = (root / "safety" / f"{rid}_dfa.dot").resolve()
                if dot_path.exists():
                    try:
                        dfa_map[rid] = dot_path.read_text(encoding="utf-8")
                    except Exception:
                        continue
        if not dfa_map:
            raise ValueError("plan set has no DFA DOT artifacts for safety validation")

        ProductAgent, _, PlanSafetyValidator, CameraModule, _ = (
            self.bundle_compiler._import_runtime_classes()
        )
        resources = self.bundle_compiler._collect_resource_refs(
            robot_env=str(manifest.get("robot_env", "gazebo") or "gazebo")
        )
        resource_jids = [str(r.jid) for r in resources]
        product_name = str(manifest.get("product_name", "product")).strip() or "product"
        product_spec_file = str(manifest.get("product_spec_file", "")).strip()
        safety_file = str(manifest.get("safety_file", "")).strip()
        product_jid = f"{product_name}@localhost"

        safety_text = ""
        if safety_file:
            safety_path = self._abs_project_path(safety_file).resolve()
            if safety_path.exists():
                safety_text = safety_path.read_text(encoding="utf-8").strip()

        product_agent = ProductAgent(
            product_jid,
            "none",
            name=product_name,
            resource_jids=resource_jids,
            resource_agents=resources,
            product_specification_file=product_spec_file,
            safety_file=safety_file or None,
            camera=CameraModule(backend="none"),
        )
        product_agent.process_planner.load(plan_path)
        product_agent.safety_text = safety_text

        try:
            task_nodes = [
                n for n in product_agent.process_planner.nodes
                if n.get("type") == "task"
            ]
            pred_map = {
                str(n.get("id")): list(n.get("predecessors", []))
                for n in task_nodes
                if n.get("id")
            }
            seed_violations = [
                {
                    "violated_rule_id": "USER_EDIT",
                    "violation_text": f"Operator requested plan changes: {message}",
                    "violation_logic": "manual_edit_request",
                    "witness_trace": [],
                    "relevant_tasks": task_nodes,
                    "relevant_pred_map": pred_map,
                }
            ]
            validator = PlanSafetyValidator(
                rules=rules,
                dfa_map=dfa_map,
                tools_catalog=getattr(product_agent, "tools_catalog", []),
            )
            replan_policy = manifest.get("replan_policy", {}) if isinstance(manifest.get("replan_policy"), dict) else {}
            try:
                auto_replan_max_attempts = int(replan_policy.get("auto_replan_max_attempts", 3) or 0)
            except Exception:
                auto_replan_max_attempts = 3
            auto_replan_max_attempts = max(0, min(auto_replan_max_attempts, 10))

            validation_payload = asyncio.run(
                self.bundle_compiler.run_offline_repair_loop(
                    product_agent=product_agent,
                    validator=validator,
                    product_jid=product_jid,
                    auto_replan_max_attempts=auto_replan_max_attempts,
                    seed_replan_violations=seed_violations,
                )
            )
            product_agent.process_planner.save(plan_path)
            product_agent.process_planner.save_global_fsa(fsa_path)

            ok = bool(validation_payload.get("ok", False))
            violated_rules = list(validation_payload.get("violated_rules", []))
            witness_count = int(validation_payload.get("witness_count", 0))
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            validation_path.write_text(
                json.dumps(validation_payload, indent=2),
                encoding="utf-8",
            )

            new_status = BUNDLE_STATUS_DRAFT if ok else BUNDLE_STATUS_INVALID
            manifest["status"] = new_status
            manifest["verified"] = False
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["replan_policy"] = {
                "auto_replan_max_attempts": auto_replan_max_attempts,
            }
            manifest["validation_summary"] = {
                "ok": bool(ok),
                "violated_rules": violated_rules,
                "witness_count": witness_count,
                "auto_replans_used": int(validation_payload.get("auto_replans_used", 0)),
                "stop_reason": str(validation_payload.get("stop_reason", "")),
            }
            self.bundle_store.overwrite_manifest(bid, manifest)
            summary = self.bundle_store.update_bundle_summary(
                bid,
                {
                    "status": new_status,
                    "verified": False,
                },
            )
            if self.bundle_store.get_active_bundle_id() == bid:
                self.bundle_store.set_active_bundle_id(None)

            return {
                "bundle_id": bid,
                "status": new_status,
                "validation_ok": bool(ok),
                "witness_count": witness_count,
                "auto_replans_used": int(validation_payload.get("auto_replans_used", 0)),
                "auto_replan_max_attempts": auto_replan_max_attempts,
                "stop_reason": str(validation_payload.get("stop_reason", "")),
                "summary": summary,
                "manifest": manifest,
                "task_count": len(
                    [n for n in product_agent.process_planner.nodes if n.get("type") == "task"]
                ),
            }
        finally:
            try:
                if hasattr(product_agent, "camera") and product_agent.camera:
                    product_agent.camera.destroy()
            except Exception:
                pass

    def _resolve_active_bundle_context(
        self,
        *,
        product_spec_file: str,
        execution_mode: str,
        robot_env: str,
        safety_requirement_file: str | None = None,
    ) -> dict[str, Any] | None:
        active_id = self.bundle_store.get_active_bundle_id()
        if not active_id:
            return None
        ok, reasons = self.check_bundle_compatibility(
            active_id,
            product_spec_file,
            execution_mode,
            robot_env,
            safety_requirement_file,
        )
        if not ok:
            msg = ", ".join(reasons) if reasons else "unknown mismatch"
            self._diag_emit(f"[Bundle] Compatibility check failed: {msg}")
            raise RuntimeError(
                f"Active plan set '{active_id}' is incompatible with current selection: {msg}"
            )

        manifest = self.bundle_store.load_manifest(active_id)
        if not manifest:
            raise RuntimeError(f"Active plan-set manifest missing: {active_id}")

        root = self.bundle_store.bundle_dir(active_id)
        raw_artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
        artifacts_abs: dict[str, Any] = {}
        for key, value in raw_artifacts.items():
            if isinstance(value, list):
                artifacts_abs[key] = [str((root / str(v)).resolve()) for v in value]
            else:
                artifacts_abs[key] = str((root / str(value)).resolve())

        return {
            "bundle_id": active_id,
            "product_name": manifest.get("product_name"),
            "product_spec_file": manifest.get("product_spec_file"),
            "safety_file": manifest.get("safety_file"),
            "execution_mode": manifest.get("execution_mode"),
            "robot_env": manifest.get("robot_env"),
            "status": manifest.get("status"),
            "replan_policy": (
                dict(manifest.get("replan_policy", {}))
                if isinstance(manifest.get("replan_policy"), dict)
                else {}
            ),
            "validation_summary": (
                dict(manifest.get("validation_summary", {}))
                if isinstance(manifest.get("validation_summary"), dict)
                else {}
            ),
            "manifest_path": str(self.bundle_store.manifest_path(active_id)),
            "artifacts": artifacts_abs,
        }

    # ------------------------------------------------------------------
    # XMPP server lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _tcp_port_open(host: str, port: int, timeout_sec: float = 0.25) -> bool:
        try:
            with socket.create_connection((host, int(port)), timeout=max(0.05, float(timeout_sec))):
                return True
        except Exception:
            return False

    async def _wait_for_xmpp_ready(self, timeout_sec: float = 45.0) -> None:
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        while time.monotonic() < deadline:
            proc = self._xmpp_proc
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"Embedded XMPP process exited early (rc={proc.returncode})")
            ready = await asyncio.to_thread(
                self._tcp_port_open,
                self._xmpp_host,
                self._xmpp_port,
                0.25,
            )
            if ready:
                return
            await asyncio.sleep(0.2)
        raise RuntimeError(
            f"Embedded XMPP server startup timed out after {timeout_sec:.0f}s "
            f"(host={self._xmpp_host} port={self._xmpp_port})"
        )

    async def _ensure_xmpp_server(self) -> None:
        """Ensure an embedded XMPP server is reachable on localhost:5222."""
        if self._xmpp_proc is not None and self._xmpp_proc.poll() is None:
            return
        # If an external XMPP is already up, reuse it.
        already_up = await asyncio.to_thread(self._tcp_port_open, self._xmpp_host, self._xmpp_port, 0.25)
        if already_up:
            self._diag_emit("xmpp already listening on localhost:5222 (reusing existing server)")
            return

        cmd = [sys.executable, "-m", "cais_spade_llm.ui.xmpp_server_runner"]
        self._xmpp_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        self._diag_emit(f"spawned xmpp runner pid={self._xmpp_proc.pid}")
        await self._wait_for_xmpp_ready(timeout_sec=45.0)
        log.info("Embedded XMPP server started on localhost:5222 (pid=%s)", self._xmpp_proc.pid)

    async def _stop_xmpp_server(self) -> None:
        proc = self._xmpp_proc
        self._xmpp_proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                return
        for _ in range(25):
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.1)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # System lifecycle
    # ------------------------------------------------------------------
    async def start_system(self) -> None:
        """Start the SPADE agent system (mirrors spade_main.spade_main)."""
        if self.system_running or self._starting:
            return
        self._starting = True
        self.last_error = None
        self.last_notice = None
        self._clear_cached_plan_safety_alerts()
        self._startup_seq += 1
        startup_id = self._startup_seq
        startup_t0 = time.monotonic()
        self._set_startup_phase("start_requested")
        self._diag_emit(f"startup#{startup_id} begin mode={self.execution_mode} env={self.robot_env}")

        try:
            # Archive previous monitor outputs.
            self._set_startup_phase("archive_monitors")
            await asyncio.to_thread(self._archive_monitors)
            self._diag_emit(
                f"startup#{startup_id} archive_monitors done in {time.monotonic() - startup_t0:.2f}s"
            )

            # Set environment variables that agent_creator reads.
            self._set_startup_phase("prepare_environment")
            os.environ["ROBOT_ENV"] = self.robot_env
            os.environ["EXECUTION_MODE"] = self.execution_mode
            perception_backend = self._perception_backend_for_mode()
            os.environ["PERCEPTION_BACKEND"] = perception_backend

            mode = str(self.execution_mode or "").strip().lower()
            prewarmed: dict[str, Any] = {}
            if mode == "simulation":
                self._set_startup_phase("simulation_readiness")
                sim_ready, sim_reason = await asyncio.to_thread(
                    self.simulation_start_ready,
                    True,
                )
                if not sim_ready:
                    raise RuntimeError(sim_reason)
                # Hand off prewarmed controllers to SPADE agents instead of
                # destroying them — avoids duplicate ROS2 init on first task.
                with self._gazebo_prewarm_lock:
                    prewarmed = dict(self._gazebo_prewarm_controllers)
                    self._gazebo_prewarm_controllers.clear()
                    self._gazebo_prewarm_pending.clear()
                # Drop any stale prewarmed controller whose spin thread died.
                invalid_keys: list[str] = []
                for key, ctrl in list(prewarmed.items()):
                    try:
                        usable = bool(getattr(ctrl, "is_usable", lambda: True)())
                    except Exception:
                        usable = False
                    if usable:
                        continue
                    invalid_keys.append(key)
                    prewarmed.pop(key, None)
                    try:
                        ctrl.shutdown()
                    except Exception:
                        pass
                if invalid_keys:
                    self._diag_emit(
                        "discarded stale prewarmed controllers: " + ",".join(sorted(invalid_keys))
                    )
                if prewarmed:
                    log.info(
                        "Handing off prewarmed controllers to agents: %s",
                        ",".join(sorted(prewarmed.keys())),
                    )
                else:
                    log.info("No prewarmed controllers available for handoff.")

            self._set_startup_phase("physical_readiness")
            ready, reason = self.physical_perception_ready()
            if not ready:
                raise RuntimeError(reason)

            # Ensure XMPP server is up.
            self._set_startup_phase("xmpp_server")
            await self._ensure_xmpp_server()

            # Import/configure agent_creator off the UI event loop.
            self._set_startup_phase("import_agent_creator")
            self._maybe_start_agent_creator_prefetch()
            ac = self._agent_creator_cached
            if ac is None:
                ac = await asyncio.to_thread(self._import_agent_creator_module)
                self._agent_creator_cached = ac
            self._set_startup_phase("configure_agent_creator_runtime")
            await asyncio.to_thread(
                self._configure_agent_creator_runtime,
                ac,
                self.robot_env,
                self.execution_mode,
                perception_backend,
            )
            from function_analyzer import FunctionAnalyzer

            # Collect init files.
            self._set_startup_phase("collect_init_files")
            prod_files, res_files = await asyncio.to_thread(self._collect_init_files)
            if not prod_files:
                raise RuntimeError("No product initialization files found.")

            selected_requirement_file = str(self.selected_requirement_file or "").strip()
            if selected_requirement_file:
                try:
                    selected_product_file = await asyncio.to_thread(
                        self.resolve_product_init_for_requirement,
                        selected_requirement_file,
                    )
                    self.selected_product = selected_product_file
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to resolve product init for requirement file '{selected_requirement_file}': {exc}"
                    ) from exc

            selected_product_file = str(self.selected_product or "").strip()
            if not selected_product_file:
                selected_product_file = prod_files[0]
                self.selected_product = selected_product_file
            selected_norm = self._norm_path(selected_product_file)
            matched_product_files = [p for p in prod_files if self._norm_path(p) == selected_norm]
            if matched_product_files:
                # Align runtime with UI selection and bundle scope.
                prod_files = [matched_product_files[0]]
                selected_product_file = matched_product_files[0]
                self.selected_product = selected_product_file
            else:
                selected_product_file = prod_files[0]
                self.selected_product = selected_product_file

            selected_safety_raw = str(self.selected_safety_file or "").strip()
            selected_safety_for_compat: str | None
            if selected_safety_raw:
                selected_safety_for_compat = selected_safety_raw
            else:
                selected_safety_for_compat = None

            startup_bundle_scope = self._startup_bundle_scope_input(
                selected_product_file,
                selected_requirement_file,
            )

            bundle_context, bundle_notice = await asyncio.to_thread(
                self._resolve_startup_bundle_context,
                product_spec_file=startup_bundle_scope,
                execution_mode=self.execution_mode,
                robot_env=self.robot_env,
                safety_requirement_file=selected_safety_for_compat,
            )
            if bundle_notice:
                self.last_notice = bundle_notice
            if bundle_context:
                self._diag_emit(
                    f"[Bundle] startup using bundle_id={bundle_context.get('bundle_id', '')}"
                )

            runtime_overrides: dict[str, Any] = {}
            if selected_requirement_file:
                runtime_overrides["product_requirement_file"] = selected_requirement_file
            if selected_safety_raw:
                runtime_overrides["safety_file_override_set"] = True
                if selected_safety_raw.upper() == "__NONE__":
                    runtime_overrides["safety_file_override"] = None
                else:
                    runtime_overrides["safety_file_override"] = selected_safety_raw
            else:
                runtime_overrides["safety_file_override_set"] = False

            # Create agents, passing prewarmed controllers for reuse.
            self._set_startup_phase("create_agents")
            (
                self.user_agent,
                self.resource_agents,
                self.product_agents,
                self.cca,
            ) = await asyncio.to_thread(
                self._create_agents,
                ac,
                prod_files,
                res_files,
                str(_CCA_INIT),
                prewarmed,
                bundle_context,
                runtime_overrides,
            )
            self._bind_agents_to_running_loop(
                asyncio.get_running_loop(),
                user_agent=self.user_agent,
                resource_agents=self.resource_agents,
                product_agents=self.product_agents,
                cca=self.cca,
            )
            self._apply_runtime_bridge_session_settings()
            self._diag_emit(
                f"startup#{startup_id} agents created resources={len(self.resource_agents)} "
                f"products={len(self.product_agents)} in {time.monotonic() - startup_t0:.2f}s"
            )

            # Build tools catalogue unless a verified bundle already provides
            # an exact snapshot for this startup context.
            self._set_startup_phase("build_tools_catalogue")
            resolved_tools_path, using_bundle_tools = await asyncio.to_thread(
                self._resolve_llm_tools_catalogue_path,
                bundle_context,
            )
            if using_bundle_tools:
                self._diag_emit(
                    f"startup#{startup_id} skipping tools rebuild; using bundled catalogue {resolved_tools_path}"
                )
            else:
                await asyncio.to_thread(
                    FunctionAnalyzer.build_tools_catalogue,
                    self.product_agents + self.resource_agents,
                    ac.ALLOWED_FUNCS,
                    str(_TOOLS_OUT),
                )
            tools_catalogue_path = await asyncio.to_thread(
                self._configure_llm_tools_catalogue,
                bundle_context,
            )
            self._diag_emit(
                f"startup#{startup_id} tools catalogue ready in {time.monotonic() - startup_t0:.2f}s"
            )
            self._diag_emit(
                f"startup#{startup_id} llm tools catalogue source={tools_catalogue_path}"
            )

            # Start agents in order: resources → CCA → user → products.
            self._set_startup_phase("start_resource_agents")
            ra_tasks = []
            for ra in self.resource_agents:
                ra_tasks.append(ra.start(auto_register=True))
            if ra_tasks:
                t_ra = time.monotonic()
                self._diag_emit(f"startup#{startup_id} starting {len(ra_tasks)} resource agents")
                await asyncio.gather(*ra_tasks)
                self._diag_emit(
                    f"startup#{startup_id} resource agents started in {time.monotonic() - t_ra:.2f}s"
                )

            self._set_startup_phase("start_cca")
            if self.cca:
                t_cca = time.monotonic()
                await self.cca.start(auto_register=True)
                self._diag_emit(
                    f"startup#{startup_id} cca started in {time.monotonic() - t_cca:.2f}s"
                )

            self._set_startup_phase("start_user")
            if self.user_agent:
                t_user = time.monotonic()
                await self.user_agent.start(auto_register=True)
                self._diag_emit(
                    f"startup#{startup_id} user started in {time.monotonic() - t_user:.2f}s"
                )

            self._set_startup_phase("start_product_agents")
            pa_tasks = []
            for pa in self.product_agents:
                pa_tasks.append(pa.start(auto_register=True))
            if pa_tasks:
                t_pa = time.monotonic()
                self._diag_emit(f"startup#{startup_id} starting {len(pa_tasks)} product agents")
                await asyncio.gather(*pa_tasks)
                self._diag_emit(
                    f"startup#{startup_id} product agents started in {time.monotonic() - t_pa:.2f}s"
                )

            self._set_startup_phase("wait_product_kickoff")
            kickoff_results: list[dict[str, Any]] = []
            for pa in self.product_agents:
                wait_for_kickoff = getattr(pa, "wait_for_kickoff_result", None)
                if not callable(wait_for_kickoff):
                    kickoff_results.append(
                        {
                            "success": False,
                            "message": f"{getattr(pa, 'agent_name', getattr(pa, 'jid', 'product'))}: kickoff wait unavailable",
                        }
                    )
                    continue
                result = await wait_for_kickoff(timeout=180.0)
                if isinstance(result, dict):
                    kickoff_results.append(result)
                else:
                    kickoff_results.append(
                        {
                            "success": False,
                            "message": f"{getattr(pa, 'agent_name', getattr(pa, 'jid', 'product'))}: invalid kickoff result",
                        }
                    )

            kickoff_failures = [r for r in kickoff_results if not bool(r.get("success", False))]
            if kickoff_failures:
                cached_alerts = []
                messages = []
                for failure in kickoff_failures:
                    alert = failure.get("alert")
                    if isinstance(alert, dict) and alert:
                        cached_alerts.append(alert)
                    msg = str(failure.get("message", "")).strip()
                    if msg:
                        messages.append(msg)
                if cached_alerts:
                    self._cache_plan_safety_alerts(cached_alerts)
                raise RuntimeError(" ; ".join(messages) or "product kickoff safety validation failed")

            self.system_running = True
            self._clear_cached_plan_safety_alerts()
            self._set_startup_phase("startup_complete")
            log.info("All agents started successfully.")
            self._diag_emit(
                f"startup#{startup_id} complete in {time.monotonic() - startup_t0:.2f}s"
            )

        except Exception as exc:
            self.last_error = str(exc)
            self._set_startup_phase("startup_failed")
            self._diag_emit(
                f"startup#{startup_id} failed after {time.monotonic() - startup_t0:.2f}s: {exc}"
            )
            log.exception("Failed to start system")
            await self._cleanup_agents()
        finally:
            self._starting = False
            if not self.system_running:
                self._set_startup_phase("idle")

    async def stop_system(self) -> None:
        """Stop all SPADE agents."""
        if not self.system_running or self._stopping:
            return
        self._stopping = True
        try:
            await self._cleanup_agents()
            self.system_running = False
            self._clear_cached_plan_safety_alerts()
            log.info("All agents stopped.")
        finally:
            self._stopping = False

    async def _cleanup_agents(self) -> None:
        all_agents = (
            self.product_agents
            + self.resource_agents
            + ([self.cca] if self.cca else [])
            + ([self.user_agent] if self.user_agent else [])
        )
        for a in all_agents:
            try:
                await a.stop()
            except Exception:
                pass
        self.product_agents = []
        self.resource_agents = []
        self.cca = None
        self.user_agent = None

        # Destroy the global CameraModule ROS2 node (if any).
        try:
            import agent_creator as ac
            if hasattr(ac, "_CAMERA") and ac._CAMERA is not None:
                ac._CAMERA.destroy()
                log.info("CameraModule ROS2 node destroyed.")
        except Exception:
            log.debug("CameraModule cleanup skipped (not loaded or already destroyed).")

    def _archive_monitors(self) -> dict[str, int]:
        archived_counts: dict[str, int] = {}
        for sub, pattern in [
            ("history", "*.jsonl"),
            ("plan", "*.json"),
            ("state", "*.json"),
        ]:
            d = _MONITOR / sub
            if not d.exists():
                archived_counts[sub] = 0
                continue
            files = [p for p in d.glob(pattern) if p.is_file()]
            if not files:
                archived_counts[sub] = 0
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive = d / "archive" / stamp
            archive.mkdir(parents=True, exist_ok=True)
            moved = 0
            for p in files:
                try:
                    shutil.move(str(p), str(archive / p.name))
                    moved += 1
                except Exception:
                    pass
            archived_counts[sub] = moved
        bridge_runtime_dir = _BRIDGE_RUNTIME_DATA_DIR
        moved = 0
        if bridge_runtime_dir.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive = bridge_runtime_dir / "archive" / stamp
            reserved_dirs = {"archive", "imported"}
            run_dir_pattern = re.compile(r"^\d{8}T\d{6}(?:_\d+)?$")
            entries_to_archive: list[Path] = []
            for entry in bridge_runtime_dir.iterdir():
                if entry.name in reserved_dirs:
                    continue
                if entry.is_file() and entry.suffix.lower() in {".txt", ".json", ".md"}:
                    entries_to_archive.append(entry)
                elif entry.is_dir() and run_dir_pattern.match(entry.name):
                    entries_to_archive.append(entry)
            if entries_to_archive:
                archive.mkdir(parents=True, exist_ok=True)
            for entry in entries_to_archive:
                try:
                    if entry.is_dir():
                        file_count = sum(1 for child in entry.rglob("*") if child.is_file())
                        shutil.move(str(entry), str(archive / entry.name))
                        moved += max(1, file_count)
                    else:
                        shutil.move(str(entry), str(archive / entry.name))
                        moved += 1
                except Exception:
                    pass
        self._invalidate_runtime_bridge_archive_cache()
        archived_counts["llm_bridge"] = moved
        return archived_counts

    @staticmethod
    def _import_agent_creator_module():
        import agent_creator as ac
        return ac

    @staticmethod
    def _configure_agent_creator_runtime(
        agent_creator_module: Any,
        robot_env: str,
        execution_mode: str,
        perception_backend: str,
    ) -> None:
        configure = getattr(agent_creator_module, "configure_runtime", None)
        if callable(configure):
            configure(
                robot_env=robot_env,
                execution_mode=execution_mode,
                perception_backend=perception_backend,
            )

    @staticmethod
    def _collect_init_files() -> tuple[list[str], list[str]]:
        import utils

        prod_files = utils.get_init_files(str(_PRODUCT_DIR))
        res_files = utils.get_init_files(str(_RESOURCE_DIR))
        return prod_files, res_files

    @staticmethod
    def _create_agents(
        agent_creator_module: Any,
        prod_files: list[str],
        res_files: list[str],
        cca_init_file: str,
        prewarmed_controllers: dict[str, Any],
        bundle_context: dict[str, Any] | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> tuple[Any, list[Any], list[Any], Any]:
        runtime_overrides = dict(runtime_overrides or {})
        product_requirement_file = runtime_overrides.get("product_requirement_file")
        safety_override_set = bool(runtime_overrides.get("safety_file_override_set", False))
        safety_override = runtime_overrides.get("safety_file_override")

        user_agent = agent_creator_module.create_user()
        resource_agents = agent_creator_module.create_resource_agents(
            res_files,
            cca_init_file,
            prewarmed_controllers=prewarmed_controllers,
        )
        product_kwargs: dict[str, Any] = {}
        if product_requirement_file:
            product_kwargs["product_requirement_file"] = product_requirement_file
        if safety_override_set:
            product_kwargs["safety_file_override"] = safety_override
        product_agents = agent_creator_module.create_product_agents(
            prod_files,
            resource_agents,
            cca_init_file,
            bundle_context=bundle_context,
            **product_kwargs,
        )
        cca_kwargs: dict[str, Any] = {}
        if safety_override_set:
            cca_kwargs["safety_file_override"] = safety_override
        cca = agent_creator_module.create_central_controller(
            cca_init_file,
            resource_agents,
            bundle_context=bundle_context,
            **cca_kwargs,
        )
        return user_agent, resource_agents, product_agents, cca

    @staticmethod
    def _bind_agents_to_running_loop(
        loop: asyncio.AbstractEventLoop,
        *,
        user_agent: Any,
        resource_agents: list[Any],
        product_agents: list[Any],
        cca: Any,
    ) -> None:
        agents = [
            agent
            for agent in [user_agent, cca, *list(resource_agents or []), *list(product_agents or [])]
            if agent is not None
        ]
        containers: list[Any] = []
        for agent in agents:
            container = getattr(agent, "container", None)
            if container is None or container in containers:
                continue
            containers.append(container)
        for container in containers:
            try:
                container.loop = loop
            except Exception:
                continue
        for agent in agents:
            setter = getattr(agent, "set_loop", None)
            if callable(setter):
                setter(loop)
            else:
                try:
                    agent.loop = loop
                except Exception:
                    continue

    @staticmethod
    def _resolve_llm_tools_catalogue_path(
        bundle_context: dict[str, Any] | None = None,
    ) -> tuple[Path, bool]:
        tools_path: Path = _TOOLS_OUT.resolve()
        artifacts = bundle_context.get("artifacts", {}) if isinstance(bundle_context, dict) else {}
        if isinstance(artifacts, dict):
            bundled_tools = str(artifacts.get("tools_json", "")).strip()
            if bundled_tools:
                candidate = Path(bundled_tools)
                if candidate.exists():
                    return candidate.resolve(), True
        return tools_path, False

    @staticmethod
    def _configure_llm_tools_catalogue(bundle_context: dict[str, Any] | None = None) -> str:
        try:
            from agents.shared_information.llm_agent import LlmAgent
        except ImportError:
            from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent

        tools_path, _ = SystemBridge._resolve_llm_tools_catalogue_path(bundle_context)
        return LlmAgent.configure_shared_tools_catalogue(tools_path)

    # ------------------------------------------------------------------
    # ROS2 process management
    # ------------------------------------------------------------------
    # Shell preamble that sources ROS2 outside of the Poetry venv.
    _ROS2_ENV = (
        "unset VIRTUAL_ENV PYTHONHOME; "
        "export PATH=/usr/bin:/usr/local/bin:$PATH; "
        "source /opt/ros/humble/setup.bash && "
        "source $HOME/ros2_ws/install/setup.bash && "
    )

    _TELEOP_SCRIPT = str(_PROJECT_ROOT / "ros2" / "cais_lab_gazebo" / "scripts" / "keyboard_teleop.py")

    # Registry: name → shell command suffix (appended after _ROS2_ENV).
    ROS2_LAUNCH_CMDS: dict[str, str] = {
        "gazebo_dual": "ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py",
        "gazebo_xarm6": "ros2 launch xarm_gazebo xarm6_moveit_single_gazebo.launch.py",
        "gazebo_ur5e": "ros2 launch xarm_gazebo ur5e_rg2_moveit_gazebo.launch.py",
        "hardware_xarm6_driver": "ros2 launch xarm_api xarm6_driver.launch.py robot_ip:={xarm6_ip}",
        "hardware_xarm6_moveit": (
            "ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py "
            "robot_ip:={xarm6_ip} add_gripper:=true"
        ),
        "hardware_ur5e_driver": "ros2 launch ur_robot_driver ur5e.launch.py robot_ip:={ur5e_ip} launch_rviz:=false",
        "hardware_ur5e_moveit": "ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true use_sim_time:=false",
        "perception": f"python3.10 {_PROJECT_ROOT / 'ros2' / 'cais_lab_gazebo' / 'sensor' / 'gazebo_camera_detector.py'}",
        "teleop_xarm6": f"python3.10 {_TELEOP_SCRIPT} --robot xarm6",
        "teleop_ur5e": f"python3.10 {_TELEOP_SCRIPT} --robot ur5e",
    }

    def ros2_proc_status(self, name: str) -> str:
        """Return 'running', 'stopped', or 'unknown' for a tracked ROS2 process."""
        proc = self._ros2_procs.get(name)
        if proc is None:
            return "stopped"
        rc = proc.poll()
        if rc is None:
            return "running"
        return "stopped"

    def ros2_all_statuses(self) -> dict[str, str]:
        return {name: self.ros2_proc_status(name) for name in self.ROS2_LAUNCH_CMDS}

    def teleop_target_environment(self) -> str:
        """Infer teleop target environment from active ROS2 stacks."""
        hardware_names = (
            "hardware_xarm6_driver",
            "hardware_xarm6_moveit",
            "hardware_ur5e_driver",
            "hardware_ur5e_moveit",
        )
        gazebo_names = ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")

        if any(self.ros2_proc_status(name) == "running" for name in hardware_names):
            return "real"
        if any(self.ros2_proc_status(name) == "running" for name in gazebo_names):
            return "gazebo"
        return "real" if str(self.robot_env).strip().lower() == "real" else "gazebo"

    def teleop_connection_status(self) -> dict[str, Any]:
        """Return teleop backend connectivity and resolved target environment."""
        with self._teleop_server_lock:
            proc = self._teleop_server_proc
            connected = proc is not None and proc.poll() is None
            pid = proc.pid if connected else None
        return {
            "connected": bool(connected),
            "environment": self.teleop_target_environment(),
            "pid": pid,
        }

    @staticmethod
    def _extract_robot_ip_from_resource(path: Path, robot_key: str) -> str | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            return None

        robot_block = data.get(robot_key, {})
        if not isinstance(robot_block, dict):
            return None
        real_block = robot_block.get("real", {})
        if not isinstance(real_block, dict):
            return None
        controller_block = real_block.get("controller", {})
        if isinstance(controller_block, dict):
            ip = controller_block.get("robot_ip")
            if isinstance(ip, str) and ip.strip():
                return ip.strip()
        ip = real_block.get("robot_ip")
        if isinstance(ip, str) and ip.strip():
            return ip.strip()
        return None

    @staticmethod
    def _is_ipv4(ip: str) -> bool:
        return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", ip or ""))

    def _load_hardware_ips(self) -> dict[str, str]:
        xarm6_ip = self._extract_robot_ip_from_resource(_XARM6_RESOURCE, "xarm6") or self._HW_IP_DEFAULTS["xarm6"]
        ur5e_ip = self._extract_robot_ip_from_resource(_UR5E_RESOURCE, "ur5e") or self._HW_IP_DEFAULTS["ur5e"]
        return {"xarm6": xarm6_ip, "ur5e": ur5e_ip}

    def get_hardware_ips(self) -> dict[str, str]:
        return dict(self.hardware_ips)

    def set_hardware_ip(self, robot: str, ip: str) -> str | None:
        key = str(robot).strip().lower()
        if key not in {"xarm6", "ur5e"}:
            return f"unknown robot: {robot}"
        value = str(ip).strip()
        if not self._is_ipv4(value):
            return f"invalid IPv4: {ip}"
        octets = [int(part) for part in value.split(".")]
        if any(part < 0 or part > 255 for part in octets):
            return f"invalid IPv4: {ip}"
        self.hardware_ips[key] = value
        self._hw_ping_cache.clear()
        self._hw_ping_last_ts = 0.0
        return None

    @staticmethod
    def _ping_once(ip: str, timeout_sec: float = 0.35) -> tuple[bool, float | None, str | None]:
        wait_sec = 1
        try:
            cp = subprocess.run(
                ["ping", "-c", "1", "-W", str(wait_sec), ip],
                capture_output=True,
                text=True,
                timeout=max(0.2, float(timeout_sec) + 0.1),
            )
        except FileNotFoundError:
            return False, None, "ping command not found"
        except subprocess.TimeoutExpired:
            return False, None, "timeout"
        except Exception as exc:
            return False, None, str(exc)

        if cp.returncode != 0:
            return False, None, "unreachable"

        latency = None
        for line in (cp.stdout or "").splitlines():
            match = re.search(r"time=([0-9]+(?:\.[0-9]+)?)\s*ms", line)
            if match:
                latency = float(match.group(1))
                break
        return True, latency, None

    def hardware_connection_statuses_cached(self) -> dict[str, dict[str, Any]]:
        if self._hw_ping_cache:
            return {k: dict(v) for k, v in self._hw_ping_cache.items()}

        return {
            robot: {
                "ip": self.hardware_ips.get(robot, ""),
                "reachable": False,
                "latency_ms": None,
                "message": "checking...",
            }
            for robot in ("xarm6", "ur5e")
        }

    def hardware_connection_statuses(self, force: bool = False) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        if (not force) and self._hw_ping_cache and (now - self._hw_ping_last_ts) < 5.0:
            return {k: dict(v) for k, v in self._hw_ping_cache.items()}

        result: dict[str, dict[str, Any]] = {}
        for robot in ("xarm6", "ur5e"):
            ip = self.hardware_ips.get(robot, "")
            if not ip:
                result[robot] = {
                    "ip": "",
                    "reachable": False,
                    "latency_ms": None,
                    "message": "no IP configured",
                }
                continue
            ok, latency, error = self._ping_once(ip, timeout_sec=0.35)
            result[robot] = {
                "ip": ip,
                "reachable": bool(ok),
                "latency_ms": latency,
                "message": "OK" if ok else (error or "unreachable"),
            }
        self._hw_ping_cache = result
        self._hw_ping_last_ts = now
        return {k: dict(v) for k, v in result.items()}

    def _render_ros2_launch_cmd(self, name: str) -> str:
        cmd = self.ROS2_LAUNCH_CMDS[name]
        return cmd.format(
            xarm6_ip=self.hardware_ips.get("xarm6", self._HW_IP_DEFAULTS["xarm6"]),
            ur5e_ip=self.hardware_ips.get("ur5e", self._HW_IP_DEFAULTS["ur5e"]),
        )

    @staticmethod
    def _ros2_setup_path() -> Path:
        return Path("/opt/ros/humble/setup.bash")

    @staticmethod
    def _ros2_workspace_root() -> Path:
        return Path.home() / "ros2_ws"

    @classmethod
    def _ros2_workspace_setup_path(cls) -> Path:
        return cls._ros2_workspace_root() / "install" / "setup.bash"

    @classmethod
    def _ros2_workspace_launch_dir(cls) -> Path:
        return cls._ros2_workspace_root() / "src" / "xarm_ros2" / "xarm_gazebo" / "launch"

    @classmethod
    def _ros2_workspace_install_pkg_path(cls, pkg_name: str) -> Path:
        return cls._ros2_workspace_root() / "install" / str(pkg_name).strip()

    @classmethod
    def _ros2_workspace_install_share_pkg_path(cls, pkg_name: str) -> Path:
        pkg_name = str(pkg_name).strip()
        return cls._ros2_workspace_install_pkg_path(pkg_name) / "share" / pkg_name

    @staticmethod
    def _ros2_system_share_pkg_path(pkg_name: str) -> Path:
        return Path("/opt/ros/humble/share") / str(pkg_name).strip()

    @classmethod
    def _ros2_launch_required_paths(cls, name: str) -> list[tuple[Path, str]]:
        launch_key = str(name or "").strip().lower()
        xarm_gazebo_share = cls._ros2_workspace_install_share_pkg_path("xarm_gazebo")
        workspace_xarm = (
            (
                cls._ros2_workspace_install_pkg_path("xarm_gazebo"),
                "ROS2 workspace is missing package 'xarm_gazebo'. Re-run `make bootstrap-gazebo`.",
            ),
            (
                cls._ros2_workspace_install_pkg_path("xarm_moveit_config"),
                "ROS2 workspace is missing package 'xarm_moveit_config'. Re-run `make bootstrap-gazebo`.",
            ),
        )
        moveit_core = (
            (
                cls._ros2_system_share_pkg_path("moveit_ros_move_group"),
                "MoveIt is not installed. Install `ros-humble-moveit`.",
            ),
        )
        ur_stack = (
            (
                cls._ros2_system_share_pkg_path("ur_description"),
                "UR description is not installed. Install `ros-humble-ur-description`.",
            ),
            (
                cls._ros2_system_share_pkg_path("ur_moveit_config"),
                "UR MoveIt config is not installed. Install `ros-humble-ur-moveit-config`.",
            ),
        )
        onrobot_ws = (
            (
                cls._ros2_workspace_install_pkg_path("onrobot_description"),
                "ROS2 workspace is missing package 'onrobot_description'. Re-run `make bootstrap-gazebo`.",
            ),
        )
        link_attacher_ws = (
            (
                cls._ros2_workspace_install_pkg_path("linkattacher_msgs"),
                "ROS2 workspace is missing package 'linkattacher_msgs'. Re-run `make bootstrap-gazebo` to install IFRA LinkAttacher.",
            ),
            (
                cls._ros2_workspace_install_pkg_path("ros2_linkattacher"),
                "ROS2 workspace is missing package 'ros2_linkattacher'. Re-run `make bootstrap-gazebo` to install IFRA LinkAttacher.",
            ),
        )
        dual_assets = (
            (
                xarm_gazebo_share / "config" / "xarm6_ur5e_controllers.yaml",
                "ROS2 workspace is missing the dual-robot xarm_gazebo controller config. Re-run `make bootstrap-gazebo`.",
            ),
            (
                xarm_gazebo_share / "config" / "ur5e_initial_positions.yaml",
                "ROS2 workspace is missing the UR5e initial positions config. Re-run `make bootstrap-gazebo`.",
            ),
            (
                xarm_gazebo_share / "rviz" / "dual_moveit.rviz",
                "ROS2 workspace is missing the dual-robot RViz config. Re-run `make bootstrap-gazebo`.",
            ),
        )
        ur_assets = (
            (
                xarm_gazebo_share / "config" / "ur5e_rg2_controllers.yaml",
                "ROS2 workspace is missing the UR5e RG2 controller config. Re-run `make bootstrap-gazebo`.",
            ),
        )

        if launch_key == "gazebo_dual":
            return [*workspace_xarm, *moveit_core, *ur_stack, *onrobot_ws, *link_attacher_ws, *dual_assets]
        if launch_key == "gazebo_xarm6":
            return [*workspace_xarm, *moveit_core, *link_attacher_ws]
        if launch_key == "gazebo_ur5e":
            return [*workspace_xarm, *moveit_core, *ur_stack, *onrobot_ws, *link_attacher_ws, *ur_assets]
        return []

    def _ros2_launch_prereq_error(self, name: str) -> str | None:
        ros_setup = self._ros2_setup_path()
        if not ros_setup.is_file():
            return (
                f"ROS 2 Humble is not installed: missing {ros_setup}. "
                "Install the README simulation dependencies first."
            )

        ws_setup = self._ros2_workspace_setup_path()
        if not ws_setup.is_file():
            return (
                f"ROS2 workspace is not built yet: missing {ws_setup}. "
                "Run `make bootstrap-gazebo` after installing the README simulation packages."
            )

        launch_file = self._GAZEBO_WORKSPACE_LAUNCH_FILES.get(str(name or "").strip().lower())
        if launch_file:
            launch_path = self._ros2_workspace_launch_dir() / launch_file
            if not launch_path.is_file():
                return (
                    f"ROS2 workspace is missing {launch_file} at {launch_path}. "
                    "Re-run `make bootstrap-gazebo` to copy the custom Gazebo launch files and rebuild."
                )
        for required_path, remedy in self._ros2_launch_required_paths(name):
            if not required_path.exists():
                return f"Missing ROS dependency at {required_path}. {remedy}"
        return None

    def _perception_backend_for_mode(self) -> str:
        mode = str(self.execution_mode or "").strip().lower()
        if mode == "simulation":
            return "gazebo_gt"
        if mode == "physical":
            return "yolo"
        return "none"

    def physical_perception_ready(self) -> tuple[bool, str]:
        """Gate physical mode until real YOLO perception backend is integrated."""
        if self._perception_backend_for_mode() != "yolo":
            return True, ""

        allow_placeholder = str(
            os.environ.get("ALLOW_PLACEHOLDER_PHYSICAL_PERCEPTION", "")
        ).strip().lower() in {"1", "true", "yes"}
        if allow_placeholder:
            return True, ""

        return (
            False,
            "Physical mode is blocked: perception backend 'yolo' is still a placeholder. "
            "Set ALLOW_PLACEHOLDER_PHYSICAL_PERCEPTION=1 to override intentionally.",
        )

    def simulation_start_ready(self, force: bool = False) -> tuple[bool, str]:
        """Return whether Gazebo simulation startup is ready enough for agent start."""
        now = time.monotonic()
        if not self._any_running(self._GAZEBO_PROCESS_NAMES):
            result = (False, "Gazebo stack is not running. Launch Gazebo + MoveIt first.")
            self._sim_ready_cache_ts = now
            self._sim_ready_cache = result
            return result

        # Non-force path is called from a 1s UI timer. Keep it cheap and avoid
        # repeatedly spawning `ros2 service list`, which can be expensive.
        with self._gazebo_prewarm_lock:
            prewarm_done = self._gazebo_prewarm_done.is_set()
            prewarm_inflight = bool(
                (self._gazebo_prewarm_thread and self._gazebo_prewarm_thread.is_alive())
                or self._gazebo_prewarm_pending
            )
            probe_inflight = self._sim_ready_probe_inflight

        # If prewarm completed successfully, core services were confirmed ready.
        # Perception is probed separately so a slow /detect_all does not block
        # agent startup or the dashboard readiness banner.
        if prewarm_done:
            if force:
                result = self._probe_sim_services(timeout_sec=6.0)
                if not result[0]:
                    # Successful controller prewarm is the authoritative core-service
                    # readiness check. Shell probes for ATTACHLINK/DETACHLINK are
                    # known to be flaky under WSL even when live ROS clients already
                    # connected successfully during prewarm.
                    self._diag_emit(
                        "simulation_start_ready(force) downgraded shell probe failure "
                        f"after completed prewarm: {result[1]}"
                    )
                    result = (
                        True,
                        self._simulation_perception_warning(),
                    )
                self._sim_ready_cache_ts = now
                self._sim_ready_cache = result
                return result
            if (not probe_inflight) and (now - self._sim_ready_cache_ts) >= 3.0:
                self._schedule_sim_ready_probe()
            if self._sim_ready_cache[0]:
                return self._sim_ready_cache
            result = (True, self._simulation_perception_warning())
            self._sim_ready_cache = result
            self._sim_ready_cache_ts = now
            return result

        if prewarm_inflight:
            result = (
                False,
                "Simulation startup is still initializing ROS services and controller prewarm. Please wait...",
            )
            self._sim_ready_cache_ts = now
            self._sim_ready_cache = result
            return result

        if force:
            result = self._probe_sim_services(timeout_sec=6.0)
            self._sim_ready_cache_ts = now
            self._sim_ready_cache = result
            return result

        if (not probe_inflight) and (now - self._sim_ready_cache_ts) >= 2.0:
            self._schedule_sim_ready_probe()

        # If prewarm is not running (e.g., user launched Gazebo outside the UI),
        # expose the last cached answer but do not shell out on every timer tick.
        if self._sim_ready_cache[0]:
            return self._sim_ready_cache
        return (
            False,
            "Simulation startup check pending. Launch Gazebo from Control (to enable prewarm) or wait a moment.",
        )

    def _probe_sim_services(self, timeout_sec: float = 3.0) -> tuple[bool, str]:
        ok, out = self._ros2_command_output("ros2 service list", timeout_sec=timeout_sec)
        if not ok:
            return (
                False,
                "Simulation startup is still initializing ROS services. "
                "Please wait a few seconds and try Start again.",
            )

        services = [line.strip() for line in out.splitlines() if line.strip()]
        missing_core = [
            svc
            for svc in self._SIM_CORE_SERVICES
            if not any(name == svc or name.endswith(svc) for name in services)
        ]
        if missing_core:
            return (
                False,
                "Simulation startup is not done yet. Waiting for core services: "
                + ", ".join(missing_core),
            )
        missing_perception = [
            svc
            for svc in self._SIM_PERCEPTION_SERVICES
            if not any(name == svc or name.endswith(svc) for name in services)
        ]
        if missing_perception:
            return (True, self._simulation_perception_warning(missing_perception))
        return (True, "")

    def _probe_sim_services_worker(self) -> None:
        try:
            result = self._probe_sim_services(timeout_sec=3.0)
            self._sim_ready_cache = result
            self._sim_ready_cache_ts = time.monotonic()
        finally:
            with self._gazebo_prewarm_lock:
                self._sim_ready_probe_inflight = False

    def _schedule_sim_ready_probe(self) -> None:
        with self._gazebo_prewarm_lock:
            if self._sim_ready_probe_inflight:
                return
            self._sim_ready_probe_inflight = True
        threading.Thread(
            target=self._probe_sim_services_worker,
            daemon=True,
        ).start()

    @classmethod
    def _simulation_perception_warning(
        cls,
        missing_services: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        missing = [
            str(name).strip()
            for name in (missing_services or cls._SIM_PERCEPTION_SERVICES)
            if str(name).strip()
        ]
        if not missing:
            return ""
        return (
            "Perception is still warming up: "
            + ", ".join(missing)
            + ". Start is allowed, but perception-dependent tasks may need a few more seconds."
        )

    @staticmethod
    def _simulation_prewarm_failure_message(
        failures: list[str] | tuple[str, ...],
    ) -> str:
        details = [str(item).strip() for item in failures if str(item).strip()]
        if not details:
            return (
                "Simulation startup is not done yet. Controller prewarm failed before all robots were ready."
            )
        return (
            "Simulation startup is not done yet. Controller prewarm failed: "
            + "; ".join(details)
        )

    def _any_running(self, names: set[str]) -> bool:
        return any(self.ros2_proc_status(name) == "running" for name in names)

    @staticmethod
    def _driver_service_hint_matches(robot: str, service_name: str) -> bool:
        name = str(service_name or "").strip().lower()
        key = str(robot).strip().lower()
        if key == "xarm6":
            return ("xarm" in name) and any(token in name for token in ("motion_enable", "set_mode", "set_state"))
        if key == "ur5e":
            return ("dashboard_client" in name) or ("ur_hardware_interface" in name)
        return False

    def _ros2_command_output(
        self,
        command: str,
        timeout_sec: float = 8.0,
        *,
        emit_slow_diag: bool = True,
        emit_failure_diag: bool = True,
        emit_timeout_diag: bool = True,
    ) -> tuple[bool, str]:
        full_cmd = self._ROS2_ENV + command
        t0 = time.monotonic()
        try:
            cp = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_sec)),
            )
        except subprocess.TimeoutExpired:
            if emit_timeout_diag:
                self._diag_emit(f"ros2 command timeout ({timeout_sec:.1f}s): {command}")
            return False, f"command timed out after {timeout_sec:.0f}s"

        elapsed = time.monotonic() - t0
        if cp.returncode != 0:
            detail = self._tail_output(cp.stderr) or self._tail_output(cp.stdout)
            if emit_failure_diag:
                self._diag_emit(
                    f"ros2 command failed rc={cp.returncode} elapsed={elapsed:.2f}s: {command}"
                )
            if detail:
                return False, detail
            return False, f"command failed (exit {cp.returncode})"
        if emit_slow_diag and elapsed > 1.0:
            self._diag_emit(f"ros2 command slow elapsed={elapsed:.2f}s: {command}")
        return True, cp.stdout or ""

    def _wait_for_ros_service(
        self,
        service_name: str,
        timeout_sec: float = 20.0,
        process_name: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str | None:
        return self._wait_for_ros_services(
            [service_name],
            timeout_sec=timeout_sec,
            process_name=process_name,
            cancel_event=cancel_event,
        )

    def _wait_for_ros_services(
        self,
        service_names: list[str] | tuple[str, ...],
        timeout_sec: float = 20.0,
        process_name: str | None = None,
        cancel_event: threading.Event | None = None,
        poll_interval_sec: float = 0.5,
        progress_cb: Callable[[str, float], None] | None = None,
        pending_cb: Callable[[list[str], float], None] | None = None,
    ) -> str | None:
        targets = [str(name).strip() for name in service_names if str(name).strip()]
        if not targets:
            return "service name is empty"

        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        seen_targets: set[str] = set()
        last_missing = list(targets)
        while time.monotonic() < deadline:
            if cancel_event and cancel_event.is_set():
                return f"{', '.join(targets)} wait cancelled"

            if process_name and self.ros2_proc_status(process_name) != "running":
                return f"{process_name} exited before {', '.join(targets)} became available"

            ok, out = self._ros2_command_output(
                "ros2 service list",
                timeout_sec=3.0,
                emit_slow_diag=False,
                emit_failure_diag=False,
                emit_timeout_diag=False,
            )
            if ok:
                services = [line.strip() for line in out.splitlines() if line.strip()]
                target_matches: dict[str, bool] = {}
                for target in targets:
                    matched = any(s == target or s.endswith(target) for s in services)
                    target_matches[target] = matched
                    if matched and target not in seen_targets:
                        seen_targets.add(target)
                        if progress_cb is not None:
                            try:
                                progress_cb(target, time.monotonic())
                            except Exception:
                                pass
                last_missing = [target for target, matched in target_matches.items() if not matched]
                if last_missing and pending_cb is not None:
                    try:
                        pending_cb(last_missing, time.monotonic())
                    except Exception:
                        pass
                if all(target_matches.values()):
                    return None
            time.sleep(max(0.1, float(poll_interval_sec)))

        targets_text = ", ".join(targets)
        if process_name and self.ros2_proc_status(process_name) != "running":
            return f"{process_name} exited before {targets_text} became available"
        missing_text = ", ".join(last_missing or targets)
        return f"{missing_text} not available within {timeout_sec:.0f}s"

    def _wait_for_driver_ready(self, robot: str, timeout_sec: float = 18.0) -> str | None:
        key = str(robot).strip().lower()
        stack = self._HARDWARE_STACKS.get(key)
        if not stack:
            return f"unknown hardware robot: {robot}"
        if len(stack) < 2:
            return None
        driver_name = stack[0]

        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        stable_since: float | None = None

        while time.monotonic() < deadline:
            if self.ros2_proc_status(driver_name) != "running":
                stable_since = None
                time.sleep(0.6)
                continue

            now = time.monotonic()
            if stable_since is None:
                stable_since = now

            # Prefer explicit driver-service detection when available.
            ok, out = self._ros2_command_output("ros2 service list", timeout_sec=3.0)
            if ok:
                services = [line.strip() for line in out.splitlines() if line.strip()]
                if any(self._driver_service_hint_matches(key, service_name) for service_name in services):
                    return None

            # Best-effort fallback: if process stays alive for a short warmup, continue.
            if (now - stable_since) >= 3.0:
                return None

            time.sleep(0.7)

        if self.ros2_proc_status(driver_name) != "running":
            return f"{driver_name} exited before hardware became ready"
        return f"{driver_name} did not become ready within {timeout_sec:.0f}s"

    def hardware_stack_status(self, robot: str) -> dict[str, str]:
        key = str(robot).strip().lower()
        stack = self._HARDWARE_STACKS.get(key)
        if not stack:
            return {"overall": "unknown", "driver": "unknown", "moveit": "unknown"}
        if len(stack) == 1:
            moveit_state = self.ros2_proc_status(stack[0])
            overall = "running" if moveit_state == "running" else "stopped"
            return {"overall": overall, "driver": "embedded", "moveit": moveit_state}

        driver_name, moveit_name = stack[0], stack[1]
        driver_state = self.ros2_proc_status(driver_name)
        moveit_state = self.ros2_proc_status(moveit_name)
        if driver_state == "running" and moveit_state == "running":
            overall = "running"
        elif driver_state == "stopped" and moveit_state == "stopped":
            overall = "stopped"
        else:
            overall = "partial"
        return {"overall": overall, "driver": driver_state, "moveit": moveit_state}

    def ros2_start_hardware_stack(self, robot: str) -> str | None:
        key = str(robot).strip().lower()
        stack = self._HARDWARE_STACKS.get(key)
        if not stack:
            return f"unknown hardware robot: {robot}"
        if self._any_running(self._GAZEBO_PROCESS_NAMES):
            return "Hardware stack blocked: Gazebo is running. Stop Gazebo first."

        hw_links = self.hardware_connection_statuses(force=True)
        entry = hw_links.get(key, {})
        if not entry.get("reachable", False):
            ip = str(entry.get("ip", "")).strip() or "(unknown IP)"
            msg = str(entry.get("message", "unreachable")).strip()
            return f"{key} hardware is unreachable at {ip} ({msg})"

        self.robot_env = "real"
        self._stop_teleop_server()

        if len(stack) == 1:
            moveit_name = stack[0]
            # Ensure legacy external xarm driver is not concurrently connected.
            self.ros2_stop("hardware_xarm6_driver")
            if self.ros2_proc_status(moveit_name) != "running":
                err = self.ros2_start(moveit_name)
                if err:
                    return err
            err = self._wait_for_ros_service(
                "/controller_manager/list_controllers",
                timeout_sec=22.0,
                process_name=moveit_name,
            )
            if err:
                return (
                    "xArm6 hardware stack is not ready: "
                    f"{err}. "
                    "MoveIt may have failed to launch correctly."
                )
            return None

        driver_name, moveit_name = stack[0], stack[1]
        if self.ros2_proc_status(driver_name) != "running":
            err = self.ros2_start(driver_name)
            if err:
                return err

        err = self._wait_for_driver_ready(key, timeout_sec=18.0)
        if err:
            return err

        if self.ros2_proc_status(moveit_name) != "running":
            err = self.ros2_start(moveit_name)
            if err:
                return err
        err = self._wait_for_ros_service(
            "/controller_manager/list_controllers",
            timeout_sec=22.0,
            process_name=moveit_name,
        )
        if err:
            return (
                f"{key} hardware stack is not ready: {err}. "
                "MoveIt may have failed to launch correctly."
            )
        return None

    def ros2_stop_hardware_stack(self, robot: str) -> str | None:
        key = str(robot).strip().lower()
        stack = self._HARDWARE_STACKS.get(key)
        if not stack:
            return f"unknown hardware robot: {robot}"
        if len(stack) == 1:
            self.ros2_stop(stack[0])
            # Also stop potential legacy standalone driver if user launched it.
            self.ros2_stop("hardware_xarm6_driver")
        else:
            driver_name, moveit_name = stack[0], stack[1]
            self.ros2_stop(moveit_name)
            self.ros2_stop(driver_name)
        self._stop_teleop_server()
        return None

    def _load_gazebo_controller_settings(self, robot: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        robot_key = str(robot or "").strip().lower()
        path_map = {
            "xarm6": _XARM6_RESOURCE,
            "ur5e": _UR5E_RESOURCE,
        }
        cfg_path = path_map.get(robot_key)
        if not cfg_path or not cfg_path.exists():
            log.warning("Gazebo prewarm skipped; config file missing for %s", robot_key)
            return None

        try:
            raw = self.load_config(str(cfg_path))
        except Exception:
            log.exception("Gazebo prewarm failed reading config for %s", robot_key)
            return None

        block = raw.get(robot_key, {}) if isinstance(raw, dict) else {}
        gazebo_block = block.get("gazebo", {}) if isinstance(block, dict) else {}
        controller = gazebo_block.get("controller", {}) if isinstance(gazebo_block, dict) else {}
        named_positions = gazebo_block.get("named_positions", {}) if isinstance(gazebo_block, dict) else {}
        if not controller:
            log.warning("Gazebo prewarm skipped; controller config empty for %s", robot_key)
            return None
        if not isinstance(named_positions, dict):
            named_positions = {}
        return controller, named_positions

    def _prewarm_gazebo_controller(self, robot: str) -> tuple[bool, float, str]:
        start_ts = time.monotonic()
        robot_key = str(robot or "").strip().lower()
        settings = self._load_gazebo_controller_settings(robot_key)
        if settings is None:
            return False, time.monotonic() - start_ts, "controller config missing"
        controller_cfg, named_positions = settings

        with self._gazebo_prewarm_lock:
            existing = self._gazebo_prewarm_controllers.get(robot_key)
        if existing is not None:
            try:
                if existing.wait_for_services(timeout_sec=1.0):
                    log.info("Gazebo prewarm already ready for %s", robot_key)
                    return True, time.monotonic() - start_ts, "reused existing controller"
            except Exception:
                pass
            try:
                existing.shutdown()
            except Exception:
                pass
            with self._gazebo_prewarm_lock:
                self._gazebo_prewarm_controllers.pop(robot_key, None)

        try:
            from cais_spade_llm.resources.robot.ros2_pick_place_controller import (
                Ros2PickPlaceController,
            )
            from cais_spade_llm.resources.robot.ur5e_controller import (
                JOINT_NAMES as UR5E_JOINT_NAMES,
                JOINT_STATES_TOPIC as UR5E_JOINT_STATES_TOPIC,
            )
            from cais_spade_llm.resources.robot.xarm6_controller import (
                JOINT_NAMES as XARM6_JOINT_NAMES,
                JOINT_STATES_TOPIC as XARM6_JOINT_STATES_TOPIC,
            )
        except Exception:
            log.exception("Gazebo prewarm failed importing controller modules for %s", robot_key)
            return False, time.monotonic() - start_ts, "controller import failure"

        controller = None
        detail = ""
        try:
            if robot_key == "xarm6":
                prewarm_node = f"xarm6_prewarm_controller_{os.getpid()}_{int(time.monotonic() * 1000) % 1000000}"
                controller = Ros2PickPlaceController(
                    robot_name="xarm6",
                    node_name=prewarm_node,
                    controller_config=controller_cfg,
                    named_positions=named_positions,
                    execution_mode="simulation",
                    arm_joint_names=XARM6_JOINT_NAMES,
                    arm_trajectory_topic=None,
                    joint_states_topic=XARM6_JOINT_STATES_TOPIC,
                )
            elif robot_key == "ur5e":
                prewarm_node = f"ur5e_prewarm_controller_{os.getpid()}_{int(time.monotonic() * 1000) % 1000000}"
                controller = Ros2PickPlaceController(
                    robot_name="ur5e",
                    node_name=prewarm_node,
                    controller_config=controller_cfg,
                    named_positions=named_positions,
                    execution_mode="simulation",
                    arm_joint_names=UR5E_JOINT_NAMES,
                    arm_trajectory_topic=_UR5E_GAZEBO_ARM_TRAJECTORY_TOPIC,
                    joint_states_topic=UR5E_JOINT_STATES_TOPIC,
                )
            else:
                return False, time.monotonic() - start_ts, f"unknown robot {robot_key}"

            start = time.monotonic()
            ok = bool(controller.wait_for_services(timeout_sec=self._GAZEBO_PREWARM_TIMEOUT_S))
            elapsed = time.monotonic() - start
            if ok:
                with self._gazebo_prewarm_lock:
                    old = self._gazebo_prewarm_controllers.pop(robot_key, None)
                    self._gazebo_prewarm_controllers[robot_key] = controller
                if old is not None and old is not controller:
                    try:
                        old.shutdown()
                    except Exception:
                        pass
                controller = None  # kept alive for reuse; cleaned up when Gazebo stops
                log.info("Gazebo prewarm ready for %s in %.2fs", robot_key, elapsed)
                return True, time.monotonic() - start_ts, "controller ready"
            else:
                log.warning("Gazebo prewarm not ready for %s after %.2fs", robot_key, elapsed)
                detail = str(
                    getattr(controller, "_last_failure_message", "") or "wait_for_services returned false"
                ).strip()
                return False, time.monotonic() - start_ts, detail
        except Exception:
            log.exception("Gazebo prewarm exception for %s", robot_key)
            detail = "exception during controller prewarm"
            return False, time.monotonic() - start_ts, detail
        finally:
            if controller is not None:
                try:
                    controller.shutdown()
                except Exception:
                    log.exception("Gazebo prewarm cleanup failed for %s", robot_key)

    def _shutdown_gazebo_prewarm_controllers(self) -> None:
        # Signal any in-progress prewarm wait to stop immediately.
        self._gazebo_prewarm_cancel.set()
        prewarm_thread = None
        with self._gazebo_prewarm_lock:
            controllers = dict(self._gazebo_prewarm_controllers)
            self._gazebo_prewarm_controllers.clear()
            self._gazebo_prewarm_pending.clear()
            self._sim_ready_probe_inflight = False
            prewarm_thread = self._gazebo_prewarm_thread
        # Wait for the prewarm worker to exit (it checks cancel_event).
        if prewarm_thread is not None:
            prewarm_thread.join(timeout=5.0)
        for robot_key, controller in controllers.items():
            try:
                controller.shutdown()
            except Exception:
                log.exception("Gazebo prewarm controller shutdown failed for %s", robot_key)

    def _gazebo_prewarm_worker(self) -> None:
        try:
            # Use interruptible wait instead of blocking sleep.
            if self._gazebo_prewarm_cancel.wait(timeout=self._GAZEBO_PREWARM_START_DELAY_S):
                log.info("Gazebo prewarm cancelled during initial delay.")
                return
            while True:
                if self._gazebo_prewarm_cancel.is_set():
                    log.info("Gazebo prewarm cancelled.")
                    return

                with self._gazebo_prewarm_lock:
                    targets = sorted(self._gazebo_prewarm_pending)
                    self._gazebo_prewarm_pending.clear()

                if not targets:
                    return
                if not self._any_running(self._GAZEBO_PROCESS_NAMES):
                    log.info("Gazebo prewarm skipped; no Gazebo stack running.")
                    return

                # Phase 1: Wait only for the lightweight, consistently visible
                # MoveIt service via `ros2 service list`.  Custom Gazebo world
                # plugins like ATTACHLINK/DETACHLINK can be slow or flaky to
                # appear in shell probes under WSL even when direct ROS clients
                # can already connect, so those are verified in phase 2 by the
                # controller's actual service clients.
                phase1_targets = list(self._SIM_PREWARM_SHELL_SERVICES)
                self._gazebo_phase1_begin(phase1_targets)
                wait_err = self._wait_for_ros_services(
                    phase1_targets,
                    timeout_sec=self._GAZEBO_PREWARM_READY_WAIT_S,
                    cancel_event=self._gazebo_prewarm_cancel,
                    progress_cb=self._gazebo_note_service_ready,
                    pending_cb=self._gazebo_note_services_pending,
                )
                if wait_err:
                    self._gazebo_phase1_failed(wait_err)
                    log.info("Gazebo prewarm skipped (service not ready): %s", wait_err)
                    return

                self._gazebo_phase1_complete()
                probe_result = self._probe_sim_services(timeout_sec=3.0)
                if not probe_result[0]:
                    probe_result = (True, self._simulation_perception_warning())
                self._sim_ready_cache = probe_result
                self._sim_ready_cache_ts = time.monotonic()
                if probe_result[1]:
                    log.info("Gazebo prewarm: %s", probe_result[1])
                log.info(
                    "Gazebo prewarm: all required services detected for %s — "
                    "creating controllers...",
                    ", ".join(targets),
                )

                # Phase 2: Create controllers now that services are confirmed
                # available.  This pre-initialises ROS2 nodes, spin threads,
                # MoveIt / TF clients so that REQ_1_T1 doesn't pay the startup
                # cost when "Start System" is pressed.
                if self._gazebo_prewarm_cancel.is_set():
                    return
                ready_count = 0
                failures: list[str] = []
                for robot_key in targets:
                    if self._gazebo_prewarm_cancel.is_set():
                        return
                    self._gazebo_note_prewarm_start(robot_key)
                    ok, elapsed, detail = self._prewarm_gazebo_controller(robot_key)
                    if ok:
                        ready_count += 1
                    else:
                        detail_text = str(detail or "controller not ready").strip() or "controller not ready"
                        failures.append(f"{robot_key}: {detail_text}")
                    self._gazebo_note_prewarm_result(robot_key, ok, elapsed, detail)

                if ready_count == len(targets):
                    # Signal that prewarm is done so simulation_start_ready() unblocks.
                    self._gazebo_prewarm_done.set()
                    self._gazebo_launch_complete(ready_count, len(targets))
                    continue

                self._sim_ready_cache = (
                    False,
                    self._simulation_prewarm_failure_message(failures),
                )
                self._sim_ready_cache_ts = time.monotonic()
                meta = self._gazebo_launch_timing_snapshot()
                if meta:
                    self._gazebo_timing_emit(
                        f"launch#{meta.get('id', 0)} prewarm incomplete "
                        f"targets_ready={ready_count}/{len(targets)} detail={' | '.join(failures)}"
                    )
                return
        except Exception:
            log.exception("Gazebo prewarm worker crashed.")
        finally:
            with self._gazebo_prewarm_lock:
                self._gazebo_prewarm_thread = None
                if self._gazebo_prewarm_pending and not self._gazebo_prewarm_cancel.is_set():
                    self._gazebo_prewarm_thread = threading.Thread(
                        target=self._gazebo_prewarm_worker,
                        daemon=True,
                    )
                    self._gazebo_prewarm_thread.start()

    def _queue_gazebo_prewarm(self, launch_name: str) -> None:
        launch_key = str(launch_name or "").strip().lower()
        targets_by_launch = {
            "gazebo_dual": {"xarm6", "ur5e"},
            "gazebo_xarm6": {"xarm6"},
            "gazebo_ur5e": {"ur5e"},
        }
        targets = targets_by_launch.get(launch_key)
        if not targets:
            return

        with self._gazebo_prewarm_lock:
            self._gazebo_prewarm_cancel.clear()
            self._gazebo_prewarm_done.clear()
            self._gazebo_prewarm_pending.update(targets)
            if self._gazebo_prewarm_thread and self._gazebo_prewarm_thread.is_alive():
                return
            self._gazebo_prewarm_thread = threading.Thread(
                target=self._gazebo_prewarm_worker,
                daemon=True,
            )
            self._gazebo_prewarm_thread.start()
        log.info("Queued Gazebo controller prewarm for %s", ",".join(sorted(targets)))
        meta = self._gazebo_launch_timing_snapshot()
        if meta:
            launch_elapsed = time.monotonic() - float(meta.get("t0", time.monotonic()))
            self._gazebo_timing_emit(
                f"launch#{meta.get('id', 0)} prewarm queued targets={','.join(sorted(targets))} "
                f"launch_elapsed={launch_elapsed:.2f}s"
            )

    @staticmethod
    def _kill_stale_gazebo_helpers() -> None:
        """Best-effort cleanup for helper nodes that can survive ros2 launch shutdown."""
        for cmd in [
            "pkill -9 -f auto_link_attacher_node.py 2>/dev/null",
            "pkill -9 -f gazebo_camera_detector.py 2>/dev/null",
        ]:
            subprocess.run(["bash", "-c", cmd], capture_output=True)

    @staticmethod
    def _force_kill_gazebo_core(reason: str = "unspecified") -> None:
        """Hard-kill gzserver/gzclient and clean DDS shared memory.

        Called after stopping a Gazebo simulation and before starting a new one
        so that a surviving gzserver from a previous session does not hold port
        11345 or stale shared memory that would prevent the new simulation from
        spawning robots.
        """
        log.info("ROS2/Gazebo hard cleanup requested reason=%s", reason)
        for cmd in [
            "killall -9 gzserver gzclient 2>/dev/null",
            "pkill -9 -f gazebo 2>/dev/null",
            "pkill -9 -f spawn_entity.py 2>/dev/null",
            "pkill -9 -f 'spawner' 2>/dev/null",
        ]:
            subprocess.run(["bash", "-c", cmd], capture_output=True)
        for pattern in [
            "/dev/shm/fastrtps_*",
            "/dev/shm/cyclonedds_*",
            "/tmp/gazebo-*",
            "/tmp/.gazebo-*",
        ]:
            subprocess.run(
                ["bash", "-c", f"rm -rf {pattern} 2>/dev/null"],
                capture_output=True,
                timeout=5,
            )
        time.sleep(2)

    def ros2_start(self, name: str) -> str | None:
        """Start a ROS2 process by name. Returns error string or None on success."""
        if name not in self.ROS2_LAUNCH_CMDS:
            return f"Unknown process: {name}"
        if self.ros2_proc_status(name) == "running":
            return f"{name} is already running"
        prereq_err = self._ros2_launch_prereq_error(name)
        if prereq_err:
            return prereq_err
        if name == "perception":
            backend = str(
                os.environ.get("PERCEPTION_BACKEND", self._perception_backend_for_mode())
            ).strip().lower()
            if backend != "gazebo_gt":
                return (
                    "Perception ROS2 process is simulation-only (gazebo_gt). "
                    "Physical backend 'yolo' runs in-app under "
                    "cais_spade_llm/resources/sensor/physical (no ROS2 node)."
                )

        # Keep simulation and hardware stacks mutually exclusive.
        if name in self._HARDWARE_PROCESS_NAMES and self._any_running(self._GAZEBO_PROCESS_NAMES):
            return "Hardware stack blocked: Gazebo is running. Stop Gazebo first."
        if name in self._GAZEBO_PROCESS_NAMES and self._any_running(self._HARDWARE_PROCESS_NAMES):
            return "Gazebo stack blocked: hardware processes are running. Stop hardware first."

        # Switching stack/environment should restart teleop backend on next command.
        if name in self._HARDWARE_PROCESS_NAMES:
            self.robot_env = "real"
            self._stop_teleop_server()
        elif name in self._GAZEBO_PROCESS_NAMES:
            self.robot_env = "gazebo"
            self._stop_teleop_server()
            # Always purge any surviving gzserver/gzclient and stale shared
            # memory before launching a new simulation.  Without this, a
            # previous session that was not cleanly stopped leaves gzserver
            # holding port 11345, causing spawn_entity.py to time out and the
            # robot to never appear in the new Gazebo window.
            self._shutdown_gazebo_prewarm_controllers()
            self._kill_stale_gazebo_helpers()
            self._force_kill_gazebo_core(reason="prelaunch_restart")

        cmd = self._ROS2_ENV + self._render_ros2_launch_cmd(name)
        try:
            proc = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,  # New process group for clean shutdown.
            )
            self._ros2_procs[name] = proc
            log.info("Started ROS2 process %s (pid=%d)", name, proc.pid)
            if name in self._GAZEBO_PROCESS_NAMES:
                self._begin_gazebo_launch_timing(name, proc.pid)
                self._queue_gazebo_prewarm(name)
            return None
        except Exception as exc:
            return str(exc)

    def ros2_stop(self, name: str, *, reason: str = "explicit_stop") -> str | None:
        """Stop a tracked ROS2 process. Returns error string or None on success."""
        proc = self._ros2_procs.get(name)
        if proc is None or proc.poll() is not None:
            self._ros2_procs.pop(name, None)
            if name in self._GAZEBO_PROCESS_NAMES and not self._any_running(self._GAZEBO_PROCESS_NAMES):
                self._shutdown_gazebo_prewarm_controllers()
                self._kill_stale_gazebo_helpers()
            return None
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=3)
            log.info("Stopped ROS2 process %s", name)
        except Exception as exc:
            log.warning("Error stopping %s: %s", name, exc)
        self._ros2_procs.pop(name, None)
        if name in self._GAZEBO_PROCESS_NAMES and not self._any_running(self._GAZEBO_PROCESS_NAMES):
            self._shutdown_gazebo_prewarm_controllers()
            self._kill_stale_gazebo_helpers()
            self._force_kill_gazebo_core(reason=reason)
        return None

    def ros2_stop_all(self, *, reason: str = "explicit_stop") -> None:
        """Stop all tracked ROS2 processes."""
        self._stop_teleop_server()
        for name in list(self._ros2_procs):
            self.ros2_stop(name, reason=reason)
        self._shutdown_gazebo_prewarm_controllers()
        self._kill_stale_gazebo_helpers()
        self._force_kill_gazebo_core(reason=reason)

    def ros2_kill_gazebo(self) -> None:
        """Kill any orphan Gazebo / ROS2 processes (cleanup helper)."""
        self._stop_teleop_server()
        self._shutdown_gazebo_prewarm_controllers()
        log.info("ROS2/Gazebo hard cleanup requested reason=explicit_cleanup")
        for cmd in [
            (
                "killall -9 gzserver gzclient robot_state_publisher spawner spawn_entity.py "
                "rviz2 move_group static_transform_publisher ros2 2>/dev/null"
            ),
            "pkill -9 -f gazebo 2>/dev/null",
            "pkill -9 -f keyboard_teleop.py 2>/dev/null",
        ]:
            subprocess.run(["bash", "-c", cmd], capture_output=True)
        self._kill_stale_gazebo_helpers()
        self._cleanup_shm_and_tmp()

    def ros2_cleanup_processes(self) -> None:
        """Aggressively clean stale ROS2/MoveIt/driver processes without killing the UI."""
        self._stop_teleop_server()
        self._shutdown_gazebo_prewarm_controllers()
        for name in list(self._ros2_procs):
            self.ros2_stop(name, reason="explicit_cleanup")

        # Kill common stale processes that often block hardware reconnection.
        log.info("ROS2/Gazebo hard cleanup requested reason=explicit_cleanup")
        for cmd in [
            "killall -9 gzserver gzclient 2>/dev/null",
            "killall -9 move_group rviz2 robot_state_publisher joint_state_publisher static_transform_publisher ros2_control_node 2>/dev/null",
            "pkill -9 -f keyboard_teleop.py 2>/dev/null",
            "pkill -9 -f spawn_entity.py 2>/dev/null",
            "pkill -9 -f xarm_driver_node 2>/dev/null",
            "pkill -9 -f controller_manager 2>/dev/null",
            "pkill -9 -f 'spawner' 2>/dev/null",
            "pkill -9 -f xarm6_moveit_realmove.launch.py 2>/dev/null",
            "pkill -9 -f ur_moveit.launch.py 2>/dev/null",
            "pkill -9 -f ur_robot_driver 2>/dev/null",
            "pkill -9 -f gazebo 2>/dev/null",
        ]:
            subprocess.run(["bash", "-c", cmd], capture_output=True)
        self._kill_stale_gazebo_helpers()
        self._cleanup_shm_and_tmp()

        self._ros2_procs = {k: p for k, p in self._ros2_procs.items() if p.poll() is None}

    @staticmethod
    def _cleanup_shm_and_tmp() -> None:
        """Remove stale DDS shared-memory and Gazebo temp/lock files (WSL2)."""
        import time
        for pattern in [
            "/dev/shm/fastrtps_*",
            "/dev/shm/cyclonedds_*",
            "/tmp/gazebo-*",
            "/tmp/.gazebo-*",
        ]:
            subprocess.run(
                ["bash", "-c", f"rm -rf {pattern} 2>/dev/null"],
                capture_output=True, timeout=5,
            )
        time.sleep(3)

    @staticmethod
    def _tail_output(text: str) -> str:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[-1]

    def ros2_exec(self, command: str, timeout_sec: float = 20.0) -> tuple[bool, str]:
        """Execute a short ROS2 shell command in a sourced environment."""
        full_cmd = self._ROS2_ENV + command
        try:
            cp = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_sec)),
            )
        except subprocess.TimeoutExpired:
            return False, f"command timed out after {timeout_sec:.0f}s"

        if cp.returncode != 0:
            detail = self._tail_output(cp.stderr) or self._tail_output(cp.stdout)
            if detail:
                return False, detail
            return False, f"command failed (exit {cp.returncode})"

        return True, self._tail_output(cp.stdout) or "OK"

    def reset_plan_runtime_state(self) -> tuple[bool, str]:
        """
        Reset runtime plan/product/resource snapshots kept under monitor/.

        This archives current monitor files (plan/state/history/debug) so the
        next run starts from a clean runtime state.
        """
        if self.system_running or self._starting:
            return (
                False,
                "Stop the agent system before resetting plan/runtime state.",
            )
        if self._stopping:
            return False, "System stop is in progress. Wait before reset."

        self._clear_cached_plan_safety_alerts()
        counts = self._archive_monitors()
        total = sum(int(v) for v in counts.values()) if isinstance(counts, dict) else 0
        if total <= 0:
            return True, "Plan/runtime state already clean."
        return True, f"Plan/runtime state reset ({total} file(s) archived)."

    def _load_gazebo_reset_model_poses(self) -> dict[str, tuple[float, float, float, float, float, float]]:
        if self._gazebo_reset_pose_cache is not None:
            return dict(self._gazebo_reset_pose_cache)
        if not _GAZEBO_WORLD_FILE.exists():
            return {}

        poses: dict[str, tuple[float, float, float, float, float, float]] = {}
        try:
            root = ET.parse(_GAZEBO_WORLD_FILE).getroot()
            for model in root.findall(".//world/model"):
                name = str(model.get("name", "")).strip()
                if not name or not any(name.startswith(prefix) for prefix in _RESETTABLE_GAZEBO_MODEL_PREFIXES):
                    continue
                pose_text = str(model.findtext("pose", default="")).strip()
                if not pose_text:
                    continue
                parts = [float(v) for v in pose_text.split()]
                if len(parts) != 6:
                    continue
                poses[name] = tuple(parts)  # type: ignore[assignment]
        except Exception:
            log.exception("Failed parsing Gazebo world reset poses from %s", _GAZEBO_WORLD_FILE)
            poses = {}

        self._gazebo_reset_pose_cache = dict(poses)
        return poses

    @staticmethod
    def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def _acquire_gazebo_reset_controller(self, robot: str) -> tuple[Any | None, bool, str]:
        robot_key = str(robot or "").strip().lower()
        with self._gazebo_prewarm_lock:
            existing = self._gazebo_prewarm_controllers.get(robot_key)
        if existing is not None:
            try:
                if bool(getattr(existing, "is_usable", lambda: False)()):
                    return existing, False, ""
            except Exception:
                pass
            with self._gazebo_prewarm_lock:
                self._gazebo_prewarm_controllers.pop(robot_key, None)
            try:
                existing.shutdown()
            except Exception:
                pass

        settings = self._load_gazebo_controller_settings(robot_key)
        if settings is None:
            return None, False, f"{robot_key}: controller config unavailable"
        controller_cfg, named_positions = settings

        try:
            from cais_spade_llm.resources.robot.ros2_pick_place_controller import (
                Ros2PickPlaceController,
            )
            from cais_spade_llm.resources.robot.ur5e_controller import (
                JOINT_NAMES as UR5E_JOINT_NAMES,
                JOINT_STATES_TOPIC as UR5E_JOINT_STATES_TOPIC,
            )
            from cais_spade_llm.resources.robot.xarm6_controller import (
                JOINT_NAMES as XARM6_JOINT_NAMES,
                JOINT_STATES_TOPIC as XARM6_JOINT_STATES_TOPIC,
            )
        except Exception as exc:
            return None, False, f"{robot_key}: controller import failed ({exc})"

        try:
            if robot_key == "xarm6":
                controller = Ros2PickPlaceController(
                    robot_name="xarm6",
                    node_name=f"xarm6_reset_controller_{os.getpid()}_{int(time.monotonic() * 1000) % 1000000}",
                    controller_config=controller_cfg,
                    named_positions=named_positions,
                    execution_mode="simulation",
                    arm_joint_names=XARM6_JOINT_NAMES,
                    arm_trajectory_topic=None,
                    joint_states_topic=XARM6_JOINT_STATES_TOPIC,
                )
            elif robot_key == "ur5e":
                controller = Ros2PickPlaceController(
                    robot_name="ur5e",
                    node_name=f"ur5e_reset_controller_{os.getpid()}_{int(time.monotonic() * 1000) % 1000000}",
                    controller_config=controller_cfg,
                    named_positions=named_positions,
                    execution_mode="simulation",
                    arm_joint_names=UR5E_JOINT_NAMES,
                    arm_trajectory_topic=_UR5E_GAZEBO_ARM_TRAJECTORY_TOPIC,
                    joint_states_topic=UR5E_JOINT_STATES_TOPIC,
                )
            else:
                return None, False, f"unknown robot: {robot_key}"
        except Exception as exc:
            return None, False, f"{robot_key}: controller init failed ({exc})"

        if not controller.wait_for_services(timeout_sec=15.0):
            detail = getattr(controller, "_last_failure_message", "") or "services not ready"
            try:
                controller.shutdown()
            except Exception:
                pass
            return None, False, f"{robot_key}: {detail}"
        return controller, True, ""

    def _restore_gazebo_scene_in_place(self) -> tuple[list[str], list[str]]:
        success_messages: list[str] = []
        warning_messages: list[str] = []
        part_poses = self._load_gazebo_reset_model_poses()
        part_names = sorted(part_poses.keys())

        controllers: list[tuple[str, Any, bool]] = []
        try:
            for robot_key in ("xarm6", "ur5e"):
                controller, owned, err = self._acquire_gazebo_reset_controller(robot_key)
                if controller is None:
                    warning_messages.append(err or f"{robot_key}: controller unavailable")
                    continue
                controllers.append((robot_key, controller, owned))

            primary_controller = controllers[0][1] if controllers else None

            for robot_key, controller, _owned in controllers:
                try:
                    try:
                        controller.open_gripper()
                    except Exception:
                        pass

                    detached_count = 0
                    for model_name in part_names:
                        try:
                            out = controller.detach_model(model_name, quiet=True)
                            if bool(out.get("success", False)):
                                detached_count += 1
                        except Exception:
                            continue
                    if detached_count > 0:
                        success_messages.append(f"{robot_key}: detached {detached_count} part(s)")

                    home_out = controller.move_home()
                    if bool(home_out.get("success", False)):
                        success_messages.append(f"{robot_key}: moved home")
                    else:
                        warning_messages.append(
                            f"{robot_key}: {str(home_out.get('message', 'move_home failed')).strip()}"
                        )
                except Exception as exc:
                    warning_messages.append(f"{robot_key}: reset failed ({exc})")

            if primary_controller and part_poses:
                restored = 0
                for model_name, pose in sorted(part_poses.items()):
                    x, y, z, roll, pitch, yaw = pose
                    qx, qy, qz, qw = self._quaternion_from_rpy(roll, pitch, yaw)
                    out = primary_controller.set_entity_pose(
                        model_name,
                        x=x,
                        y=y,
                        z=z,
                        qx=qx,
                        qy=qy,
                        qz=qz,
                        qw=qw,
                    )
                    if bool(out.get("success", False)):
                        restored += 1
                    else:
                        warning_messages.append(
                            f"{model_name}: {str(out.get('message', 'set_entity_pose failed')).strip()}"
                        )
                if restored > 0:
                    success_messages.append(f"restored {restored} part pose(s)")
            elif part_poses:
                warning_messages.append("No Gazebo controller available to restore part poses.")
        finally:
            for _robot_key, controller, owned in controllers:
                if not owned:
                    continue
                try:
                    controller.shutdown()
                except Exception:
                    pass

        return success_messages, warning_messages

    def ros2_reset_gazebo_environment(self) -> tuple[bool, str]:
        """
        Reset Gazebo to a clean initial scene.

        Sequence:
        1. Detach any gripped parts and open grippers.
        2. Move both robots home via MoveIt (before the world reset so
           the controllers and MoveIt state are still in sync).
        3. Call /reset_world to reset Gazebo physics/sim time.
        4. Restore loose parts to their initial poses.
        """
        if not self._any_running(self._GAZEBO_PROCESS_NAMES):
            return False, "Gazebo is not running."
        if self.system_running or self._starting:
            return (
                False,
                "Stop the agent system before resetting Gazebo to avoid state mismatch.",
            )

        # ── Discover the Gazebo reset service ──────────────────────
        ok, out = self._ros2_command_output(
            "ros2 service list",
            timeout_sec=5.0,
            emit_slow_diag=False,
            emit_failure_diag=False,
            emit_timeout_diag=False,
        )
        if not ok:
            return False, f"Failed to query ROS services: {out}"

        services = [line.strip() for line in out.splitlines() if line.strip()]
        candidates = (
            "/reset_world",
            "/gazebo/reset_world",
            "/reset_simulation",
            "/gazebo/reset_simulation",
        )
        selected = next(
            (
                svc
                for svc in candidates
                if any(name == svc or name.endswith(svc) for name in services)
            ),
            None,
        )
        if not selected:
            return (
                False,
                "No Gazebo reset service found (expected /reset_world or /reset_simulation).",
            )

        # ── Step 1+2: Detach parts, open grippers, move robots home ──
        success_messages, warning_messages = self._restore_gazebo_scene_in_place()

        # ── Step 3: Reset the Gazebo world ─────────────────────────
        call_ok, call_msg = self.ros2_exec(
            f'ros2 service call {selected} std_srvs/srv/Empty "{{}}"',
            timeout_sec=10.0,
        )
        if not call_ok:
            warning_messages.append(f"Gazebo reset failed via {selected}: {call_msg}")

        headline = f"Gazebo environment reset via {selected}."
        if warning_messages:
            detail = " ; ".join([headline] + success_messages + warning_messages)
            return False, detail
        detail = " ; ".join([headline] + success_messages) if success_messages else headline
        return True, detail

    def _stop_teleop_server_locked(self) -> None:
        proc = self._teleop_server_proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=2)
        except Exception:
            pass
        self._teleop_server_proc = None

    def _stop_teleop_server(self) -> None:
        with self._teleop_server_lock:
            self._stop_teleop_server_locked()

    def _read_teleop_response_locked(self, timeout_sec: float) -> tuple[bool, str, dict[str, Any]]:
        proc = self._teleop_server_proc
        if proc is None or proc.stdout is None:
            return False, "teleop server not running", {}
        if proc.poll() is not None:
            code = proc.returncode
            self._stop_teleop_server_locked()
            return False, f"teleop server exited ({code})", {}

        ready, _, _ = select.select([proc.stdout.fileno()], [], [], max(0.1, timeout_sec))
        if not ready:
            return False, f"teleop response timeout after {timeout_sec:.1f}s", {}

        line = proc.stdout.readline()
        if not line:
            self._stop_teleop_server_locked()
            return False, "teleop server closed stdout", {}
        try:
            payload = json.loads(line)
        except Exception:
            return False, f"invalid teleop response: {line.strip()}", {}

        ok = bool(payload.get("ok"))
        msg = str(payload.get("msg", "")).strip()
        return ok, (msg or ("OK" if ok else "teleop command failed")), payload

    def _ensure_teleop_server_locked(self) -> str | None:
        proc = self._teleop_server_proc
        if proc is not None and proc.poll() is None:
            return None
        self._stop_teleop_server_locked()

        quoted_script = shlex.quote(self._TELEOP_SCRIPT)
        cmd = (
            f"python3.10 {quoted_script} "
            "--server "
            "--service-timeout-sec 8 "
            "--tf-warmup-sec 0.2"
        )
        try:
            proc = subprocess.Popen(
                ["bash", "-c", self._ROS2_ENV + cmd],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )
        except Exception as exc:
            return f"failed to start teleop server: {exc}"

        self._teleop_server_proc = proc
        ok, msg, _payload = self._read_teleop_response_locked(timeout_sec=25.0)
        if not ok:
            self._stop_teleop_server_locked()
            return msg
        if msg.lower() != "ready":
            self._stop_teleop_server_locked()
            return f"unexpected teleop startup response: {msg}"
        return None

    def _teleop_request_payload(
        self,
        payload: dict[str, Any],
        timeout_sec: float,
    ) -> tuple[bool, str, dict[str, Any]]:
        with self._teleop_server_lock:
            err = self._ensure_teleop_server_locked()
            if err:
                return False, err, {}

            proc = self._teleop_server_proc
            if proc is None or proc.stdin is None:
                return False, "teleop server stdin unavailable", {}

            try:
                proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                proc.stdin.flush()
            except Exception as exc:
                self._stop_teleop_server_locked()
                return False, f"failed to send teleop command: {exc}", {}

            ok, msg, response_payload = self._read_teleop_response_locked(timeout_sec=timeout_sec)
            if not ok and (self._teleop_server_proc is None or self._teleop_server_proc.poll() is not None):
                # One transparent restart/retry for crashed backend.
                err = self._ensure_teleop_server_locked()
                if err:
                    return False, err, {}
                proc = self._teleop_server_proc
                if proc is None or proc.stdin is None:
                    return False, "teleop server unavailable after restart", {}
                proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                return self._read_teleop_response_locked(timeout_sec=timeout_sec)
            return ok, msg, response_payload

    def _teleop_request(self, payload: dict[str, Any], timeout_sec: float) -> tuple[bool, str]:
        ok, msg, _payload = self._teleop_request_payload(payload=payload, timeout_sec=timeout_sec)
        return ok, msg

    def _run_teleop_once(self, args: list[str], timeout_sec: float) -> tuple[bool, str]:
        quoted_script = shlex.quote(self._TELEOP_SCRIPT)
        quoted_args = " ".join(shlex.quote(str(arg)) for arg in args)
        cmd = f"python3.10 {quoted_script} {quoted_args}".strip()
        return self.ros2_exec(cmd, timeout_sec=timeout_sec)

    def teleop_jog(
        self,
        robot: str,
        axis: str,
        step_mm: float,
        velocity_scale: float = 1.0,
    ) -> tuple[bool, str]:
        robot = str(robot).strip().lower()
        axis = str(axis).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}"
        if axis not in {"x", "y", "z"}:
            return False, f"unknown axis: {axis}"
        return self._teleop_request(
            payload={
                "op": "cartesian",
                "robot": robot,
                "axis": axis,
                "step_mm": float(step_mm),
                "velocity_scale": float(velocity_scale),
            },
            timeout_sec=12.0,
        )

    def teleop_gripper(
        self,
        robot: str,
        action: str,
        step: float | None = None,
        velocity_scale: float = 1.0,
    ) -> tuple[bool, str]:
        robot = str(robot).strip().lower()
        action = str(action).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}"
        if action not in {"open", "close"}:
            return False, f"unknown gripper action: {action}"
        # Fast path: persistent backend for low-latency repeated clicks.
        payload: dict[str, Any] = {
            "op": "gripper",
            "robot": robot,
            "action": action,
            "velocity_scale": float(velocity_scale),
        }
        if step is not None:
            payload["step"] = float(step)
        ok, msg = self._teleop_request(payload=payload, timeout_sec=3.0)
        if ok:
            return True, msg

        # Fallback to one-shot command only for backend/channel failures.
        transient_markers = (
            "teleop server",
            "stdin unavailable",
            "failed to send",
            "timeout",
            "exited",
            "closed stdout",
        )
        if not any(marker in msg.lower() for marker in transient_markers):
            return False, msg

        once_args = [
            "--once",
            "gripper",
            "--robot",
            robot,
            "--gripper-action",
            action,
            "--service-timeout-sec",
            "6",
            "--tf-warmup-sec",
            "0.1",
        ]
        if step is not None:
            once_args += ["--gripper-step", str(float(step))]
        return self._run_teleop_once(once_args, timeout_sec=10.0)

    def teleop_home(self, robot: str) -> tuple[bool, str]:
        robot = str(robot).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}"
        return self._teleop_request(
            payload={
                "op": "home",
                "robot": robot,
            },
            timeout_sec=25.0,
        )

    def list_named_positions(self, robot: str) -> dict[str, list[float]]:
        """Return named positions for *robot* in the current environment."""
        robot = str(robot).strip().lower()
        env = self.teleop_target_environment()
        # Map "real" → JSON key "real", "gazebo" → "gazebo"
        env_key = env if env in ("gazebo", "real") else "gazebo"
        path = _RESOURCE_DIR / f"robot_{robot}.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            robot_block = data.get(robot, {})
            return dict(robot_block.get(env_key, {}).get("named_positions", {}))
        except Exception:
            log.exception("Failed to read named positions for %s", robot)
            return {}

    def teleop_go_to_position(self, robot: str, name: str) -> tuple[bool, str]:
        """Move *robot* to a stored named position by sending joint targets."""
        robot = str(robot).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}"
        positions = self.list_named_positions(robot)
        joints = positions.get(name)
        if not joints or not isinstance(joints, list):
            return False, f"named position '{name}' not found for {robot}"
        return self._teleop_request(
            payload={
                "op": "move_joints",
                "robot": robot,
                "positions": [float(j) for j in joints],
            },
            timeout_sec=25.0,
        )

    def teleop_joint(
        self,
        robot: str,
        joint_idx: int,
        delta_deg: float,
        velocity_scale: float = 1.0,
    ) -> tuple[bool, str]:
        robot = str(robot).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}"
        idx = int(joint_idx)
        if idx < 1 or idx > 6:
            return False, f"invalid joint index: {joint_idx}"
        return self._teleop_request(
            payload={
                "op": "joint",
                "robot": robot,
                "joint": idx,
                "delta_deg": float(delta_deg),
                "velocity_scale": float(velocity_scale),
            },
            timeout_sec=8.0,
        )

    def teleop_save_position(self, robot: str, name: str) -> tuple[bool, str]:
        robot = str(robot).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}"
        position_name = str(name).strip()
        if not position_name:
            return False, "position name is empty"
        if len(position_name) > 64:
            return False, "position name too long (max 64 chars)"
        return self._teleop_request(
            payload={
                "op": "save_position",
                "robot": robot,
                "name": position_name,
                "env": self.teleop_target_environment(),
            },
            timeout_sec=6.0,
        )

    def teleop_state(self, robot: str) -> tuple[bool, str, dict[str, Any]]:
        robot = str(robot).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}", {}
        ok, msg, payload = self._teleop_request_payload(
            payload={
                "op": "state",
                "robot": robot,
            },
            timeout_sec=4.0,
        )
        if not ok:
            return False, msg, {}
        state = payload.get("state", {})
        if not isinstance(state, dict):
            return False, "invalid state payload", {}
        return True, msg, state

    # ------------------------------------------------------------------
    # State accessors (called by UI pages via ui.timer)
    # ------------------------------------------------------------------
    def get_agent_statuses(self) -> list[dict[str, Any]]:
        statuses = []
        for a in self.resource_agents:
            statuses.append({
                "name": getattr(a, "agent_name", str(a.jid)),
                "jid": str(a.jid),
                "type": "robot",
                "alive": a.is_alive() if hasattr(a, "is_alive") else False,
            })
        for a in self.product_agents:
            statuses.append({
                "name": getattr(a, "name", str(a.jid)),
                "jid": str(a.jid),
                "type": "product",
                "alive": a.is_alive() if hasattr(a, "is_alive") else False,
            })
        if self.cca:
            statuses.append({
                "name": getattr(self.cca, "agent_name", "cca"),
                "jid": str(self.cca.jid),
                "type": "cca",
                "alive": self.cca.is_alive() if hasattr(self.cca, "is_alive") else False,
            })
        if self.user_agent:
            statuses.append({
                "name": "user",
                "jid": str(self.user_agent.jid),
                "type": "user",
                "alive": self.user_agent.is_alive() if hasattr(self.user_agent, "is_alive") else False,
            })
        return statuses

    def get_robot_states(self) -> dict[str, dict[str, Any]]:
        result = {}
        for ra in self.resource_agents:
            name = getattr(ra, "agent_name", str(ra.jid))
            if hasattr(ra, "_snapshot_state"):
                result[name] = ra._snapshot_state()
            else:
                result[name] = {"execution_mode": "unknown"}
        return result

    def get_plan_nodes(self) -> list[dict[str, Any]]:
        nodes = []
        for pa in self.product_agents:
            pp = getattr(pa, "process_planner", None)
            if pp and hasattr(pp, "nodes"):
                nodes.extend(pp.nodes)
        return nodes

    @staticmethod
    def _read_json_dict(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def get_global_fsa(self) -> dict[str, Any]:
        # Prefer in-memory FSA from currently running product agents.
        for pa in self.product_agents:
            pp = getattr(pa, "process_planner", None)
            fsa = getattr(pp, "global_fsa", None) if pp else None
            if isinstance(fsa, dict) and (fsa.get("A") or fsa.get("meta")):
                return fsa

            fsa_path = getattr(pa, "global_fsa_path", None)
            if fsa_path:
                data = self._read_json_dict(Path(fsa_path))
                if data.get("A") or data.get("meta"):
                    return data

        # If a product was selected in UI, try its expected monitor path.
        if self.selected_product:
            selected_name = Path(self.selected_product).stem
            selected_fsa_path = _MONITOR / "plan" / f"{selected_name}_global_fsa.json"
            if selected_fsa_path.exists():
                data = self._read_json_dict(selected_fsa_path)
                if data.get("A") or data.get("meta"):
                    return data

        # Fallback: newest global FSA snapshot in monitor/plan.
        plan_dir = _MONITOR / "plan"
        if plan_dir.exists():
            candidates = sorted(
                plan_dir.glob("*_global_fsa.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for path in candidates:
                data = self._read_json_dict(path)
                if data.get("A") or data.get("meta"):
                    return data

        return {}

    def get_task_states(self) -> dict[str, str]:
        merged = {}
        for pa in self.product_agents:
            planner = getattr(pa, "process_planner", None)
            nodes = getattr(planner, "nodes", []) if planner is not None else []
            for node in nodes or []:
                if not isinstance(node, dict):
                    continue
                task_id = str(node.get("id") or node.get("task_id") or "").strip()
                if not task_id:
                    continue
                merged[task_id] = str(node.get("status", "pending") or "pending")
            ts = getattr(pa, "task_states", {})
            merged.update(ts)
        return merged

    def _find_product_agent(self, product_jid: str):
        target = str(product_jid or "").strip()
        if not target:
            raise ValueError("product_jid is required")
        for pa in self.product_agents:
            if str(getattr(pa, "jid", "")).strip() == target:
                return pa
        raise ValueError(f"product agent not found: {target}")

    @staticmethod
    def _run_product_agent_coroutine(
        agent: Any,
        coroutine: Any,
        *,
        timeout_sec: float = 20.0,
        operation_name: str = "product agent request",
    ) -> Any:
        loop = getattr(agent, "loop", None)
        if loop is None:
            raise RuntimeError("product agent loop is unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=max(1.0, float(timeout_sec)))
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError(
                f"{str(operation_name or 'product agent request').strip() or 'product agent request'} "
                f"timed out after {max(1.0, float(timeout_sec)):.0f}s"
            ) from exc
        except Exception:
            future.cancel()
            raise

    def get_execution_timeline(self) -> list[dict[str, Any]]:
        timeline = []
        for pa in self.product_agents:
            tl = getattr(pa, "execution_timeline", [])
            timeline.extend(tl)
        return timeline

    def get_runtime_recoveries(self) -> list[dict[str, Any]]:
        recoveries: list[dict[str, Any]] = []
        for pa in self.product_agents:
            getter = getattr(pa, "get_runtime_recovery", None)
            if callable(getter):
                try:
                    recovery = getter()
                except Exception:
                    recovery = None
            else:
                recovery = getattr(pa, "runtime_recovery", None)
            if isinstance(recovery, dict):
                recoveries.append(dict(recovery))
        return recoveries

    def submit_runtime_recovery_guidance(self, product_jid: str, message: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        submitter = getattr(agent, "submit_runtime_recovery_guidance", None)
        if not callable(submitter):
            raise RuntimeError("product agent does not support runtime recovery guidance")
        return self._run_product_agent_coroutine(agent, submitter(message))

    def retry_runtime_recovery_des(self, product_jid: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        retry = getattr(agent, "retry_runtime_recovery_des", None)
        if not callable(retry):
            raise RuntimeError("product agent does not support DES runtime recovery retry")
        return self._run_product_agent_coroutine(agent, retry())

    def generate_runtime_bridge_proposal(self, product_jid: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        generate = getattr(agent, "generate_runtime_bridge_proposal", None)
        if not callable(generate):
            raise RuntimeError("product agent does not support runtime bridge generation")
        return self._run_product_agent_coroutine(agent, generate())

    def load_runtime_bridge_archive_proposal(
        self,
        product_jid: str,
        artifact_path: str | None = None,
    ) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        loader = getattr(agent, "load_runtime_bridge_archive_proposal", None)
        if not callable(loader):
            raise RuntimeError("product agent does not support archived runtime bridge loading")
        return self._run_product_agent_coroutine(
            agent,
            loader(artifact_path),
            timeout_sec=60.0,
            operation_name="loading archived bridge proposal",
        )

    def approve_runtime_bridge_outline(self, product_jid: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        approve = getattr(agent, "approve_runtime_bridge_outline", None)
        if not callable(approve):
            raise RuntimeError("product agent does not support outline approval")
        return self._run_product_agent_coroutine(
            agent,
            approve(),
            timeout_sec=60.0,
            operation_name="approving outline checkpoint",
        )

    def refine_runtime_bridge_outline(self, product_jid: str, feedback: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        refine = getattr(agent, "refine_runtime_bridge_outline", None)
        if not callable(refine):
            raise RuntimeError("product agent does not support outline refinement")
        return self._run_product_agent_coroutine(
            agent,
            refine(feedback),
            timeout_sec=60.0,
            operation_name="refining outline checkpoint",
        )

    def reject_runtime_bridge_outline(self, product_jid: str, feedback: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        reject = getattr(agent, "reject_runtime_bridge_outline", None)
        if not callable(reject):
            raise RuntimeError("product agent does not support outline rejection")
        return self._run_product_agent_coroutine(agent, reject(feedback))

    def approve_runtime_bridge_primitives(self, product_jid: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        approve = getattr(agent, "approve_runtime_bridge_primitives", None)
        if not callable(approve):
            raise RuntimeError("product agent does not support primitive approval")
        return self._run_product_agent_coroutine(
            agent,
            approve(),
            timeout_sec=60.0,
            operation_name="approving primitive checkpoint",
        )

    def refine_runtime_bridge_primitives(self, product_jid: str, feedback: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        refine = getattr(agent, "refine_runtime_bridge_primitives", None)
        if not callable(refine):
            raise RuntimeError("product agent does not support primitive refinement")
        return self._run_product_agent_coroutine(
            agent,
            refine(feedback),
            timeout_sec=60.0,
            operation_name="refining primitive checkpoint",
        )

    def reject_runtime_bridge_primitives(self, product_jid: str, feedback: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        reject = getattr(agent, "reject_runtime_bridge_primitives", None)
        if not callable(reject):
            raise RuntimeError("product agent does not support primitive rejection")
        return self._run_product_agent_coroutine(agent, reject(feedback))

    def load_preprogrammed_runtime_bridge_scenario(
        self,
        product_jid: str,
        scenario_id: str,
    ) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        loader = getattr(agent, "load_preprogrammed_runtime_bridge_scenario", None)
        loader_sync = getattr(agent, "load_preprogrammed_runtime_bridge_scenario_sync", None)
        if not callable(loader):
            if not callable(loader_sync):
                raise RuntimeError("product agent does not support preprogrammed runtime bridge scenarios")
        log.info(
            "[ui.bridge] Loading preprogrammed runtime bridge scenario product=%s scenario=%s",
            product_jid,
            scenario_id,
        )
        if callable(loader_sync):
            log.info(
                "[ui.bridge] Using direct sync path for preprogrammed runtime bridge scenario product=%s scenario=%s",
                product_jid,
                scenario_id,
            )
            return loader_sync(scenario_id)
        return self._run_product_agent_coroutine(
            agent,
            loader(scenario_id),
            timeout_sec=60.0,
            operation_name="loading preprogrammed recovery scenario",
        )

    def approve_runtime_bridge_proposal(self, product_jid: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        approve = getattr(agent, "approve_runtime_bridge_proposal", None)
        approve_sync = getattr(agent, "approve_runtime_bridge_proposal_sync", None)
        if not callable(approve):
            if not callable(approve_sync):
                raise RuntimeError("product agent does not support runtime bridge approval")
        log.info("[ui.bridge] Approving runtime bridge proposal product=%s", product_jid)
        if callable(approve_sync):
            log.info(
                "[ui.bridge] Using direct sync path for runtime bridge approval product=%s",
                product_jid,
            )
            return approve_sync()
        return self._run_product_agent_coroutine(
            agent,
            approve(),
            timeout_sec=60.0,
            operation_name="approving bridge proposal",
        )

    def reject_runtime_bridge_proposal(self, product_jid: str, feedback: str) -> dict[str, Any]:
        if not self.system_running:
            raise RuntimeError("system is not running")
        agent = self._find_product_agent(product_jid)
        reject = getattr(agent, "reject_runtime_bridge_proposal", None)
        if not callable(reject):
            raise RuntimeError("product agent does not support runtime bridge rejection")
        return self._run_product_agent_coroutine(agent, reject(feedback))

    def get_plan_safety_alerts(self) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        for pa in self.product_agents:
            getter = getattr(pa, "get_plan_safety_alert", None)
            if callable(getter):
                try:
                    alert = getter()
                except Exception:
                    alert = None
            else:
                alert = getattr(pa, "plan_safety_alert", None)
            if isinstance(alert, dict) and alert:
                alerts.append(dict(alert))
        if alerts:
            return alerts
        return [dict(a) for a in self._cached_plan_safety_alerts]

    def get_part_tracker(self) -> dict[str, dict[str, Any]]:
        merged = {}
        for pa in self.product_agents:
            pt = getattr(pa, "part_tracker", {})
            merged.update(pt)
        return merged

    def get_safety_rules(self, bundle_id: str | None = None) -> list[dict[str, Any]]:
        """
        Return safety rules currently enforced at runtime when available.

        When the system is not running yet, fall back to the selected or active
        verified bundle's compiled safety rules so the dashboard can preview the
        exact rules that will be enforced after startup.
        """
        if self.cca and hasattr(self.cca, "safety_rules"):
            live_rules = self.cca.safety_rules or []
            if live_rules:
                normalized: list[dict[str, Any]] = []
                for rule in live_rules:
                    if not isinstance(rule, dict):
                        continue
                    item = dict(rule)
                    interpretation = str(item.get("generated_interpretation", "")).strip()
                    if not interpretation:
                        interpretation = self._ltlf_plain_feedback(item)
                    item["generated_interpretation"] = interpretation
                    normalized.append(item)
                return normalized
        target_bundle_id = str(bundle_id or "").strip() or str(self.bundle_store.get_active_bundle_id() or "").strip()
        if target_bundle_id:
            return self.get_bundle_safety_rules(target_bundle_id)
        return []

    def get_safety_state(self) -> dict[str, Any]:
        if not self.cca:
            alerts = self.get_plan_safety_alerts()
            recoveries = self.get_runtime_recoveries()
            result = {"plan_alerts": alerts} if alerts else {}
            if recoveries:
                result["runtime_recoveries"] = recoveries
            return result
        result: dict[str, Any] = {}
        sm = getattr(self.cca, "safety_monitor", None)
        if sm:
            result["dfa_states"] = getattr(sm, "current_states", {})
            result["running_aps"] = list(getattr(sm, "running_aps", set()))
        fm = getattr(self.cca, "plan_fsa_monitor", None)
        if fm:
            result["fsa_state"] = getattr(fm, "current_state", None)
            result["fsa_completed"] = list(getattr(fm, "completed_tasks", []))
        result["blocked_tasks"] = getattr(self.cca, "blocked_tasks", {})
        result["plan_alerts"] = self.get_plan_safety_alerts()
        result["runtime_recoveries"] = self.get_runtime_recoveries()
        return result

    def get_log_paths(self) -> dict[str, str]:
        paths = {}
        if _LOG_DIR.exists():
            for p in sorted(_LOG_DIR.glob("*.log")):
                paths[p.stem] = str(p)
        return paths
