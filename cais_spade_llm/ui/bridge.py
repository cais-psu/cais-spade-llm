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
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_artifacts import (
    DEFAULT_BRIDGE_RUNTIME_DATA_DIR,
)
from cais_spade_llm.bundles import BundleCompiler, BundleStore
from cais_spade_llm.bundles.models import (
    BUNDLE_STATUS_DRAFT,
    BUNDLE_STATUS_INVALID,
    BUNDLE_STATUS_STALE,
    BUNDLE_STATUS_VERIFIED,
    atomic_json_write,
    sha256_file,
    sha256_text,
    slug,
)
from cais_spade_llm.product.order import (
    derive_ordering_constraints_from_safety,
    geometry_slot_names,
    load_product_order_file,
    validate_product_order,
)
from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.ui import digital_twin

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
_PRODUCT_ORDERS_DIR = _BASE / "specification" / "products" / "orders"
_SAFETY_REQUIREMENTS_DIR = _BASE / "specification" / "safety"
_XARM6_RESOURCE = _RESOURCE_DIR / "robot_xarm6.json"
_UR5E_RESOURCE = _RESOURCE_DIR / "robot_ur5e.json"
_UR5E_GAZEBO_ARM_TRAJECTORY_TOPIC = "/ur5e_joint_trajectory_controller/joint_trajectory"
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_UR5E_RG2_GRIPPER_SCRIPT = _PROJECT_ROOT / "ros2" / "cais_lab_gazebo" / "scripts" / "ur5e_rg2_rtde_gripper.py"
_UR5E_RTDE_TRAJECTORY_SCRIPT = _PROJECT_ROOT / "ros2" / "cais_lab_gazebo" / "scripts" / "ur5e_rtde_trajectory_server.py"
_UR5E_RTDE_TRAJECTORY_ACTION = "/cais_ur5e_rtde_trajectory_controller/follow_joint_trajectory"
_UR5E_RTDE_TRAJECTORY_STATUS = Path("/tmp") / "cais_ur5e_rtde_trajectory_status.json"
_UR5E_RG2_GRIPPER_ACTION = "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory"
_UR5E_RG2_GRIPPER_STATUS = Path("/tmp") / "cais_ur5e_rg2_gripper_status.json"
_USER_VERIFIED_PLAN = _BASE / "user_verified_plan"
_USER_VERIFIED_SAFETY = _BASE / "user_verified_safety"
_SAFETY_INTENT_APPROVALS = _USER_VERIFIED_SAFETY / "intent_approvals.json"
_SAFETY_INTENT_PREVIEWS = _USER_VERIFIED_SAFETY / "intent_previews.json"
_SAFETY_PREVIEW_DIR = _USER_VERIFIED_SAFETY / "previews"
_SAFETY_VERIFIED_DIR = _USER_VERIFIED_SAFETY / "verified"
_SAFETY_PREVIEW_HISTORY_LIMIT = 10
_GAZEBO_WORLD_FILE = _PROJECT_ROOT / "ros2" / "cais_lab_gazebo" / "worlds" / "table.world"
_RESETTABLE_GAZEBO_MODEL_PREFIXES = ("gear_", "rect_pin_", "circ_pin_")
_ROBOT_TAUGHT_FUNCTIONS_DIR = _BASE / "resources" / "robot" / "taught_functions"


def _legacy_safety_intent_approvals_path() -> Path:
    return _SAFETY_REQUIREMENTS_DIR / "intent_approvals.json"


def _legacy_safety_intent_previews_path() -> Path:
    return _SAFETY_REQUIREMENTS_DIR / "intent_previews.json"


def _legacy_safety_preview_dir() -> Path:
    return _USER_VERIFIED_PLAN / "safety_previews"


class SystemBridge:
    """Singleton that owns the SPADE lifecycle and exposes agent state to the UI."""

    _instance: SystemBridge | None = None
    _HW_IP_DEFAULTS = {
        "xarm6": "192.168.1.240",
        "ur5e": "192.168.1.172",
    }
    _BASE_GAZEBO_PROCESS_NAMES = {"gazebo_dual", "gazebo_xarm6", "gazebo_ur5e"}
    _BASE_HARDWARE_PROCESS_NAMES = {
        "hardware_xarm6_driver",
        "hardware_xarm6_moveit",
        "hardware_ur5e_rtde_trajectory_server",
        "hardware_ur5e_rg2_gripper",
        "hardware_ur5e_moveit",
    }
    _DIGITAL_TWIN_GAZEBO_PROCESS_NAMES = {
        "digital_twin_xarm_only_gazebo",
        "digital_twin_ur5e_only_gazebo",
        "digital_twin_dual_robots_gazebo",
        "digital_twin_dual_robots_gazebo_moveit",
    }
    _DIGITAL_TWIN_HARDWARE_PROCESS_NAMES = {
        "digital_twin_xarm_only_hardware_xarm6_moveit",
        "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server",
        "digital_twin_ur5e_only_hardware_ur5e_rg2_gripper",
        "digital_twin_ur5e_only_hardware_ur5e_moveit",
        "digital_twin_dual_robots_hardware_xarm6_driver",
        "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server",
        "digital_twin_dual_robots_hardware_ur5e_rg2_gripper",
        "digital_twin_dual_robots_hardware_moveit",
    }
    _DIGITAL_TWIN_SYNC_PROCESS_NAMES = {
        "digital_twin_xarm_only_sync",
        "digital_twin_ur5e_only_sync",
        "digital_twin_dual_robots_sync_xarm6",
        "digital_twin_dual_robots_sync_ur5e",
    }
    _DIGITAL_TWIN_MARKER_PROCESS_NAMES = {
        "digital_twin_dual_robots_paired_markers",
    }
    _DIGITAL_TWIN_PROCESS_NAMES = (
        _DIGITAL_TWIN_GAZEBO_PROCESS_NAMES
        | _DIGITAL_TWIN_HARDWARE_PROCESS_NAMES
        | _DIGITAL_TWIN_SYNC_PROCESS_NAMES
        | _DIGITAL_TWIN_MARKER_PROCESS_NAMES
    )
    _GAZEBO_PROCESS_NAMES = _BASE_GAZEBO_PROCESS_NAMES | _DIGITAL_TWIN_GAZEBO_PROCESS_NAMES
    _HARDWARE_PROCESS_NAMES = _BASE_HARDWARE_PROCESS_NAMES | _DIGITAL_TWIN_HARDWARE_PROCESS_NAMES
    _HARDWARE_STACKS = {
        # xArm6 MoveIt realmove includes UFRobotSystemHardware (embedded driver path).
        "xarm6": ("hardware_xarm6_moveit",),
        # UR5e arm execution is RTDE-only; RG2 remains a separate bridge.
        "ur5e": (
            "hardware_ur5e_rtde_trajectory_server",
            "hardware_ur5e_rg2_gripper",
            "hardware_ur5e_moveit",
        ),
    }
    _DIGITAL_TWIN_TARGETS = {
        "xarm only": {
            "slug": "xarm_only",
            "robot": "xarm6",
            "gazebo": "gazebo_xarm6_passive",
            "gazebo_launches": {
                "monitor": "gazebo_xarm6_passive",
                "teach": "gazebo_xarm6",
            },
            "gazebo_process": "digital_twin_xarm_only_gazebo",
            "hardware": ("xarm6",),
            "hardware_processes": {
                "moveit": "digital_twin_xarm_only_hardware_xarm6_moveit",
            },
            "model_name": "xarm6",
            "sync_process": "digital_twin_xarm_only_sync",
            "hardware_supported": True,
        },
        "ur5e only": {
            "slug": "ur5e_only",
            "robot": "ur5e",
            "gazebo": "gazebo_ur5e_passive",
            "gazebo_launches": {
                "monitor": "gazebo_ur5e_passive",
                "teach": "gazebo_ur5e",
            },
            "gazebo_process": "digital_twin_ur5e_only_gazebo",
            "hardware": ("ur5e",),
            "hardware_processes": {
                "rtde": "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server",
                "gripper": "digital_twin_ur5e_only_hardware_ur5e_rg2_gripper",
                "moveit": "digital_twin_ur5e_only_hardware_ur5e_moveit",
            },
            "model_name": "ur5e_rg2",
            "sync_process": "digital_twin_ur5e_only_sync",
            "hardware_supported": True,
        },
        "dual robots": {
            "slug": "dual_robots",
            "robot": "dual robots",
            "gazebo": "gazebo_dual_passive",
            "gazebo_launches": {
                "monitor": "gazebo_dual_passive",
                "teach": "gazebo_dual_gazebo_only",
            },
            "gazebo_process": "digital_twin_dual_robots_gazebo",
            "gazebo_moveit_launch": "gazebo_dual_moveit_only",
            "gazebo_moveit_process": "digital_twin_dual_robots_gazebo_moveit",
            "paired_marker_process": "digital_twin_dual_robots_paired_markers",
            "hardware": ("xarm6", "ur5e"),
            "hardware_processes": {
                "xarm6": {
                    "driver": "digital_twin_dual_robots_hardware_xarm6_driver",
                    "moveit": "digital_twin_dual_robots_hardware_moveit",
                },
                "ur5e": {
                    "rtde": "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server",
                    "gripper": "digital_twin_dual_robots_hardware_ur5e_rg2_gripper",
                    "moveit": "digital_twin_dual_robots_hardware_moveit",
                },
            },
            "model_name": "dual_robot",
            "sync_processes": {
                "xarm6": "digital_twin_dual_robots_sync_xarm6",
                "ur5e": "digital_twin_dual_robots_sync_ur5e",
            },
            "hardware_supported": True,
            "sim_modes": ("monitor",),
        },
    }
    _DIGITAL_TWIN_DIRECTIONS = ("hardware -> gazebo", "gazebo -> hardware")
    _DIGITAL_TWIN_DEFAULT_DIRECTION = "hardware -> gazebo"
    # Sim side: keep digital twin in monitor mode for now. Teach mode code is
    # intentionally left in place, but it is not exposed as an allowed mode.
    _DIGITAL_TWIN_SIM_MODES = ("monitor",)
    _DIGITAL_TWIN_DEFAULT_SIM_MODE = "monitor"
    _DIGITAL_TWIN_SIM_MODE_ALIASES = {
        "mirror": "monitor",
        "author": "teach",
    }
    _DIGITAL_TWIN_GAZEBO_DOMAIN_DEFAULT = 41
    _DIGITAL_TWIN_HARDWARE_DOMAIN_DEFAULT = 42
    _DIGITAL_TWIN_HARDWARE_XARM6_DOMAIN_DEFAULT = 42
    _DIGITAL_TWIN_HARDWARE_UR5E_DOMAIN_DEFAULT = 43
    # Absolute sanity ceiling for the first-waypoint approach (deg). The replay sizes the
    # approach by a safe joint speed; this only blocks near-180 deg deltas (encoder/wrap).
    _DIGITAL_TWIN_MAX_JOINT_DELTA_DEG = 175.0
    _DIGITAL_TWIN_REPLAY_SPEED_SCALE = 1.0
    _DIGITAL_TWIN_REPLAY_WAYPOINT_DURATION_SEC = 2.0 / _DIGITAL_TWIN_REPLAY_SPEED_SCALE
    _DIGITAL_TWIN_REPLAY_MAX_JOINT_VEL_DEG_S = 25.0 * _DIGITAL_TWIN_REPLAY_SPEED_SCALE
    _DIGITAL_TWIN_PREPARED_REPLAY_VERSION = 6
    _DIGITAL_TWIN_INITIALIZE_TIMEOUT_S = 90.0
    _DIGITAL_TWIN_MIRROR_STABILIZATION_SEC = 0.75
    # Per-robot home/initial joint pose (arm joints, radians) for the "Go Home" button.
    # xArm6 matches the dual-boot startup pose injected into the gazebo launch.
    _DIGITAL_TWIN_HOME: dict[str, list[float]] = {
        "xarm6": [-1.572631, -1.054702, -0.385494, 0.000322, 1.440603, -1.572544],
    }
    _DIGITAL_TWIN_SYNC_RESTART_COOLDOWN_S = 8.0
    _ROBOT_FUNCTION_NAMES = (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
        "move_home",
    )
    _ROBOT_FUNCTION_TEMPLATES: dict[str, list[dict[str, str]]] = {
        "pick_approach": [
            {"step_name": "approach_pose", "primitive": "move_cartesian"},
            {"step_name": "pick_pose", "primitive": "move_cartesian"},
        ],
        "pick_grasp": [
            {"step_name": "grasp_part", "primitive": "grasp_part"},
            {"step_name": "lift_pose", "primitive": "move_relative"},
        ],
        "place_approach": [
            {"step_name": "approach_pose", "primitive": "move_cartesian"},
            {"step_name": "place_pose", "primitive": "move_cartesian"},
        ],
        "place_insert": [
            {"step_name": "insert_pose", "primitive": "move_cartesian"},
            {"step_name": "release_part", "primitive": "release_part"},
            {"step_name": "retreat_pose", "primitive": "move_relative"},
        ],
        "move_home": [
            {"step_name": "home", "primitive": "move_to_named_pose"},
        ],
    }
    _ROBOT_FUNCTION_EVENT_PRIMITIVES = {
        "grasp_part",
        "release_part",
        "open_gripper",
        "close_gripper",
    }
    _DUAL_HARDWARE_LIMITATION = (
        "dual robots hardware requires merged dual-hardware MoveIt. "
        "This first digital twin launch section leaves it disabled until that ROS launch/config is implemented."
    )
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
        "gazebo_dual_gazebo_only": "dual_moveit_gazebo.launch.py",
        "gazebo_dual_moveit_only": "dual_moveit_gazebo.launch.py",
        "gazebo_dual_passive": "xarm6_ur5e_gazebo.launch.py",
        "gazebo_xarm6": "xarm6_moveit_single_gazebo.launch.py",
        "gazebo_ur5e": "ur5e_rg2_moveit_gazebo.launch.py",
        "gazebo_xarm6_passive": "xarm6_single_gazebo.launch.py",
        "gazebo_ur5e_passive": "ur5e_rg2_gazebo.launch.py",
        "hardware_ur5e_moveit": "ur5e_rg2_hardware_moveit.launch.py",
        "hardware_xarm6_driver": "xarm6_hardware_driver.launch.py",
        "hardware_dual_robots_moveit": "dual_robots_hardware_moveit.launch.py",
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
        self._xmpp_proc: subprocess.Popen | None = None
        self._xmpp_host: str = "127.0.0.1"
        self._xmpp_port: int = 5222
        self._gazebo_reset_pose_cache: dict[str, tuple[float, float, float, float, float, float]] | None = None

        # Lifecycle flags.
        self.system_running: bool = False
        self._starting: bool = False
        self._stopping: bool = False
        self.last_error: str | None = None
        self.last_notice: str | None = None
        self._safety_preview_failures: dict[str, dict[str, Any]] = {}

        # Configuration (set from UI before start).
        self.execution_mode: str = "simulation"
        self.robot_env: str = "gazebo"
        self.fast_forward_simulation_enabled: bool = False
        self.selected_product: str = ""
        self.selected_requirement_file: str = ""
        self.selected_product_order_file: str = ""
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
        self._teleop_server_proc: subprocess.Popen | None = None
        self._teleop_server_ros_domain_id: int | None = None
        self._teleop_server_lock = threading.Lock()
        self._digital_twin_sim_modes: dict[str, str] = {
            target: self._DIGITAL_TWIN_DEFAULT_SIM_MODE
            for target in self._DIGITAL_TWIN_TARGETS
        }
        # In-memory capture buffer for the manual record/replay feature (per target).
        self._digital_twin_waypoints: dict[str, list[dict[str, Any]]] = {}
        self._digital_twin_function_steps: dict[str, list[dict[str, Any]]] = {}
        self._digital_twin_record_lock = threading.Lock()
        self._digital_twin_prepare_lock = threading.Lock()
        self._digital_twin_prepare_threads: dict[str, threading.Thread] = {}
        self._digital_twin_sync_restart_threads: dict[str, threading.Thread] = {}
        self._digital_twin_sync_restart_last_attempt: dict[str, float] = {}
        self._ur5e_controller_status_cache: dict[str, dict[str, Any]] = {}
        self._ur5e_controller_auto_repair_last_attempt: dict[str, float] = {}
        self._gazebo_prewarm_lock = threading.Lock()
        self._gazebo_prewarm_thread: threading.Thread | None = None
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
        self._agent_creator_prefetch_thread: threading.Thread | None = None
        self._agent_runtime_loop: asyncio.AbstractEventLoop | None = None
        self._agent_runtime_thread: threading.Thread | None = None
        self._agent_runtime_lock = threading.Lock()
        self._safety_rule_preview_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        self._safety_rule_preview_cache_lock = threading.Lock()
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
            if artifact_path.parent.name != "recovery_final":
                recovery_final_path = artifact_path.parent / "recovery_final" / artifact_path.name
                if recovery_final_path.exists() and recovery_final_path.is_file():
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

    def _ensure_agent_runtime_loop(self) -> asyncio.AbstractEventLoop:
        """Return the dedicated SPADE runtime loop, starting it if needed."""
        with self._agent_runtime_lock:
            loop = self._agent_runtime_loop
            thread = self._agent_runtime_thread
            if (
                loop is not None
                and thread is not None
                and thread.is_alive()
                and loop.is_running()
            ):
                return loop

            ready = threading.Event()
            holder: dict[str, Any] = {}

            def _runner() -> None:
                runtime_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(runtime_loop)
                holder["loop"] = runtime_loop
                ready.set()
                try:
                    runtime_loop.run_forever()
                finally:
                    pending = [task for task in asyncio.all_tasks(runtime_loop) if not task.done()]
                    for task in pending:
                        task.cancel()
                    if pending:
                        runtime_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    runtime_loop.run_until_complete(runtime_loop.shutdown_asyncgens())
                    runtime_loop.close()

            thread = threading.Thread(
                target=_runner,
                name="cais-spade-agent-runtime",
                daemon=True,
            )
            thread.start()
            if not ready.wait(timeout=5.0):
                raise RuntimeError("agent runtime loop did not start")
            loop = holder.get("loop")
            if loop is None:
                raise RuntimeError("agent runtime loop unavailable")
            self._agent_runtime_loop = loop
            self._agent_runtime_thread = thread
            return loop

    async def _run_on_agent_runtime(self, coroutine: Any) -> Any:
        loop = self._ensure_agent_runtime_loop()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return await asyncio.wrap_future(future)

    def _shutdown_agent_runtime_loop(self) -> None:
        with self._agent_runtime_lock:
            loop = self._agent_runtime_loop
            thread = self._agent_runtime_thread
            self._agent_runtime_loop = None
            self._agent_runtime_thread = None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

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

    @staticmethod
    def _default_product_order_path(product_name: str) -> Path:
        return (_PRODUCT_ORDERS_DIR / f"{str(product_name).strip()}.json").resolve()

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

    def list_product_order_files(self, product_init_file: str | None = None) -> list[str]:
        options: set[str] = set()
        if _PRODUCT_ORDERS_DIR.is_dir():
            for p in _PRODUCT_ORDERS_DIR.iterdir():
                if p.is_file() and p.suffix == ".json":
                    options.add(self._norm_path(str(p)))
        if product_init_file:
            try:
                raw = self.load_config(str(product_init_file))
                _product_name, product_meta = self._first_manifest_entry(raw)
                order_raw = str(product_meta.get("product_order_file", "")).strip()
                if order_raw:
                    options.add(self._norm_path(self._abs_project_path(order_raw)))
            except Exception:
                pass
        return sorted(options)

    def load_product_order(self, product_order_file: str) -> dict[str, Any]:
        order_path = self._abs_project_path(product_order_file).resolve()
        return load_product_order_file(order_path)

    def save_product_order(self, product_order_file: str, payload: dict[str, Any]) -> dict[str, Any]:
        order_path = self._abs_project_path(product_order_file).resolve()
        order_path.parent.mkdir(parents=True, exist_ok=True)
        order_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return load_product_order_file(order_path)

    def delete_product_order(self, product_order_file: str) -> None:
        order_path = self._abs_project_path(product_order_file).resolve()
        if order_path.exists():
            order_path.unlink()

    def product_geometry_slots_for_product(self, product_init_file: str | None = None) -> list[str]:
        product_file = str(product_init_file or self.selected_product or "").strip()
        if not product_file:
            files = self.list_product_files()
            product_file = files[0] if files else ""
        if not product_file:
            return []
        raw = self.load_config(product_file)
        _product_name, product_meta = self._first_manifest_entry(raw)
        geometry_file = str(product_meta.get("product_geometry_file", "")).strip()
        if not geometry_file:
            return []
        geometry = ProductProfile.load_product_geometry(
            str(self._abs_project_path(geometry_file)),
            robot_env="real" if self.execution_mode == "physical" else "gazebo",
        )
        return geometry_slot_names(geometry)

    def _resolve_product_init_for_product_order(self, product_order_file: str) -> dict[str, Any]:
        order_norm = self._norm_path(self._abs_project_path(product_order_file))
        candidates: list[dict[str, Any]] = []
        for init_file in self.list_product_files():
            try:
                ctx = self._resolve_product_context(init_file, include_hashes=False)
            except Exception:
                continue
            out = dict(ctx)
            out["product_init_file"] = str(Path(init_file).resolve())
            candidates.append(out)
            if self._norm_path(ctx.get("product_order_file", "")) == order_norm:
                return out
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"No product initialization manifest references product order file: {product_order_file}")

    def resolve_product_init_for_product_order(self, product_order_file: str) -> str:
        ctx = self._resolve_product_init_for_product_order(product_order_file)
        return str(ctx["product_init_file"])

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

        dot_signature: list[tuple[str, int]] = []
        for raw_dot in latest.get("dfa_dot_files", []):
            try:
                dot_path = Path(str(raw_dot or "").strip())
                dot_signature.append(
                    (str(dot_path.resolve()), dot_path.stat().st_mtime_ns if dot_path.exists() else 0)
                )
            except Exception:
                continue
        cache_signature = (
            safety_key,
            current_hash,
            str(latest.get("preview_id", "") or "").strip(),
            str(logic_path.resolve()),
            logic_path.stat().st_mtime_ns,
            tuple(sorted(dot_signature)),
        )
        with self._safety_rule_preview_cache_lock:
            cached = self._safety_rule_preview_cache.get(safety_key)
            if cached and cached[0] == cache_signature:
                return deepcopy(cached[1])

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
        payload_out = {
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
        with self._safety_rule_preview_cache_lock:
            self._safety_rule_preview_cache[safety_key] = (cache_signature, deepcopy(payload_out))
        return payload_out

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
        product_order_path_raw = str(product_meta.get("product_order_file", "")).strip()

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
        order_file_str = ""
        if product_order_path_raw:
            order_file_str = str(self._abs_project_path(product_order_path_raw).resolve())
        elif product_name:
            order_file_str = str(self._default_product_order_path(product_name))
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
            "product_order_file": order_file_str,
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

    def run_order_dry_run(
        self,
        product_order_file: str,
        safety_requirement_file: str,
        *,
        require_verified_safety: bool = True,
    ) -> dict[str, Any]:
        if self.system_running or self._starting or self._stopping:
            raise RuntimeError("cannot run order dry run while system lifecycle is active")
        self._ensure_called_from_worker_thread("run_order_dry_run")

        order_path = self._abs_project_path(product_order_file).resolve()
        if not order_path.exists():
            raise FileNotFoundError(f"product order file missing: {order_path}")
        product_ctx = self._resolve_product_init_for_product_order(str(order_path))
        product_init_file = str(product_ctx.get("product_init_file") or "")
        product_raw = self.load_config(product_init_file)
        product_name, product_meta = self._first_manifest_entry(product_raw)
        geometry_file = str(product_meta.get("product_geometry_file", "")).strip()
        if not geometry_file:
            raise ValueError(f"product {product_name} missing product_geometry_file")
        geometry = ProductProfile.load_product_geometry(
            str(self._abs_project_path(geometry_file)),
            robot_env="real" if self.execution_mode == "physical" else "gazebo",
        )
        product_order = load_product_order_file(order_path)
        validated = validate_product_order(product_order, geometry)

        safety_raw = str(safety_requirement_file or "").strip()
        if not safety_raw or safety_raw.upper() == "__NONE__":
            raise ValueError("select a verified safety file for order dry run")
        safety_path = self._abs_project_path(safety_raw).resolve()
        if not safety_path.exists():
            raise FileNotFoundError(f"safety file missing: {safety_path}")
        if require_verified_safety:
            safety_eval = self.evaluate_safety_intent_approval(str(safety_path))
            if not bool(safety_eval.get("approved", False)):
                reason = str(safety_eval.get("reason", "not_approved") or "not_approved")
                raise ValueError(f"safety file is not verified: {reason}")
        safety_text = safety_path.read_text(encoding="utf-8").strip()
        preview = self.get_safety_rule_preview(str(safety_path))
        preview_rules = preview.get("rules", []) if isinstance(preview.get("rules"), list) else []

        resources = self.bundle_compiler._collect_resource_refs(
            robot_env="real" if self.execution_mode == "physical" else "gazebo"
        )
        tools_catalog = []
        if _TOOLS_OUT.exists():
            try:
                raw_tools = json.loads(_TOOLS_OUT.read_text(encoding="utf-8"))
                if isinstance(raw_tools, list):
                    tools_catalog = raw_tools
            except Exception:
                tools_catalog = []
        dry_agent = SimpleNamespace(
            jid=str(validated.payload.get("product_jid") or f"{product_name}@localhost"),
            logger=log,
            product_geometry=geometry,
            product_order_file=str(order_path),
            tools_catalog=tools_catalog,
        )
        planner = ProcessPlanner(dry_agent, resources)
        artifact = planner.build_from_product_order(
            dict(validated.payload),
            safety_text=safety_text,
        )
        try:
            planner.compile_global_fsa()
        except Exception as exc:
            artifact["derived_dag_compile_error"] = str(exc)

        simulation = self._simulate_order_dry_run(planner.nodes)
        monitor_rules = [
            {
                "id": str(rule.get("id", "")),
                "raw_text": str(rule.get("raw_text", "")),
                "constraint_type": str(rule.get("constraint_type", "")),
                "ltlf": str(rule.get("ltlf", "")),
                "generated_interpretation": str(rule.get("generated_interpretation", "")),
            }
            for rule in preview_rules
            if isinstance(rule, dict)
        ]
        if not monitor_rules:
            monitor_rules = list(artifact.get("monitor_rules") or [])

        out = {
            **artifact,
            "product_order": dict(validated.payload),
            "selected_parts": list(validated.selected_parts),
            "ordering_constraints": derive_ordering_constraints_from_safety(
                safety_text,
                list(validated.selected_parts),
            ),
            "monitor_rules": monitor_rules,
            "derived_nodes": deepcopy(planner.nodes),
            "simulated_event_trace": simulation["simulated_event_trace"],
            "ready_operations": simulation["ready_operations"],
            "blocked_operations": simulation["blocked_operations"],
            "completed_operations": simulation["completed_operations"],
            "monitor_state": simulation["monitor_state"],
            "product_order_file": str(order_path),
            "safety_file": str(safety_path),
        }
        dry_run_dir = _MONITOR / "dry_run"
        dry_run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_path = dry_run_dir / f"{stamp}__{slug(order_path.stem)}.json"
        suffix = 1
        while artifact_path.exists():
            suffix += 1
            artifact_path = dry_run_dir / f"{stamp}__{slug(order_path.stem)}_{suffix}.json"
        out["artifact_path"] = str(artifact_path.resolve())
        atomic_json_write(artifact_path, out)
        return out

    @staticmethod
    def _simulate_order_dry_run(nodes: list[dict[str, Any]]) -> dict[str, Any]:
        task_nodes = [
            deepcopy(node)
            for node in nodes
            if isinstance(node, dict) and node.get("type") == "task"
        ]
        pending: dict[str, dict[str, Any]] = {
            str(node.get("id")): node
            for node in task_nodes
            if str(node.get("id") or "").strip()
        }
        completed: set[str] = set()
        completed_operations: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []

        def _operation_row(node: dict[str, Any], *, missing: list[str] | None = None) -> dict[str, Any]:
            params = node.get("params") if isinstance(node.get("params"), dict) else {}
            return {
                "task_id": str(node.get("id", "")),
                "function_name": str(node.get("function_name", "")),
                "part_name": str(params.get("part_name", "")),
                "resource_jid": str(node.get("resource_jid", "")),
                "missing_predecessors": list(missing or []),
            }

        step = 0
        last_ready: list[dict[str, Any]] = []
        last_blocked: list[dict[str, Any]] = []
        while pending:
            ready_nodes: list[dict[str, Any]] = []
            blocked_rows: list[dict[str, Any]] = []
            for node in sorted(
                pending.values(),
                key=lambda item: (
                    int(item.get("sequence_index", 10**9) or 0),
                    str(item.get("id", "")),
                ),
            ):
                missing = [
                    str(pred_id)
                    for pred_id in (node.get("predecessors") or [])
                    if str(pred_id) not in completed
                ]
                if missing:
                    blocked_rows.append(_operation_row(node, missing=missing))
                else:
                    ready_nodes.append(node)

            ready_rows = [_operation_row(node) for node in ready_nodes]
            trace.append(
                {
                    "step": step,
                    "ready_operations": ready_rows,
                    "blocked_operations": blocked_rows,
                    "completed_task_ids": sorted(completed),
                }
            )
            last_ready = ready_rows
            last_blocked = blocked_rows
            if not ready_nodes:
                return {
                    "ready_operations": last_ready,
                    "blocked_operations": last_blocked,
                    "completed_operations": completed_operations,
                    "simulated_event_trace": trace,
                    "monitor_state": {
                        "status": "blocked",
                        "completed_task_count": len(completed),
                        "remaining_task_count": len(pending),
                    },
                }

            for node in ready_nodes:
                task_id = str(node.get("id", ""))
                if task_id not in pending:
                    continue
                event = _operation_row(node)
                event["status"] = "completed"
                event["start_event"] = f"{task_id}.start"
                event["done_event"] = f"{task_id}.done"
                completed_operations.append(event)
                completed.add(task_id)
                pending.pop(task_id, None)
            step += 1

        trace.append(
            {
                "step": step,
                "ready_operations": [],
                "blocked_operations": [],
                "completed_task_ids": sorted(completed),
            }
        )
        return {
            "ready_operations": last_ready,
            "blocked_operations": last_blocked,
            "completed_operations": completed_operations,
            "simulated_event_trace": trace,
            "monitor_state": {
                "status": "completed",
                "completed_task_count": len(completed),
                "remaining_task_count": 0,
            },
        }

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
            fast_forward_runtime = self._fast_forward_simulation_runtime_enabled()
            os.environ["CAIS_GAZEBO_WAIT_SCALE"] = (
                "0.35" if fast_forward_runtime else "1.0"
            )
            os.environ["CAIS_SKIP_RECOVERY_HOME_AFTER_PLACE"] = (
                "1" if fast_forward_runtime else "0"
            )
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
                # Hand off prewarmed controllers only when explicitly kept alive.
                # By default prewarm is readiness-only and destroys controllers
                # after wait_for_services() to keep the UI/Gazebo process stable.
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

            selected_product_order_file = str(self.selected_product_order_file or "").strip()
            if selected_product_order_file:
                try:
                    selected_product_file = await asyncio.to_thread(
                        self.resolve_product_init_for_product_order,
                        selected_product_order_file,
                    )
                    self.selected_product = selected_product_file
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to resolve product init for product order file '{selected_product_order_file}': {exc}"
                    ) from exc

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

            bundle_context: dict[str, Any] | None = None
            bundle_notice: str | None = None
            if selected_product_order_file:
                self.bundle_store.set_active_bundle_id(None)
            else:
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
            if selected_product_order_file:
                runtime_overrides["product_order_file"] = selected_product_order_file
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
            agent_loop = self._ensure_agent_runtime_loop()

            async def _create_agents_on_runtime_loop() -> tuple[Any, list[Any], list[Any], Any]:
                return self._create_agents(
                    ac,
                    prod_files,
                    res_files,
                    str(_CCA_INIT),
                    prewarmed,
                    bundle_context,
                    runtime_overrides,
                )

            (
                self.user_agent,
                self.resource_agents,
                self.product_agents,
                self.cca,
            ) = await self._run_on_agent_runtime(
                _create_agents_on_runtime_loop()
            )
            self._bind_agents_to_running_loop(
                agent_loop,
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

            kickoff_results = await self._run_on_agent_runtime(
                self._start_agents_and_wait_kickoff(startup_id, startup_t0)
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
            await self._run_on_agent_runtime(self._cleanup_agents())
        finally:
            self._starting = False
            if not self.system_running:
                self._set_startup_phase("idle")

    async def _start_agents_and_wait_kickoff(
        self,
        startup_id: int,
        startup_t0: float,
    ) -> list[dict[str, Any]]:
        """Start SPADE agents on the dedicated agent runtime loop."""
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
        self._diag_emit(
            f"startup#{startup_id} agent kickoff wait done in {time.monotonic() - startup_t0:.2f}s"
        )
        return kickoff_results

    async def stop_system(self) -> None:
        """Stop all SPADE agents."""
        if not self.system_running or self._stopping:
            return
        self._stopping = True
        try:
            await self._run_on_agent_runtime(self._cleanup_agents())
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
                if entry.is_file() and entry.suffix.lower() in {".txt", ".json", ".md"} or entry.is_dir() and run_dir_pattern.match(entry.name):
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
        if str(execution_mode or "").strip().lower() == "simulation":
            os.environ["ENABLE_ROBOT_AGENT_PREWARM"] = "0"
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
        product_order_file = runtime_overrides.get("product_order_file")
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
        if product_order_file:
            product_kwargs["product_order_file"] = product_order_file
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
    _DUAL_DRAG_MARKERS_SCRIPT = str(
        _PROJECT_ROOT / "ros2" / "cais_lab_gazebo" / "scripts" / "dual_drag_markers.py"
    )
    _DIGITAL_TWIN_SYNC_SCRIPT = (
        _PROJECT_ROOT / "ros2" / "cais_lab_gazebo" / "scripts" / "digital_twin_sync.py"
    )

    # Registry: name → shell command suffix (appended after _ROS2_ENV).
    ROS2_LAUNCH_CMDS: dict[str, str] = {
        "gazebo_dual": (
            "ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py "
            "run_perception:=false include_assembly_parts:=false"
        ),
        "gazebo_dual_gazebo_only": (
            "ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py "
            "launch_moveit:=false launch_rviz:=false "
            "run_perception:=false include_assembly_parts:=false"
        ),
        "gazebo_dual_moveit_only": (
            "ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py "
            "launch_gazebo:=false launch_moveit:=true launch_rviz:=true "
            "run_perception:=false include_assembly_parts:=false"
        ),
        "gazebo_dual_passive": (
            "ros2 launch xarm_gazebo xarm6_ur5e_gazebo.launch.py "
            "passive:=true run_perception:=false include_assembly_parts:=false"
        ),
        "gazebo_xarm6": "ros2 launch xarm_gazebo xarm6_moveit_single_gazebo.launch.py",
        "gazebo_ur5e": "ros2 launch xarm_gazebo ur5e_rg2_moveit_gazebo.launch.py",
        "gazebo_xarm6_passive": "ros2 launch xarm_gazebo xarm6_single_gazebo.launch.py passive:=true",
        "gazebo_ur5e_passive": "ros2 launch xarm_gazebo ur5e_rg2_gazebo.launch.py passive:=true run_perception:=false",
        "hardware_xarm6_driver": "ros2 launch xarm_gazebo xarm6_hardware_driver.launch.py robot_ip:={xarm6_ip}",
        "hardware_xarm6_moveit": (
            "ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py "
            "robot_ip:={xarm6_ip} add_gripper:=true"
        ),
        "hardware_ur5e_rtde_trajectory_server": (
            f"{_VENV_PYTHON} {_UR5E_RTDE_TRAJECTORY_SCRIPT} "
            f"--robot-ip {{ur5e_ip}} --status-file {_UR5E_RTDE_TRAJECTORY_STATUS}"
        ),
        "hardware_ur5e_rg2_gripper": (
            f"{_VENV_PYTHON} {_UR5E_RG2_GRIPPER_SCRIPT} "
            "--robot-ip {ur5e_ip} --backend xmlrpc"
        ),
        "hardware_ur5e_moveit": (
            "ros2 launch xarm_gazebo ur5e_rg2_hardware_moveit.launch.py "
            "launch_rviz:=true"
        ),
        "hardware_dual_robots_moveit": (
            "ros2 launch xarm_gazebo dual_robots_hardware_moveit.launch.py "
            "launch_rviz:=true"
        ),
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
        names = set(self.ROS2_LAUNCH_CMDS) | self._DIGITAL_TWIN_PROCESS_NAMES
        return {name: self.ros2_proc_status(name) for name in sorted(names)}

    @classmethod
    def _default_ros_domain_id(cls) -> int:
        return cls._env_int("ROS_DOMAIN_ID", 0)

    def _normal_gazebo_teleop_process(self, robot: str) -> str | None:
        key = str(robot).strip().lower()
        if self.ros2_proc_status("gazebo_dual") == "running":
            return "gazebo_dual"
        if key == "xarm6" and self.ros2_proc_status("gazebo_xarm6") == "running":
            return "gazebo_xarm6"
        if key == "ur5e" and self.ros2_proc_status("gazebo_ur5e") == "running":
            return "gazebo_ur5e"
        return None

    def _normal_hardware_teleop_processes(self, robot: str) -> dict[str, str]:
        key = str(robot).strip().lower()
        if key == "xarm6":
            return {
                "moveit": "hardware_xarm6_moveit",
            }
        if key == "ur5e":
            return {
                "rtde": "hardware_ur5e_rtde_trajectory_server",
                "gripper": "hardware_ur5e_rg2_gripper",
                "moveit": "hardware_ur5e_moveit",
            }
        return {}

    def _digital_twin_teleop_target(self, robot: str) -> tuple[str, dict[str, Any]] | None:
        key = str(robot).strip().lower()
        for target, cfg in self._DIGITAL_TWIN_TARGETS.items():
            hardware_robots = {str(r).strip().lower() for r in (cfg.get("hardware") or ())}
            if str(cfg.get("robot") or "").strip().lower() != key and key not in hardware_robots:
                continue
            process_names = self._digital_twin_process_names(cfg)
            if self._running_process_names(process_names):
                return target, cfg
        return None

    def _any_teleop_environment_running(self) -> bool:
        names = self._GAZEBO_PROCESS_NAMES | self._HARDWARE_PROCESS_NAMES
        return self._any_running(names)

    def teleop_environment_running(self) -> bool:
        return self._any_teleop_environment_running()

    def _teleop_process_running(self, process_name: str | None) -> bool:
        return bool(process_name) and self.ros2_proc_status(str(process_name)) == "running"

    def teleop_target(self, robot: str | None = None, op: str | None = None) -> dict[str, Any]:
        """Resolve the ROS graph and process requirements for Interactive Teleop."""
        key = str(robot or "").strip().lower()
        op_key = str(op or "").strip().lower()
        default_domain = self._default_ros_domain_id()
        domains = self._digital_twin_domain_ids()

        result: dict[str, Any] = {
            "environment": "real" if str(self.robot_env).strip().lower() == "real" else "gazebo",
            "ros_domain_id": default_domain,
            "source": "default",
            "robot": key,
            "moveit_process": "",
            "gripper_process": "",
            "required_processes": [],
            "ready": False,
            "warning": "MoveIt is not running. Start the matching Gazebo, Hardware Stack, or Digital Twin launch first.",
        }

        if key not in {"xarm6", "ur5e"}:
            if key:
                result["warning"] = f"unknown robot: {key}"
            elif self._any_running(self._HARDWARE_PROCESS_NAMES):
                result.update({"environment": "real", "source": "hardware", "warning": ""})
            elif self._any_running(self._GAZEBO_PROCESS_NAMES):
                result.update({"environment": "gazebo", "source": "gazebo", "warning": ""})
            return result

        digital_twin_target = self._digital_twin_teleop_target(key)
        if digital_twin_target is not None:
            target, cfg = digital_twin_target
            hardware_processes = self._digital_twin_hardware_processes_for_robot(cfg, key)
            gripper_process = str(
                hardware_processes.get("gripper")
                or hardware_processes.get("driver")
                or hardware_processes.get("moveit")
                or ""
            ).strip()
            result.update(
                {
                    "environment": "real",
                    "ros_domain_id": self._digital_twin_hardware_domain_id(cfg, key, domains),
                    "source": f"digital_twin:{target}",
                    "moveit_process": str(hardware_processes.get("moveit") or "").strip(),
                    "gripper_process": gripper_process,
                }
            )
        else:
            hardware_processes = self._normal_hardware_teleop_processes(key)
            if any(self._teleop_process_running(name) for name in hardware_processes.values()):
                result.update(
                    {
                        "environment": "real",
                        "ros_domain_id": default_domain,
                        "source": "hardware",
                        "moveit_process": str(hardware_processes.get("moveit") or "").strip(),
                        "gripper_process": str(hardware_processes.get("gripper") or "").strip(),
                    }
                )
            else:
                gazebo_process = self._normal_gazebo_teleop_process(key)
                if gazebo_process:
                    result.update(
                        {
                            "environment": "gazebo",
                            "ros_domain_id": default_domain,
                            "source": "gazebo",
                            "moveit_process": gazebo_process,
                            "gripper_process": gazebo_process,
                        }
                    )

        moveit_process = str(result.get("moveit_process") or "").strip()
        gripper_process = str(result.get("gripper_process") or "").strip()
        required: list[str] = []
        warning = ""

        if op_key == "gripper":
            if result.get("environment") == "real" and key == "ur5e":
                required = [gripper_process]
                if not self._teleop_process_running(gripper_process):
                    warning = "RG2 gripper bridge is not running. Start UR5e Hardware Stack, ur5e only Digital Twin, or dual robots Digital Twin first."
            elif result.get("environment") == "real" and key == "xarm6":
                required = [gripper_process or moveit_process]
                if not self._teleop_process_running(gripper_process or moveit_process):
                    warning = "xArm6 gripper driver is not running. Start xArm6 Hardware Stack or dual robots Digital Twin first."
            else:
                required = [moveit_process]
                if not self._teleop_process_running(moveit_process):
                    warning = f"MoveIt for {key} is not running. Start the matching {key} launch first."
        elif op_key in {"cartesian", "joint", "home", "move_joints", "save_position", "state"}:
            required = [moveit_process]
            if not self._teleop_process_running(moveit_process):
                warning = f"MoveIt for {key} is not running. Start the matching {key} launch first."
        else:
            required = [moveit_process] if moveit_process else []
            if moveit_process and not self._teleop_process_running(moveit_process):
                warning = f"MoveIt for {key} is not running. Start the matching {key} launch first."

        if not moveit_process and not gripper_process:
            if self._any_teleop_environment_running():
                warning = f"MoveIt for {key} is not running. Start the matching {key} launch first."
            else:
                warning = "MoveIt is not running. Start the matching Gazebo, Hardware Stack, or Digital Twin launch first."

        required = [name for name in required if name]
        ready = bool(required) and all(self._teleop_process_running(name) for name in required)
        result["required_processes"] = required
        result["ready"] = ready
        result["warning"] = "" if ready else warning
        return result

    def teleop_target_environment(self) -> str:
        """Infer teleop target environment from active ROS2 stacks."""
        return str(self.teleop_target().get("environment") or "gazebo")

    def teleop_connection_status(self, robot: str | None = None) -> dict[str, Any]:
        """Return teleop backend connectivity and resolved target environment."""
        target = self.teleop_target(robot, "state") if robot else self.teleop_target()
        with self._teleop_server_lock:
            proc = self._teleop_server_proc
            connected = proc is not None and proc.poll() is None
            pid = proc.pid if connected else None
        return {
            "connected": bool(connected),
            "environment": str(target.get("environment") or self.teleop_target_environment()),
            "ros_domain_id": target.get("ros_domain_id"),
            "warning": str(target.get("warning") or ""),
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

    def _fast_forward_simulation_runtime_enabled(self) -> bool:
        return (
            bool(getattr(self, "fast_forward_simulation_enabled", False))
            and str(self.execution_mode or "").strip().lower() == "simulation"
            and str(self.robot_env or "").strip().lower() == "gazebo"
        )

    def _render_ros2_launch_cmd(
        self,
        name: str,
        *,
        fast_forward_simulation: bool | None = None,
    ) -> str:
        cmd = self.ROS2_LAUNCH_CMDS[name]
        if (
            str(name or "").strip().lower() == "gazebo_dual"
            and bool(fast_forward_simulation)
        ):
            cmd = f"{cmd} fast_sim:=true launch_rviz:=false"
        return cmd.format(
            xarm6_ip=self.hardware_ips.get("xarm6", self._HW_IP_DEFAULTS["xarm6"]),
            ur5e_ip=self.hardware_ips.get("ur5e", self._HW_IP_DEFAULTS["ur5e"]),
        )

    def _hardware_stack_for_robot(self, robot: str) -> tuple[str, ...] | None:
        key = str(robot or "").strip().lower()
        return self._HARDWARE_STACKS.get(key)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)

    @classmethod
    def _digital_twin_domain_ids(cls) -> dict[str, int]:
        return {
            "gazebo": cls._env_int(
                "CAIS_DIGITAL_TWIN_GAZEBO_DOMAIN_ID",
                cls._DIGITAL_TWIN_GAZEBO_DOMAIN_DEFAULT,
            ),
            "hardware": cls._env_int(
                "CAIS_DIGITAL_TWIN_HARDWARE_DOMAIN_ID",
                cls._DIGITAL_TWIN_HARDWARE_DOMAIN_DEFAULT,
            ),
            "hardware_xarm6": cls._env_int(
                "CAIS_DIGITAL_TWIN_HARDWARE_XARM6_DOMAIN_ID",
                cls._DIGITAL_TWIN_HARDWARE_XARM6_DOMAIN_DEFAULT,
            ),
            "hardware_ur5e": cls._env_int(
                "CAIS_DIGITAL_TWIN_HARDWARE_UR5E_DOMAIN_ID",
                cls._DIGITAL_TWIN_HARDWARE_UR5E_DOMAIN_DEFAULT,
            ),
        }

    @staticmethod
    def _ros2_domain_export(ros_domain_id: int | None) -> str:
        if ros_domain_id is None:
            return ""
        try:
            domain = int(ros_domain_id)
        except (TypeError, ValueError):
            return ""
        return f"export ROS_DOMAIN_ID={domain}; "

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
        dual_passive_assets = (
            (
                xarm_gazebo_share / "config" / "xarm6_ur5e_controllers.yaml",
                "ROS2 workspace is missing the dual-robot xarm_gazebo controller config. Re-run `make bootstrap-gazebo`.",
            ),
            (
                xarm_gazebo_share / "config" / "ur5e_initial_positions.yaml",
                "ROS2 workspace is missing the UR5e initial positions config. Re-run `make bootstrap-gazebo`.",
            ),
        )
        xarm_hardware_driver_assets = (
            (
                xarm_gazebo_share / "launch" / "xarm6_hardware_driver.launch.py",
                "ROS2 workspace is missing the xArm6 hardware driver launch file. Re-run `make bootstrap-gazebo`.",
            ),
        )
        dual_hardware_moveit_assets = (
            (
                xarm_gazebo_share / "launch" / "dual_robots_hardware_moveit.launch.py",
                "ROS2 workspace is missing the dual robots hardware MoveIt launch file. Re-run `make bootstrap-gazebo`.",
            ),
            (
                xarm_gazebo_share / "rviz" / "dual_robots_hardware_moveit.rviz",
                "ROS2 workspace is missing the dual robots hardware RViz config. Re-run `make bootstrap-gazebo`.",
            ),
        )
        ur_assets = (
            (
                xarm_gazebo_share / "config" / "ur5e_rg2_controllers.yaml",
                "ROS2 workspace is missing the UR5e RG2 controller config. Re-run `make bootstrap-gazebo`.",
            ),
        )
        ur_hardware_rg2_assets = (
            (
                xarm_gazebo_share / "launch" / "ur5e_rg2_hardware_moveit.launch.py",
                "ROS2 workspace is missing the UR5e RG2 hardware MoveIt launch file. Re-run `make bootstrap-gazebo`.",
            ),
            (
                xarm_gazebo_share / "rviz" / "ur5e_rg2_hardware_moveit.rviz",
                "ROS2 workspace is missing the UR5e RG2 hardware RViz config. Re-run `make bootstrap-gazebo`.",
            ),
        )
        ur_rg2_bridge_assets = (
            (
                _VENV_PYTHON,
                f"Python venv is missing at {_VENV_PYTHON}. Create the project venv before starting the UR5e RG2 bridge.",
            ),
            (
                _UR5E_RG2_GRIPPER_SCRIPT,
                f"UR5e RG2 bridge script is missing at {_UR5E_RG2_GRIPPER_SCRIPT}.",
            ),
        )
        ur_rtde_trajectory_assets = (
            (
                _VENV_PYTHON,
                f"Python venv is missing at {_VENV_PYTHON}. Create the project venv before starting the UR5e RTDE trajectory server.",
            ),
            (
                _UR5E_RTDE_TRAJECTORY_SCRIPT,
                f"UR5e RTDE trajectory server script is missing at {_UR5E_RTDE_TRAJECTORY_SCRIPT}.",
            ),
        )

        if launch_key == "gazebo_dual":
            return [*workspace_xarm, *moveit_core, *ur_stack, *onrobot_ws, *link_attacher_ws, *dual_assets]
        if launch_key == "gazebo_dual_passive":
            return [*workspace_xarm, *ur_stack, *onrobot_ws, *dual_passive_assets]
        if launch_key == "gazebo_xarm6":
            return [*workspace_xarm, *moveit_core, *link_attacher_ws]
        if launch_key == "gazebo_ur5e":
            return [*workspace_xarm, *moveit_core, *ur_stack, *onrobot_ws, *link_attacher_ws, *ur_assets]
        if launch_key == "gazebo_xarm6_passive":
            return [*workspace_xarm]
        if launch_key == "gazebo_ur5e_passive":
            return [*workspace_xarm, *ur_stack, *onrobot_ws, *ur_assets]
        if launch_key == "hardware_xarm6_driver":
            return [*workspace_xarm, *xarm_hardware_driver_assets]
        if launch_key == "hardware_dual_robots_moveit":
            return [*workspace_xarm, *moveit_core, *ur_stack, *onrobot_ws, *dual_hardware_moveit_assets]
        if launch_key == "hardware_ur5e_moveit":
            return [*moveit_core, *ur_stack, *onrobot_ws, *ur_hardware_rg2_assets]
        if launch_key == "hardware_ur5e_rg2_gripper":
            return [*ur_rg2_bridge_assets]
        if launch_key == "hardware_ur5e_rtde_trajectory_server":
            return [*ur_rtde_trajectory_assets]
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
        ok, out = self._ros2_command_output(
            "ros2 service list --no-daemon --spin-time 2.0",
            timeout_sec=timeout_sec,
        )
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
        return False

    def _ros2_command_output(
        self,
        command: str,
        timeout_sec: float = 8.0,
        *,
        ros_domain_id: int | None = None,
        emit_slow_diag: bool = True,
        emit_failure_diag: bool = True,
        emit_timeout_diag: bool = True,
    ) -> tuple[bool, str]:
        full_cmd = self._ROS2_ENV + self._ros2_domain_export(ros_domain_id) + command
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

    def _restart_ros2_daemon_for_discovery(
        self,
        *,
        ros_domain_id: int | None = None,
    ) -> str | None:
        ok, out = self._ros2_command_output(
            "ros2 daemon stop",
            timeout_sec=5.0,
            ros_domain_id=ros_domain_id,
            emit_slow_diag=False,
            emit_failure_diag=False,
            emit_timeout_diag=False,
        )
        if not ok:
            detail = self._tail_output(out) or str(out or "").strip()
            if detail:
                self._diag_emit(f"ros2 daemon stop during discovery retry: {detail}")

        time.sleep(0.4)
        ok, out = self._ros2_command_output(
            "ros2 daemon start",
            timeout_sec=5.0,
            ros_domain_id=ros_domain_id,
            emit_slow_diag=False,
            emit_failure_diag=False,
            emit_timeout_diag=False,
        )
        if not ok:
            detail = self._tail_output(out) or str(out or "").strip()
            return detail or "ros2 daemon restart failed"
        time.sleep(0.4)
        self._diag_emit("ros2 daemon restarted for ROS discovery retry")
        return None

    @staticmethod
    def _ros2_discovery_wait_can_retry(error_message: str) -> bool:
        text = str(error_message or "").strip().lower()
        if not text:
            return False
        return not any(
            token in text
            for token in (
                "exited before",
                "wait cancelled",
                "controller name is empty",
                "service name is empty",
                "action name is empty",
                "topic name must be absolute",
                "invalid controller name",
                "ok=false",
            )
        )

    def _wait_with_ros2_daemon_retry(
        self,
        label: str,
        wait_fn: Callable[[], str | None],
        *,
        ros_domain_id: int | None = None,
    ) -> str | None:
        err = wait_fn()
        if err is None:
            return None
        if not self._ros2_discovery_wait_can_retry(err):
            return err

        restart_err = self._restart_ros2_daemon_for_discovery(
            ros_domain_id=ros_domain_id,
        )
        if restart_err:
            return f"{err}; ros2 daemon restart failed: {restart_err}"

        retry_err = wait_fn()
        if retry_err is None:
            self._diag_emit(f"{label} recovered after ros2 daemon restart")
            return None
        return f"{err}; after ros2 daemon restart: {retry_err}"

    def _wait_for_ros_service(
        self,
        service_name: str,
        timeout_sec: float = 20.0,
        process_name: str | None = None,
        cancel_event: threading.Event | None = None,
        ros_domain_id: int | None = None,
    ) -> str | None:
        return self._wait_for_ros_services(
            [service_name],
            timeout_sec=timeout_sec,
            process_name=process_name,
            cancel_event=cancel_event,
            ros_domain_id=ros_domain_id,
        )

    def _wait_for_ros_services(
        self,
        service_names: list[str] | tuple[str, ...],
        timeout_sec: float = 20.0,
        process_name: str | None = None,
        cancel_event: threading.Event | None = None,
        ros_domain_id: int | None = None,
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
                "ros2 service list --no-daemon --spin-time 2.0",
                timeout_sec=7.0,
                ros_domain_id=ros_domain_id,
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

    def _wait_for_ros_action(
        self,
        action_name: str,
        timeout_sec: float = 12.0,
        process_name: str | None = None,
        cancel_event: threading.Event | None = None,
        ros_domain_id: int | None = None,
        poll_interval_sec: float = 0.5,
    ) -> str | None:
        target = str(action_name or "").strip()
        if not target:
            return "action name is empty"

        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        while time.monotonic() < deadline:
            if cancel_event and cancel_event.is_set():
                return f"{target} wait cancelled"

            if process_name and self.ros2_proc_status(process_name) != "running":
                return f"{process_name} exited before {target} became available"

            ok, out = self._ros2_command_output(
                "ros2 action list",
                timeout_sec=3.0,
                ros_domain_id=ros_domain_id,
                emit_slow_diag=False,
                emit_failure_diag=False,
                emit_timeout_diag=False,
            )
            if ok:
                actions = [line.strip() for line in out.splitlines() if line.strip()]
                if any(action == target or action.endswith(target) for action in actions):
                    return None
            time.sleep(max(0.1, float(poll_interval_sec)))

        if process_name and self.ros2_proc_status(process_name) != "running":
            return f"{process_name} exited before {target} became available"
        return f"{target} not available within {timeout_sec:.0f}s"

    @staticmethod
    def _topic_publisher_count_from_output(output: str) -> int | None:
        match = re.search(r"\bPublisher count:\s*(\d+)\b", str(output or ""))
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _wait_for_ros_topic_publisher(
        self,
        topic_name: str,
        timeout_sec: float = 12.0,
        process_name: str | None = None,
        cancel_event: threading.Event | None = None,
        ros_domain_id: int | None = None,
        poll_interval_sec: float = 0.5,
    ) -> str | None:
        target = str(topic_name or "").strip()
        if not target:
            return "topic name is empty"
        if not target.startswith("/"):
            return f"topic name must be absolute: {target}"

        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        last_count: int | None = None
        while time.monotonic() < deadline:
            if cancel_event and cancel_event.is_set():
                return f"{target} publisher wait cancelled"

            if process_name and self.ros2_proc_status(process_name) != "running":
                return f"{process_name} exited before {target} had a publisher"

            ok, out = self._ros2_command_output(
                "ros2 topic info -v "
                + shlex.quote(target)
                + " --no-daemon --spin-time 2.0",
                timeout_sec=7.0,
                ros_domain_id=ros_domain_id,
                emit_slow_diag=False,
                emit_failure_diag=False,
                emit_timeout_diag=False,
            )
            if ok:
                last_count = self._topic_publisher_count_from_output(out)
                if last_count is not None and last_count > 0:
                    return None
            time.sleep(max(0.1, float(poll_interval_sec)))

        if process_name and self.ros2_proc_status(process_name) != "running":
            return f"{process_name} exited before {target} had a publisher"
        if last_count == 0:
            return f"{target} has no publishers"
        return f"{target} publisher state not available within {timeout_sec:.0f}s"

    def _ur5e_rtde_trajectory_status(self) -> dict[str, Any]:
        return self._read_json_file(_UR5E_RTDE_TRAJECTORY_STATUS)

    @staticmethod
    def _attach_ur5e_rtde_trajectory_status(
        status: dict[str, Any],
        rtde_status: dict[str, Any],
    ) -> dict[str, Any]:
        status["rtde_trajectory_server"] = str(rtde_status.get("state") or "unknown")
        status["rtde_trajectory_message"] = str(rtde_status.get("message") or "")
        for key in (
            "robot_ip",
            "action_name",
            "point_count",
            "start_delta_rad",
            "start_delta_joint",
            "first_point_time",
            "min_point_spacing",
            "max_segment_velocity_rad_s",
            "max_segment_velocity_joint",
            "time_scale_applied",
            "blocked_reason",
            "final_error_rad",
            "final_error_joint",
        ):
            if key in rtde_status:
                status[f"rtde_trajectory_{key}"] = rtde_status.get(key)
        return status

    def _wait_for_driver_ready(
        self,
        robot: str,
        timeout_sec: float = 18.0,
        *,
        process_name: str | None = None,
        ros_domain_id: int | None = None,
    ) -> str | None:
        key = str(robot).strip().lower()
        stack = self._hardware_stack_for_robot(key)
        if not stack:
            return f"unknown hardware robot: {robot}"
        if len(stack) < 2:
            return None
        driver_name = str(process_name or stack[0])

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
            ok, out = self._ros2_command_output(
                "ros2 service list --no-daemon --spin-time 2.0",
                timeout_sec=7.0,
                ros_domain_id=ros_domain_id,
            )
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
        stack = self._hardware_stack_for_robot(key)
        if not stack:
            return {"overall": "unknown", "driver": "unknown", "moveit": "unknown"}
        if len(stack) == 1:
            moveit_state = self.ros2_proc_status(stack[0])
            overall = "running" if moveit_state == "running" else "stopped"
            return {"overall": overall, "driver": "embedded", "moveit": moveit_state}

        driver_name = stack[0]
        moveit_name = stack[-1]
        gripper_name = next((name for name in stack if "rg2_gripper" in name), "")
        driver_state = self.ros2_proc_status(driver_name)
        moveit_state = self.ros2_proc_status(moveit_name)
        gripper_state = self.ros2_proc_status(gripper_name) if gripper_name else ""
        states = [driver_state, moveit_state]
        if gripper_name:
            states.insert(1, gripper_state)
        if all(state == "running" for state in states):
            overall = "running"
        elif all(state == "stopped" for state in states):
            overall = "stopped"
        else:
            overall = "partial"
        result = {"overall": overall, "driver": driver_state, "moveit": moveit_state}
        if gripper_name:
            result["gripper"] = gripper_state
            result["gripper_action"] = "ready" if gripper_state == "running" else gripper_state
        if key == "ur5e":
            self._attach_ur5e_rtde_trajectory_status(result, self._ur5e_rtde_trajectory_status())
        return result

    def _digital_twin_target(self, target: str) -> dict[str, Any] | None:
        return digital_twin.digital_twin_target(self._DIGITAL_TWIN_TARGETS, target)

    @staticmethod
    def _digital_twin_slug(cfg: dict[str, Any]) -> str:
        return digital_twin.digital_twin_slug(cfg)

    def _digital_twin_status_path(self, target: str) -> Path:
        return digital_twin.digital_twin_status_path(
            self._DIGITAL_TWIN_TARGETS,
            target,
            Path("/tmp"),
        )

    def _digital_twin_direction_path(self, target: str) -> Path:
        return digital_twin.digital_twin_direction_path(
            self._DIGITAL_TWIN_TARGETS,
            target,
            Path("/tmp"),
        )

    def _digital_twin_sync_status_path(self, target: str, robot: str = "") -> Path:
        return digital_twin.digital_twin_sync_status_path(
            self._DIGITAL_TWIN_TARGETS,
            target,
            robot,
            Path("/tmp"),
        )

    def _digital_twin_dual_drag_markers_status_path(self, target: str) -> Path:
        return digital_twin.digital_twin_dual_drag_markers_status_path(
            self._DIGITAL_TWIN_TARGETS,
            target,
            Path("/tmp"),
        )

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        try:
            with Path(path).open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _ur5e_rg2_gripper_status(self) -> dict[str, Any]:
        return self._read_json_file(_UR5E_RG2_GRIPPER_STATUS)

    @staticmethod
    def _digital_twin_hardware_processes_for_robot(
        cfg: dict[str, Any],
        robot: str,
    ) -> dict[str, str]:
        return digital_twin.digital_twin_hardware_processes_for_robot(cfg, robot)

    @staticmethod
    def _digital_twin_sync_process_items(cfg: dict[str, Any]) -> list[tuple[str, str]]:
        return digital_twin.digital_twin_sync_process_items(cfg)

    @staticmethod
    def _digital_twin_allowed_sim_modes(cfg: dict[str, Any]) -> tuple[str, ...]:
        return digital_twin.digital_twin_allowed_sim_modes(
            cfg,
            SystemBridge._DIGITAL_TWIN_SIM_MODES,
            SystemBridge._DIGITAL_TWIN_SIM_MODE_ALIASES,
        )

    @classmethod
    def _normalize_digital_twin_sim_mode(cls, mode: object) -> str:
        return digital_twin.normalize_digital_twin_sim_mode(
            mode,
            cls._DIGITAL_TWIN_SIM_MODE_ALIASES,
        )

    @staticmethod
    def _digital_twin_has_multiple_hardware_domains(cfg: dict[str, Any]) -> bool:
        return digital_twin.digital_twin_has_multiple_hardware_domains(cfg)

    @staticmethod
    def _digital_twin_has_per_robot_hardware_processes(cfg: dict[str, Any]) -> bool:
        return digital_twin.digital_twin_has_per_robot_hardware_processes(cfg)

    def _digital_twin_hardware_domain_id(
        self,
        cfg: dict[str, Any],
        robot: str,
        domains: dict[str, int],
    ) -> int:
        return digital_twin.digital_twin_hardware_domain_id(cfg, robot, domains)

    @staticmethod
    def _digital_twin_process_names(cfg: dict[str, Any]) -> list[str]:
        return digital_twin.digital_twin_process_names(cfg)

    def _digital_twin_direction(self, target: str) -> str:
        # Direction is implied by sim mode: teach = sim leads (gazebo -> hardware),
        # monitor = hardware leads (hardware -> gazebo). There is no separate toggle.
        return (
            "gazebo -> hardware"
            if self._digital_twin_sim_mode(target) == "teach"
            else "hardware -> gazebo"
        )

    def _digital_twin_sim_mode(self, target: str) -> str:
        cfg = self._digital_twin_target(target) or {}
        allowed_modes = self._digital_twin_allowed_sim_modes(cfg)
        mode = self._normalize_digital_twin_sim_mode(
            self._digital_twin_sim_modes.get(target, self._DIGITAL_TWIN_DEFAULT_SIM_MODE)
        )
        if mode in allowed_modes:
            return mode
        if self._DIGITAL_TWIN_DEFAULT_SIM_MODE in allowed_modes:
            return self._DIGITAL_TWIN_DEFAULT_SIM_MODE
        if allowed_modes:
            return allowed_modes[0]
        return self._DIGITAL_TWIN_DEFAULT_SIM_MODE

    def _digital_twin_gazebo_launch(self, target: str, cfg: dict[str, Any]) -> str:
        launches = cfg.get("gazebo_launches") or {}
        if isinstance(launches, dict):
            mode = self._digital_twin_sim_mode(target)
            key = str(launches.get(mode) or "").strip()
            if not key:
                legacy_key = "mirror" if mode == "monitor" else "author"
                key = str(launches.get(legacy_key) or "").strip()
            if key:
                return key
        return str(cfg.get("gazebo") or "").strip()

    def digital_twin_set_sim_mode(self, target: str, mode: str) -> str | None:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return f"unknown digital twin target: {target}"
        value = self._normalize_digital_twin_sim_mode(mode)
        if value not in self._digital_twin_allowed_sim_modes(cfg):
            return f"unknown digital twin sim mode: {mode}"
        if self._running_process_names(self._digital_twin_process_names(cfg)):
            return "Stop the twin before changing sim mode."
        self._digital_twin_sim_modes[target] = value
        return None

    def _write_digital_twin_direction(self, target: str, direction: str) -> None:
        atomic_json_write(
            self._digital_twin_direction_path(target),
            {
                "target": target,
                "direction": direction,
                "updated_at": time.time(),
            },
        )

    def _write_digital_twin_status(self, target: str, payload: dict[str, Any]) -> None:
        body = dict(payload)
        body.setdefault("target", target)
        body.setdefault("updated_at", time.time())
        atomic_json_write(self._digital_twin_status_path(target), body)

    def _write_digital_twin_sync_status(
        self,
        target: str,
        cfg: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        body = dict(payload)
        body.setdefault("target", target)
        body.setdefault("updated_at", time.time())
        items = self._digital_twin_sync_process_items(cfg)
        if len(items) <= 1:
            atomic_json_write(self._digital_twin_status_path(target), body)
            return
        for robot, _process in items:
            robot_key = str(robot or "").strip().lower()
            robot_body = dict(body)
            robot_body.setdefault("robot", robot_key)
            atomic_json_write(
                self._digital_twin_sync_status_path(target, robot_key),
                robot_body,
            )

    def _running_process_names(self, names: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        return [str(name) for name in names if self.ros2_proc_status(str(name)) == "running"]

    def _active_digital_twin_target(self) -> str | None:
        for target, cfg in self._DIGITAL_TWIN_TARGETS.items():
            if not bool(cfg.get("hardware_supported", False)):
                continue
            if self._running_process_names(self._digital_twin_process_names(cfg)):
                return target
        return None

    def _digital_twin_hardware_status(self, cfg: dict[str, Any]) -> dict[str, Any]:
        hardware_processes = cfg.get("hardware_processes") or {}
        if not bool(cfg.get("hardware_supported", False)) or not isinstance(hardware_processes, dict):
            return {"overall": "unsupported", "driver": "unsupported", "moveit": "unsupported"}
        target = next(
            (
                target_name
                for target_name, target_cfg in self._DIGITAL_TWIN_TARGETS.items()
                if target_cfg is cfg
            ),
            "",
        )
        domains = self._digital_twin_domain_ids()

        def _status_for_processes(processes: dict[str, str]) -> dict[str, str]:
            driver_name = str(processes.get("rtde") or processes.get("driver") or "").strip()
            gripper_name = str(processes.get("gripper") or "").strip()
            moveit_name = str(processes.get("moveit") or "").strip()
            driver_state = "embedded" if not driver_name else self.ros2_proc_status(driver_name)
            gripper_state = self.ros2_proc_status(gripper_name) if gripper_name else ""
            moveit_state = self.ros2_proc_status(moveit_name) if moveit_name else "unknown"
            states = [driver_state, moveit_state]
            if gripper_name:
                states.insert(1, gripper_state)
            running = all(state in {"running", "embedded"} for state in states)
            stopped = all(state in {"stopped", "embedded"} for state in states)
            if running:
                overall = "running"
            elif stopped:
                overall = "stopped"
            else:
                overall = "partial"
            result = {"overall": overall, "driver": driver_state, "moveit": moveit_state}
            if gripper_name:
                result["gripper"] = gripper_state
                result["gripper_action"] = "ready" if gripper_state == "running" else gripper_state
            return result

        hardware_robots = tuple(str(r).strip().lower() for r in (cfg.get("hardware") or ()))
        if not self._digital_twin_has_per_robot_hardware_processes(cfg):
            robot = hardware_robots[0] if hardware_robots else ""
            status = _status_for_processes(self._digital_twin_hardware_processes_for_robot(cfg, robot))
            if robot == "ur5e":
                self._attach_ur5e_rtde_trajectory_status(status, self._ur5e_rtde_trajectory_status())
            return status

        result: dict[str, Any] = {}
        overall_states: list[str] = []
        moveit_states: list[str] = []
        driver_states: list[str] = []
        gripper_states: list[str] = []
        for robot in hardware_robots:
            robot_status = _status_for_processes(
                self._digital_twin_hardware_processes_for_robot(cfg, robot)
            )
            if robot == "ur5e":
                self._attach_ur5e_rtde_trajectory_status(robot_status, self._ur5e_rtde_trajectory_status())
            result[robot] = robot_status
            overall_states.append(str(robot_status.get("overall", "unknown")))
            moveit_states.append(str(robot_status.get("moveit", "unknown")))
            driver_states.append(str(robot_status.get("driver", "unknown")))
            if "gripper" in robot_status:
                gripper_states.append(str(robot_status.get("gripper", "unknown")))

        if overall_states and all(state == "running" for state in overall_states):
            overall = "running"
        elif overall_states and all(state == "stopped" for state in overall_states):
            overall = "stopped"
        else:
            overall = "partial"
        result["overall"] = overall
        result["driver"] = "running" if driver_states and all(state in {"running", "embedded"} for state in driver_states) else overall
        result["moveit"] = "running" if moveit_states and all(state == "running" for state in moveit_states) else overall
        if gripper_states:
            result["gripper"] = "running" if all(state == "running" for state in gripper_states) else overall
        return result

    def _digital_twin_blocked_reason(self, target: str, cfg: dict[str, Any]) -> str:
        if not bool(cfg.get("hardware_supported", False)):
            return self._DUAL_HARDWARE_LIMITATION

        active_target = self._active_digital_twin_target()
        if active_target and active_target != target:
            return f"Blocked: {active_target} digital twin is running. Stop it first."

        normal_running = self._running_process_names(
            sorted(self._BASE_GAZEBO_PROCESS_NAMES | self._BASE_HARDWARE_PROCESS_NAMES)
        )
        if normal_running:
            return "Blocked: stop Launch Environment processes first: " + ", ".join(normal_running)
        return ""

    def _digital_twin_sync_status_snapshot(
        self,
        target: str,
        cfg: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        items = self._digital_twin_sync_process_items(cfg)
        process_names = [process for _robot, process in items if process]
        process_states = [self.ros2_proc_status(process) for process in process_names]
        if not process_names:
            process_status = "unknown"
        elif all(state == "running" for state in process_states):
            process_status = "running"
        elif all(state == "stopped" for state in process_states):
            process_status = "stopped"
        else:
            process_status = "partial"

        if len(items) <= 1:
            status_file = self._digital_twin_status_path(target)
            status_data = self._read_json_file(status_file)
            updated_at = float(status_data.get("updated_at") or 0.0)
            return {
                "process": process_names[0] if process_names else "",
                "process_status": process_status,
                "status_file": str(status_file),
                "status_files": [str(status_file)],
                "status_data": status_data,
                "status_age_ms": (now - updated_at) * 1000.0 if updated_at > 0 else None,
            }

        entries: list[tuple[str, Path, dict[str, Any]]] = []
        for robot, _process in items:
            path = self._digital_twin_sync_status_path(target, robot)
            entries.append((robot, path, self._read_json_file(path)))

        updated_values = [
            float(data.get("updated_at") or 0.0)
            for _robot, _path, data in entries
            if float(data.get("updated_at") or 0.0) > 0.0
        ]
        oldest_updated_at = min(updated_values) if updated_values else 0.0
        messages: list[str] = []
        last_errors: list[str] = []
        states: list[str] = []
        latency_values: list[float] = []
        max_delta_values: list[float] = []
        for robot, _path, data in entries:
            state = str(data.get("state") or "starting")
            states.append(state)
            message = str(data.get("message") or "waiting for sync status.").strip()
            messages.append(f"{robot}: {message}")
            last_error = str(data.get("last_error") or "").strip()
            if last_error:
                last_errors.append(f"{robot}: {last_error}")
            latency = data.get("latency_ms")
            if latency is not None:
                latency_values.append(float(latency))
            max_delta = data.get("max_joint_delta_deg")
            if max_delta is not None:
                max_delta_values.append(float(max_delta))

        if states and all(state == "mirroring" for state in states):
            state = "mirroring"
        elif any(state == "error" for state in states):
            state = "error"
        elif any(state == "paused" for state in states):
            state = "paused"
        elif any(state == "waiting" for state in states):
            state = "waiting"
        else:
            state = "starting"

        status_data = {
            "state": state,
            "message": " | ".join(messages),
            "last_error": " | ".join(last_errors),
            "latency_ms": max(latency_values) if latency_values else None,
            "max_joint_delta_deg": max(max_delta_values) if max_delta_values else None,
        }
        return {
            "process": ", ".join(process_names),
            "process_status": process_status,
            "status_file": "",
            "status_files": [str(path) for _robot, path, _data in entries],
            "status_data": status_data,
            "status_age_ms": (now - oldest_updated_at) * 1000.0 if oldest_updated_at > 0 else None,
        }

    def _schedule_digital_twin_monitor_sync_restart(
        self,
        target: str,
        cfg: dict[str, Any],
        *,
        gazebo_process: str,
        domains: dict[str, int],
        reason: str,
    ) -> bool:
        target_key = str(target or "").strip()
        if not target_key:
            return False
        now = time.monotonic()
        last_attempt = float(self._digital_twin_sync_restart_last_attempt.get(target_key) or 0.0)
        if (now - last_attempt) < self._DIGITAL_TWIN_SYNC_RESTART_COOLDOWN_S:
            return False
        existing = self._digital_twin_sync_restart_threads.get(target_key)
        if existing is not None and existing.is_alive():
            return False

        self._digital_twin_sync_restart_last_attempt[target_key] = now
        self._write_digital_twin_sync_status(
            target_key,
            cfg,
            {
                "state": "starting",
                "direction": "hardware -> gazebo",
                "message": f"restarting hardware -> gazebo mirror: {reason}",
                "last_error": "",
            },
        )

        def _worker() -> None:
            try:
                err = self._start_digital_twin_sync_when_ready(
                    target_key,
                    cfg,
                    gazebo_process=gazebo_process,
                    domains=domains,
                    direction="hardware -> gazebo",
                )
                if err:
                    self._write_digital_twin_sync_status(
                        target_key,
                        cfg,
                        {
                            "state": "waiting",
                            "direction": "hardware -> gazebo",
                            "message": err,
                            "last_error": err,
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                message = f"hardware -> gazebo mirror restart failed: {exc}"
                log.exception(message)
                self._write_digital_twin_sync_status(
                    target_key,
                    cfg,
                    {
                        "state": "error",
                        "direction": "hardware -> gazebo",
                        "message": message,
                        "last_error": message,
                    },
                )
            finally:
                current = self._digital_twin_sync_restart_threads.get(target_key)
                if current is threading.current_thread():
                    self._digital_twin_sync_restart_threads.pop(target_key, None)

        thread = threading.Thread(
            target=_worker,
            name=f"digital-twin-sync-restart-{target_key}",
            daemon=True,
        )
        self._digital_twin_sync_restart_threads[target_key] = thread
        thread.start()
        return True

    def digital_twin_statuses(self) -> dict[str, dict[str, Any]]:
        domains = self._digital_twin_domain_ids()
        now = time.time()
        result: dict[str, dict[str, Any]] = {}

        for target, cfg in self._DIGITAL_TWIN_TARGETS.items():
            gazebo_name = self._digital_twin_gazebo_launch(target, cfg)
            gazebo_process = str(cfg.get("gazebo_process") or "").strip()
            hardware_supported = bool(cfg.get("hardware_supported", False))
            hardware_robots = tuple(str(r) for r in (cfg.get("hardware") or ()))
            hardware_status = self._digital_twin_hardware_status(cfg)
            hardware_overall = str(hardware_status.get("overall", "unknown"))
            rg2_status = (
                self._ur5e_rg2_gripper_status()
                if "ur5e" in {str(robot).strip().lower() for robot in hardware_robots}
                else {}
            )
            gazebo_status = self.ros2_proc_status(gazebo_process) if gazebo_process else "unknown"
            sync_snapshot = self._digital_twin_sync_status_snapshot(target, cfg, now)
            sync_process = str(sync_snapshot.get("process") or "")
            sync_process_status = str(sync_snapshot.get("process_status") or "unknown")
            status_data = dict(sync_snapshot.get("status_data") or {})
            status_age_ms = sync_snapshot.get("status_age_ms")
            dual_drag_markers_status_file = self._digital_twin_dual_drag_markers_status_path(target)
            dual_drag_markers_status = (
                self._read_json_file(dual_drag_markers_status_file)
                if str(cfg.get("paired_marker_process") or "").strip()
                else {}
            )
            dual_drag_markers_updated_at = float(dual_drag_markers_status.get("updated_at") or 0.0)
            direction = self._digital_twin_direction(target)
            blocked_reason = self._digital_twin_blocked_reason(target, cfg)

            sim_mode = self._digital_twin_sim_mode(target)
            if (
                target == "ur5e only"
                and hardware_supported
                and sim_mode == "monitor"
                and direction == "hardware -> gazebo"
                and gazebo_status == "running"
                and hardware_overall == "running"
            ):
                restart_reason = ""
                if sync_process_status != "running":
                    restart_reason = f"sync process is {sync_process_status}"
                elif status_age_ms is not None and float(status_age_ms) > 6000.0:
                    restart_reason = f"sync status is stale ({float(status_age_ms):.0f} ms old)"
                if restart_reason and self._schedule_digital_twin_monitor_sync_restart(
                    target,
                    cfg,
                    gazebo_process=gazebo_process,
                    domains=domains,
                    reason=restart_reason,
                ):
                    sync_snapshot = self._digital_twin_sync_status_snapshot(target, cfg, time.time())
                    sync_process = str(sync_snapshot.get("process") or "")
                    sync_process_status = str(sync_snapshot.get("process_status") or "unknown")
                    status_data = dict(sync_snapshot.get("status_data") or {})
                    status_age_ms = sync_snapshot.get("status_age_ms")

            if not hardware_supported:
                sync_state = "limited"
                sync_message = self._DUAL_HARDWARE_LIMITATION
            elif sim_mode == "teach":
                # Teach mode uses sim RViz/Gazebo as the authoring surface. Replay in Twin
                # commits validated sim waypoints back to hardware.
                if (
                    sync_process_status == "running"
                    and status_age_ms is not None
                    and status_age_ms <= 3000.0
                ):
                    sync_state = str(status_data.get("state") or "teach")
                    resumed_message = str(status_data.get("message") or "sync process is running.")
                    sync_message = (
                        "Teach: sim RViz controls Gazebo only; hardware -> Gazebo sync "
                        f"resumed after Replay in Twin: {resumed_message}"
                    )
                elif gazebo_status == "running" and hardware_overall == "running":
                    sync_state = "teach"
                    if self._digital_twin_is_dual_robots(cfg):
                        sync_message = (
                            "Teach: sim RViz controls Gazebo only; Replay in Twin commits saved sim waypoints through "
                            "/xarm6/xarm6_traj_controller/follow_joint_trajectory and "
                            "/execute_trajectory."
                        )
                    else:
                        sync_message = (
                            "Teach: sim RViz controls Gazebo only; Replay in Twin commits "
                            "saved sim waypoints through hardware MoveIt."
                        )
                elif gazebo_status == "running" or hardware_overall in {"running", "partial"}:
                    sync_state = "partial"
                    sync_message = "teach stack is partially up; start or stop the full twin."
                else:
                    sync_state = "ready"
                    sync_message = "Ready to start Teach digital twin (sim RViz authoring + hardware commit)."
            elif sync_process_status == "running":
                if status_age_ms is None:
                    sync_state = "starting"
                    sync_message = "sync process is starting."
                elif status_age_ms > 3000.0:
                    sync_state = "stale"
                    sync_message = f"sync status is stale ({status_age_ms:.0f} ms old)."
                else:
                    sync_state = str(status_data.get("state") or "running")
                    sync_message = str(status_data.get("message") or "sync process is running.")
            elif str(status_data.get("state") or "") == "waiting":
                sync_state = "waiting"
                sync_message = str(status_data.get("message") or "sync is waiting.")
            elif gazebo_status == "running" or hardware_overall in {"running", "partial"}:
                sync_state = "partial"
                sync_message = "digital twin stack is partially running; start or stop the full twin."
            else:
                sync_state = "ready"
                sync_message = "Ready to start hardware MoveIt + passive gazebo synched digital twin."

            result[target] = {
                "target": target,
                "supported": hardware_supported,
                "blocked_reason": blocked_reason,
                "domains": {
                    "gazebo": domains["gazebo"],
                    "hardware": domains["hardware"],
                },
                "direction": direction,
                "sim_mode": self._digital_twin_sim_mode(target),
                "sim_modes": list(self._digital_twin_allowed_sim_modes(cfg)) if hardware_supported else [],
                "max_joint_delta_deg": self._DIGITAL_TWIN_MAX_JOINT_DELTA_DEG,
                "gazebo": {
                    "name": gazebo_name,
                    "process": gazebo_process,
                    "status": gazebo_status,
                    "domain": domains["gazebo"],
                    "message": "monitor mode: operator motion should come from hardware MoveIt/RViz.",
                },
                "moviet": {
                    "status": str(hardware_status.get("moveit", hardware_overall)),
                    "message": "Hardware MoveIt/RViz is the operator control surface.",
                },
                "hardware": {
                    "supported": hardware_supported,
                    "robots": list(hardware_robots),
                    "status": hardware_status,
                    "overall": hardware_overall,
                    "domain": domains["hardware"],
                    "domains": {
                        robot: self._digital_twin_hardware_domain_id(cfg, robot, domains)
                        for robot in hardware_robots
                    },
                    "rg2": rg2_status,
                    "message": "" if hardware_supported else self._DUAL_HARDWARE_LIMITATION,
                },
                "sync/status": {
                    "state": sync_state,
                    "process": sync_process,
                    "process_status": sync_process_status,
                    "message": sync_message,
                    "status_file": str(sync_snapshot.get("status_file") or self._digital_twin_status_path(target)),
                    "status_files": list(sync_snapshot.get("status_files") or []),
                    "status_age_ms": status_age_ms,
                    "latency_ms": status_data.get("latency_ms"),
                    "max_joint_delta_deg": status_data.get("max_joint_delta_deg"),
                    "last_error": status_data.get("last_error"),
                },
                "dual_drag_markers": {
                    "status_file": str(dual_drag_markers_status_file),
                    "status_age_ms": (
                        (now - dual_drag_markers_updated_at) * 1000.0
                        if dual_drag_markers_updated_at > 0.0
                        else None
                    ),
                    "state": str(dual_drag_markers_status.get("state") or ""),
                    "action": str(dual_drag_markers_status.get("action") or ""),
                    "stage": str(dual_drag_markers_status.get("stage") or ""),
                    "message": str(dual_drag_markers_status.get("message") or ""),
                    "last_error": str(dual_drag_markers_status.get("last_error") or ""),
                },
            }
        return result

    def _start_tracked_ros2_command(
        self,
        process_name: str,
        command_suffix: str,
        *,
        ros_domain_id: int | None = None,
    ) -> str | None:
        name = str(process_name or "").strip()
        if not name:
            return "process name is empty"
        if self.ros2_proc_status(name) == "running":
            return f"{name} is already running"
        cmd = self._ROS2_ENV + self._ros2_domain_export(ros_domain_id) + str(command_suffix)
        try:
            proc = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            self._ros2_procs[name] = proc
            log.info("Started ROS2 process %s (pid=%d)", name, proc.pid)
            return None
        except Exception as exc:
            return str(exc)

    def _start_ur5e_rtde_trajectory_server(
        self,
        process_name: str,
        *,
        ros_domain_id: int | None = None,
    ) -> str | None:
        name = str(process_name or "").strip()
        if not name:
            return None
        if self.ros2_proc_status(name) != "running":
            prereq_err = self._ros2_launch_prereq_error("hardware_ur5e_rtde_trajectory_server")
            if prereq_err:
                return prereq_err
            command = self._render_ros2_launch_cmd("hardware_ur5e_rtde_trajectory_server")
            err = self._start_tracked_ros2_command(
                name,
                command,
                ros_domain_id=ros_domain_id,
            )
            if err:
                return err
        return self._wait_with_ros2_daemon_retry(
            "ur5e RTDE trajectory server",
            lambda: self._wait_for_ros_action(
                _UR5E_RTDE_TRAJECTORY_ACTION,
                timeout_sec=18.0,
                process_name=name,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )

    def _start_digital_twin_launch(
        self,
        process_name: str,
        launch_name: str,
        *,
        ros_domain_id: int,
        extra_args: str = "",
    ) -> str | None:
        prereq_err = self._ros2_launch_prereq_error(launch_name)
        if prereq_err:
            return prereq_err
        command = self._render_ros2_launch_cmd(launch_name)
        if extra_args.strip():
            command = f"{command} {extra_args.strip()}"
        return self._start_tracked_ros2_command(
            process_name,
            command,
            ros_domain_id=ros_domain_id,
        )

    def _ensure_digital_twin_launch(
        self,
        process_name: str,
        launch_name: str,
        *,
        ros_domain_id: int,
        extra_args: str = "",
    ) -> str | None:
        if self.ros2_proc_status(process_name) == "running":
            return None
        return self._start_digital_twin_launch(
            process_name,
            launch_name,
            ros_domain_id=ros_domain_id,
            extra_args=extra_args,
        )

    def _start_digital_twin_dual_drag_markers(
        self,
        cfg: dict[str, Any],
        *,
        ros_domain_id: int,
        mode: str,
    ) -> str | None:
        process_name = str(cfg.get("paired_marker_process") or "").strip()
        if not process_name:
            return None
        if self.ros2_proc_status(process_name) == "running":
            return None
        script = Path(self._DUAL_DRAG_MARKERS_SCRIPT)
        if not script.is_file():
            return f"dual drag markers script missing: {script}"
        args = [
            "python3.10",
            str(script),
            "--mode",
            str(mode or "monitor"),
            "--status-file",
            str(self._digital_twin_dual_drag_markers_status_path("dual robots")),
        ]
        if str(mode or "monitor").strip().lower() == "monitor":
            args.extend(["--execution-policy", "paired"])
        command = " ".join(shlex.quote(str(part)) for part in args)
        return self._start_tracked_ros2_command(
            process_name,
            command,
            ros_domain_id=ros_domain_id,
        )

    def _wait_for_digital_twin_gazebo_controller_actions(
        self,
        target: str,
        *,
        gazebo_process: str,
        ros_domain_id: int,
    ) -> str | None:
        actions_by_target = {
            "dual robots": (
                "/xarm6_xarm6_traj_controller/follow_joint_trajectory",
                "/xarm6_xarm_gripper_traj_controller/follow_joint_trajectory",
                "/ur5e_joint_trajectory_controller/follow_joint_trajectory",
                "/ur5e_rg2_gripper_traj_controller/follow_joint_trajectory",
            ),
        }
        actions = actions_by_target.get(target, ())
        for action in actions:
            err = self._wait_with_ros2_daemon_retry(
                f"{target} passive gazebo controller {action}",
                lambda action=action: self._wait_for_ros_action(
                    action,
                    timeout_sec=24.0,
                    process_name=gazebo_process,
                    ros_domain_id=ros_domain_id,
                ),
                ros_domain_id=ros_domain_id,
            )
            if err:
                return f"{target} gazebo controller is not ready: {err}"
        return None

    def _digital_twin_dual_robots_processes(
        self,
        cfg: dict[str, Any],
    ) -> tuple[str, str, str, str] | str:
        xarm_processes = self._digital_twin_hardware_processes_for_robot(cfg, "xarm6")
        ur5e_processes = self._digital_twin_hardware_processes_for_robot(cfg, "ur5e")

        xarm_driver_process = str(xarm_processes.get("driver") or "").strip()
        ur5e_driver_process = str(ur5e_processes.get("rtde") or "").strip()
        ur5e_gripper_process = str(ur5e_processes.get("gripper") or "").strip()
        moveit_process = str(
            xarm_processes.get("moveit") or ur5e_processes.get("moveit") or ""
        ).strip()
        if not xarm_driver_process or not ur5e_driver_process or not moveit_process:
            return "hardware is not configured for dual robots"
        return (
            xarm_driver_process,
            ur5e_driver_process,
            ur5e_gripper_process,
            moveit_process,
        )

    def _digital_twin_dual_robots_core_process_names(self, cfg: dict[str, Any]) -> list[str]:
        gazebo_process = str(cfg.get("gazebo_process") or "").strip()
        processes = self._digital_twin_dual_robots_processes(cfg)
        if isinstance(processes, str):
            return [gazebo_process] if gazebo_process else []
        names = [gazebo_process, *processes]
        return [name for name in names if name]

    def _digital_twin_dual_robots_core_started(self, cfg: dict[str, Any]) -> bool:
        names = self._digital_twin_dual_robots_core_process_names(cfg)
        return bool(names) and all(self.ros2_proc_status(name) == "running" for name in names)

    def _start_digital_twin_dual_robots_hardware_launches(
        self,
        cfg: dict[str, Any],
        *,
        ros_domain_id: int,
        launch_rviz: bool = True,
    ) -> str | None:
        processes = self._digital_twin_dual_robots_processes(cfg)
        if isinstance(processes, str):
            return processes
        xarm_driver_process, ur5e_driver_process, ur5e_gripper_process, moveit_process = processes

        err = self._ensure_digital_twin_launch(
            xarm_driver_process,
            "hardware_xarm6_driver",
            ros_domain_id=ros_domain_id,
        )
        if err:
            return err
        err = self._start_ur5e_rtde_trajectory_server(
            ur5e_driver_process,
            ros_domain_id=ros_domain_id,
        )
        if err:
            return err
        err = self._wait_with_ros2_daemon_retry(
            "ur5e RTDE /joint_states publisher",
            lambda: self._wait_for_ros_topic_publisher(
                "/joint_states",
                timeout_sec=20.0,
                process_name=ur5e_driver_process,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )
        if err:
            return f"ur5e RTDE trajectory server is not publishing /joint_states: {err}"
        if ur5e_gripper_process:
            err = self._ensure_digital_twin_launch(
                ur5e_gripper_process,
                "hardware_ur5e_rg2_gripper",
                ros_domain_id=ros_domain_id,
            )
            if err:
                return err
            err = self._wait_with_ros2_daemon_retry(
                "ur5e RG2 gripper bridge",
                lambda: self._wait_for_ros_action(
                    _UR5E_RG2_GRIPPER_ACTION,
                    timeout_sec=12.0,
                    process_name=ur5e_gripper_process,
                    ros_domain_id=ros_domain_id,
                ),
                ros_domain_id=ros_domain_id,
            )
            if err:
                return f"ur5e RG2 gripper bridge is not ready: {err}"
        err = self._ensure_digital_twin_launch(
            moveit_process,
            "hardware_dual_robots_moveit",
            ros_domain_id=ros_domain_id,
            extra_args="" if launch_rviz else "launch_rviz:=false",
        )
        if err:
            return err
        err = self._wait_with_ros2_daemon_retry(
            "dual robots MoveIt execute trajectory",
            lambda: self._wait_for_ros_action(
                "/execute_trajectory",
                timeout_sec=26.0,
                process_name=moveit_process,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )
        if err:
            return f"dual robots MoveIt is not ready: {err}"
        return None

    def _wait_for_digital_twin_dual_robots_hardware_ready(
        self,
        cfg: dict[str, Any],
        *,
        ros_domain_id: int,
        require_moveit: bool = True,
    ) -> str | None:
        processes = self._digital_twin_dual_robots_processes(cfg)
        if isinstance(processes, str):
            return processes
        xarm_driver_process, ur5e_driver_process, ur5e_gripper_process, moveit_process = processes
        readiness_errors: list[str] = []

        self._wait_with_ros2_daemon_retry(
            "xarm6 controller_manager",
            lambda: self._wait_for_ros_service(
                "/xarm6/controller_manager/list_controllers",
                timeout_sec=12.0,
                process_name=xarm_driver_process,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )
        err = self._wait_with_ros2_daemon_retry(
            "xarm6 trajectory controller",
            lambda: self._wait_for_ros_action(
                "/xarm6/xarm6_traj_controller/follow_joint_trajectory",
                timeout_sec=18.0,
                process_name=xarm_driver_process,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )
        if err:
            readiness_errors.append(f"xarm6 trajectory controller is not ready: {err}")
        err = self._wait_with_ros2_daemon_retry(
            "xarm6 /joint_states relay",
            lambda: self._wait_for_ros_topic_publisher(
                "/joint_states",
                timeout_sec=18.0,
                process_name=xarm_driver_process,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )
        if err:
            readiness_errors.append(f"xarm6 driver is not publishing relayed /joint_states: {err}")

        err = self._wait_with_ros2_daemon_retry(
            "ur5e RTDE trajectory action",
            lambda: self._wait_for_ros_action(
                _UR5E_RTDE_TRAJECTORY_ACTION,
                timeout_sec=18.0,
                process_name=ur5e_driver_process,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )
        if err:
            readiness_errors.append(f"ur5e RTDE trajectory action is not ready: {err}")
        err = self._wait_with_ros2_daemon_retry(
            "ur5e RTDE /joint_states publisher",
            lambda: self._wait_for_ros_topic_publisher(
                "/joint_states",
                timeout_sec=20.0,
                process_name=ur5e_driver_process,
                ros_domain_id=ros_domain_id,
            ),
            ros_domain_id=ros_domain_id,
        )
        if err:
            readiness_errors.append(f"ur5e RTDE trajectory server is not publishing /joint_states: {err}")

        if ur5e_gripper_process:
            err = self._wait_with_ros2_daemon_retry(
                "ur5e RG2 gripper bridge",
                lambda: self._wait_for_ros_action(
                    _UR5E_RG2_GRIPPER_ACTION,
                    timeout_sec=12.0,
                    process_name=ur5e_gripper_process,
                    ros_domain_id=ros_domain_id,
                ),
                ros_domain_id=ros_domain_id,
            )
            if err:
                readiness_errors.append(f"ur5e RG2 gripper bridge is not ready: {err}")

        if require_moveit:
            err = self._wait_with_ros2_daemon_retry(
                "dual robots MoveIt execute trajectory",
                lambda: self._wait_for_ros_action(
                    "/execute_trajectory",
                    timeout_sec=26.0,
                    process_name=moveit_process,
                    ros_domain_id=ros_domain_id,
                ),
                ros_domain_id=ros_domain_id,
            )
            if err:
                readiness_errors.append(f"dual robots MoveIt is not ready: {err}")
        if readiness_errors:
            return "; ".join(readiness_errors)
        return None

    def _wait_for_digital_twin_dual_robots_replay_ready(
        self,
        target: str,
        cfg: dict[str, Any],
        *,
        ros_domain_id: int,
    ) -> str | None:
        processes = self._digital_twin_dual_robots_processes(cfg)
        if isinstance(processes, str):
            return processes
        xarm_driver_process, ur5e_driver_process, ur5e_gripper_process, moveit_process = processes

        missing: list[str] = []
        for label, process_name in (
            ("xarm6 driver", xarm_driver_process),
            ("ur5e driver", ur5e_driver_process),
            ("ur5e RG2 gripper bridge", ur5e_gripper_process),
            ("dual robots MoveIt", moveit_process),
        ):
            name = str(process_name or "").strip()
            if not name:
                continue
            state = self.ros2_proc_status(name)
            if state != "running":
                missing.append(f"{label} process {name} is {state}")
        if missing:
            return "; ".join(missing)
        return None

    def _start_digital_twin_dual_robots_hardware_stack(
        self,
        target: str,
        cfg: dict[str, Any],
        *,
        ros_domain_id: int,
        domain_ids: dict[str, int] | None = None,
    ) -> str | None:
        domains = domain_ids or {"hardware": ros_domain_id}
        hardware_domain_id = int(domains.get("hardware", ros_domain_id))
        err = self._start_digital_twin_dual_robots_hardware_launches(
            cfg,
            ros_domain_id=hardware_domain_id,
        )
        if err:
            return err
        return self._wait_for_digital_twin_dual_robots_hardware_ready(
            cfg,
            ros_domain_id=hardware_domain_id,
        )

    def _digital_twin_hardware_reachability_error(
        self,
        target: str,
        cfg: dict[str, Any],
    ) -> str | None:
        robots = tuple(str(r) for r in (cfg.get("hardware") or ()))
        if not robots:
            return f"hardware is not configured for {target}"
        hw_links = self.hardware_connection_statuses(force=True)
        for robot in robots:
            entry = hw_links.get(robot, {})
            if not entry.get("reachable", False):
                ip = str(entry.get("ip", "")).strip() or "(unknown IP)"
                msg = str(entry.get("message", "unreachable")).strip()
                return f"{robot} hardware is unreachable at {ip} ({msg})"
        return None

    def _start_digital_twin_hardware_stack(
        self,
        target: str,
        cfg: dict[str, Any],
        *,
        ros_domain_id: int,
        domain_ids: dict[str, int] | None = None,
    ) -> str | None:
        robots = tuple(str(r) for r in (cfg.get("hardware") or ()))
        if not robots:
            return f"hardware is not configured for {target}"
        reachability_error = self._digital_twin_hardware_reachability_error(target, cfg)
        if reachability_error:
            return reachability_error

        hardware_processes = cfg.get("hardware_processes") or {}
        if not isinstance(hardware_processes, dict):
            return f"hardware is not configured for {target}"

        if target == "dual robots":
            return self._start_digital_twin_dual_robots_hardware_stack(
                target,
                cfg,
                ros_domain_id=ros_domain_id,
                domain_ids=domain_ids,
            )

        # In teach mode the sim MoveIt RViz (domain 41) is the authoring surface, so
        # suppress the redundant hardware-side RViz to avoid two confusing windows.
        teach_mode = self._digital_twin_sim_mode(target) == "teach"

        domains = domain_ids or {"hardware": ros_domain_id}
        for robot in robots:
            robot_domain_id = self._digital_twin_hardware_domain_id(cfg, robot, domains)
            robot_processes = self._digital_twin_hardware_processes_for_robot(cfg, robot)

            if robot == "xarm6":
                moveit_process = str(robot_processes.get("moveit") or "").strip()
                err = self._start_digital_twin_launch(
                    moveit_process,
                    "hardware_xarm6_moveit",
                    ros_domain_id=robot_domain_id,
                    extra_args="show_rviz:=false" if teach_mode else "",
                )
                if err:
                    return err
                err = self._wait_for_ros_service(
                    "/controller_manager/list_controllers",
                    timeout_sec=22.0,
                    process_name=moveit_process,
                    ros_domain_id=robot_domain_id,
                )
                if err:
                    return f"{robot} MoveIt is not ready: {err}"
                continue

            if robot == "ur5e":
                rtde_process = str(robot_processes.get("rtde") or "").strip()
                gripper_process = str(robot_processes.get("gripper") or "").strip()
                moveit_process = str(robot_processes.get("moveit") or "").strip()
                err = self._start_ur5e_rtde_trajectory_server(
                    rtde_process,
                    ros_domain_id=robot_domain_id,
                )
                if err:
                    return err
                extra_args = "launch_rviz:=false" if teach_mode else ""
                err = self._start_digital_twin_launch(
                    moveit_process,
                    "hardware_ur5e_moveit",
                    ros_domain_id=robot_domain_id,
                    extra_args=extra_args,
                )
                if err:
                    return err
                err = self._wait_with_ros2_daemon_retry(
                    f"{robot} MoveIt execute trajectory",
                    lambda: self._wait_for_ros_action(
                        "/execute_trajectory",
                        timeout_sec=22.0,
                        process_name=moveit_process,
                        ros_domain_id=robot_domain_id,
                    ),
                    ros_domain_id=robot_domain_id,
                )
                if err:
                    return f"{robot} MoveIt is not ready: {err}"
                err = self._wait_with_ros2_daemon_retry(
                    f"{robot} RTDE /joint_states publisher",
                    lambda: self._wait_for_ros_topic_publisher(
                        "/joint_states",
                        timeout_sec=20.0,
                        process_name=rtde_process,
                        ros_domain_id=robot_domain_id,
                    ),
                    ros_domain_id=robot_domain_id,
                )
                if err:
                    return f"{robot} RTDE trajectory server is not publishing /joint_states: {err}"
                if gripper_process:
                    err = self._start_digital_twin_launch(
                        gripper_process,
                        "hardware_ur5e_rg2_gripper",
                        ros_domain_id=robot_domain_id,
                    )
                    if err:
                        return err
                    err = self._wait_with_ros2_daemon_retry(
                        f"{robot} RG2 gripper bridge",
                        lambda: self._wait_for_ros_action(
                            _UR5E_RG2_GRIPPER_ACTION,
                            timeout_sec=12.0,
                            process_name=gripper_process,
                            ros_domain_id=robot_domain_id,
                        ),
                        ros_domain_id=robot_domain_id,
                    )
                    if err:
                        return f"{robot} RG2 gripper bridge is not ready: {err}"
                continue

            return f"unknown hardware robot: {robot}"
        return None

    def _start_digital_twin_sync_process(
        self,
        target: str,
        cfg: dict[str, Any],
        *,
        gazebo_domain_id: int,
        hardware_domain_id: int,
        domain_ids: dict[str, int] | None = None,
    ) -> str | None:
        if not self._DIGITAL_TWIN_SYNC_SCRIPT.is_file():
            return f"digital twin sync script missing: {self._DIGITAL_TWIN_SYNC_SCRIPT}"
        model_name = str(cfg.get("model_name") or "").strip()
        domains = domain_ids or {"hardware": hardware_domain_id, "gazebo": gazebo_domain_id}
        sync_items = self._digital_twin_sync_process_items(cfg)
        for robot, sync_process in sync_items:
            robot_key = str(robot or "").strip().lower()
            robot_hardware_domain_id = self._digital_twin_hardware_domain_id(cfg, robot_key, domains)
            status_path = (
                self._digital_twin_sync_status_path(target, robot_key)
                if len(sync_items) > 1
                else self._digital_twin_status_path(target)
            )
            args = [
                "python3.10",
                str(self._DIGITAL_TWIN_SYNC_SCRIPT),
                "--mode",
                "mirror",
                "--target",
                target,
                "--robot",
                robot_key,
                "--model-name",
                model_name,
                "--gazebo-domain-id",
                str(gazebo_domain_id),
                "--hardware-domain-id",
                str(robot_hardware_domain_id),
                "--status-file",
                str(status_path),
                "--direction-file",
                str(self._digital_twin_direction_path(target)),
            ]
            if robot_key == "ur5e":
                args.extend([
                    "--ur5e-hardware-trajectory-action",
                    _UR5E_RTDE_TRAJECTORY_ACTION,
                ])
            command = " ".join(shlex.quote(part) for part in args)
            if self.ros2_proc_status(sync_process) == "running":
                continue
            err = self._start_tracked_ros2_command(sync_process, command)
            if err:
                return err
        return None

    def _initialize_digital_twin_gazebo_from_hardware(
        self,
        target: str,
        cfg: dict[str, Any],
        *,
        gazebo_process: str,
        domains: dict[str, int],
    ) -> str | None:
        if target == "dual robots":
            processes = self._digital_twin_dual_robots_processes(cfg)
            if isinstance(processes, str):
                return processes
            _xarm_driver_process, ur5e_driver_process, _ur5e_gripper_process, _moveit_process = processes
            err = self._wait_with_ros2_daemon_retry(
                "ur5e RTDE feedback for initial gazebo pose",
                lambda: self._wait_for_ros_topic_publisher(
                    "/joint_states",
                    timeout_sec=20.0,
                    process_name=ur5e_driver_process,
                    ros_domain_id=domains["hardware"],
                ),
                ros_domain_id=domains["hardware"],
            )
            if err:
                return f"ur5e RTDE feedback is not ready for initial gazebo pose: {err}"

        err = self._wait_for_ros_service(
            "/controller_manager/list_controllers",
            timeout_sec=35.0,
            process_name=gazebo_process,
            ros_domain_id=domains["gazebo"],
        )
        if err:
            return f"{target} gazebo is not ready for initial hardware pose: {err}"

        model_name = str(cfg.get("model_name") or "").strip()
        sync_items = self._digital_twin_sync_process_items(cfg)
        for robot, _sync_process in sync_items:
            robot_key = str(robot or "").strip().lower()
            if robot_key not in {"xarm6", "ur5e"}:
                continue
            robot_hardware_domain_id = self._digital_twin_hardware_domain_id(
                cfg,
                robot_key,
                domains,
            )
            status_path = (
                self._digital_twin_sync_status_path(target, robot_key)
                if len(sync_items) > 1
                else self._digital_twin_status_path(target)
            )
            result = self._run_digital_twin_sync(
                [
                    "--mode", "initialize-gazebo-from-hardware",
                    "--target", target,
                    "--robot", robot_key,
                    "--model-name", model_name,
                    "--gazebo-domain-id", str(domains["gazebo"]),
                    "--hardware-domain-id", str(robot_hardware_domain_id),
                    "--status-file", str(status_path),
                    "--direction-file", str(self._digital_twin_direction_path(target)),
                ],
                timeout_sec=self._DIGITAL_TWIN_INITIALIZE_TIMEOUT_S,
            )
            if not result.get("success"):
                details: list[str] = []
                if result.get("max_joint_delta_rad") is not None:
                    details.append(f"max_joint_delta_rad={float(result.get('max_joint_delta_rad') or 0.0):.4f}")
                if result.get("max_joint_delta_joint"):
                    details.append(f"max_joint_delta_joint={str(result.get('max_joint_delta_joint') or '')}")
                if result.get("attempts") is not None:
                    details.append(f"attempts={int(result.get('attempts') or 0)}")
                if result.get("init_tolerance_rad") is not None:
                    details.append(f"init_tolerance_rad={float(result.get('init_tolerance_rad') or 0.0):.4f}")
                detail_suffix = f" ({'; '.join(details)})" if details else ""
                return (
                    f"{robot_key} gazebo initial hardware pose failed: "
                    f"{str(result.get('message') or 'unknown error')}{detail_suffix}"
                )
        return None

    def _stop_digital_twin_mirror_workers_for_target(
        self,
        target: str,
        cfg: dict[str, Any],
    ) -> None:
        for _robot, process_name in self._digital_twin_sync_process_items(cfg):
            process = str(process_name or "").strip()
            if process:
                self.ros2_stop(process, reason="digital_twin_replay")

        status_paths: list[Path] = []
        items = self._digital_twin_sync_process_items(cfg)
        if len(items) <= 1:
            status_paths.append(self._digital_twin_status_path(target))
        else:
            for robot, _process in items:
                robot_key = str(robot or "").strip().lower()
                if robot_key:
                    status_paths.append(self._digital_twin_sync_status_path(target, robot_key))

        for path in status_paths:
            pattern = (
                r"digital_twin_sync\.py.*--mode mirror.*--status-file "
                + re.escape(str(path))
            )
            subprocess.run(["pkill", "-TERM", "-f", pattern], capture_output=True)
        time.sleep(0.2)
        for path in status_paths:
            pattern = (
                r"digital_twin_sync\.py.*--mode mirror.*--status-file "
                + re.escape(str(path))
            )
            subprocess.run(["pkill", "-KILL", "-f", pattern], capture_output=True)

    def _start_digital_twin_sync_when_ready(
        self,
        target: str,
        cfg: dict[str, Any],
        *,
        gazebo_process: str,
        domains: dict[str, int],
        direction: str | None = None,
    ) -> str | None:
        sync_direction = str(direction or self._digital_twin_direction(target))
        self._write_digital_twin_direction(target, sync_direction)
        if target == "dual robots":
            self._write_digital_twin_sync_status(
                target,
                cfg,
                {
                    "state": "starting",
                    "direction": sync_direction,
                    "message": "starting sync workers",
                    "last_error": "",
                },
            )

        err = self._wait_for_ros_service(
            "/controller_manager/list_controllers",
            timeout_sec=35.0,
            process_name=gazebo_process,
            ros_domain_id=domains["gazebo"],
        )
        if err:
            message = f"{target} gazebo is not ready for sync: {err}"
            self._write_digital_twin_sync_status(
                target,
                cfg,
                {
                    "state": "waiting",
                    "direction": sync_direction,
                    "message": message,
                    "last_error": message,
                },
            )
            return None

        err = self._start_digital_twin_sync_process(
            target,
            cfg,
            gazebo_domain_id=domains["gazebo"],
            hardware_domain_id=domains["hardware"],
            domain_ids=domains,
        )
        if err:
            return err

        self._write_digital_twin_sync_status(
            target,
            cfg,
            {
                "state": "starting",
                "direction": sync_direction,
                "message": "sync process running",
                "last_error": "",
            },
        )
        self._write_digital_twin_status(
            target,
            {
                "state": "running",
                "direction": sync_direction,
                "message": "sync process started.",
            },
        )
        return None

    def _digital_twin_dual_robots_already_started(self, cfg: dict[str, Any]) -> bool:
        return self._digital_twin_dual_robots_core_started(cfg)

    def digital_twin_start(self, target: str) -> str | None:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return f"unknown digital twin target: {target}"
        if not bool(cfg.get("hardware_supported", False)):
            return self._DUAL_HARDWARE_LIMITATION

        blocked = self._digital_twin_blocked_reason(target, cfg)
        if blocked:
            return blocked

        domains = self._digital_twin_domain_ids()
        gazebo_name = self._digital_twin_gazebo_launch(target, cfg)
        gazebo_process = str(cfg.get("gazebo_process") or "").strip()

        self.robot_env = "real"
        self._stop_teleop_server()
        self._write_digital_twin_direction(target, self._digital_twin_direction(target))
        self._write_digital_twin_status(
            target,
            {
                "state": "starting",
                "direction": self._digital_twin_direction(target),
                "message": "starting digital twin stack.",
            },
        )

        dual_already_started = (
            target == "dual robots" and self._digital_twin_dual_robots_already_started(cfg)
        )
        if dual_already_started:
            self._write_digital_twin_status(
                target,
                {
                    "state": "starting",
                    "direction": self._digital_twin_direction(target),
                    "message": "retrying dual robots sync with existing hardware MoveIt/RViz.",
                },
            )
            self._restart_ros2_daemon_for_discovery(ros_domain_id=domains["hardware"])
        else:
            self._shutdown_gazebo_prewarm_controllers()
            self._force_kill_digital_twin_helpers()
            self._kill_stale_gazebo_helpers()
            self._force_kill_gazebo_core(reason="digital_twin_prelaunch_restart")

        if target == "dual robots":
            teach_mode = self._digital_twin_sim_mode(target) == "teach"
            if not dual_already_started:
                reachability_error = self._digital_twin_hardware_reachability_error(target, cfg)
                if reachability_error:
                    self._write_digital_twin_status(
                        target,
                        {
                            "state": "waiting",
                            "direction": self._digital_twin_direction(target),
                            "message": reachability_error,
                            "last_error": reachability_error,
                        },
                    )
                    self._write_digital_twin_sync_status(
                        target,
                        cfg,
                        {
                            "state": "waiting",
                            "direction": self._digital_twin_direction(target),
                            "message": reachability_error,
                            "last_error": reachability_error,
                        },
                    )
                    return reachability_error

            err = self._start_digital_twin_dual_robots_hardware_launches(
                cfg,
                ros_domain_id=domains["hardware"],
                launch_rviz=not teach_mode,
            )
            if err:
                self._write_digital_twin_status(
                    target,
                    {
                        "state": "waiting",
                        "direction": self._digital_twin_direction(target),
                        "message": err,
                        "last_error": err,
                    },
                )
                self._write_digital_twin_sync_status(
                    target,
                    cfg,
                    {
                        "state": "waiting",
                        "direction": self._digital_twin_direction(target),
                        "message": err,
                        "last_error": err,
                    },
                )
                return err
            if not teach_mode:
                err = self._start_digital_twin_dual_drag_markers(
                    cfg,
                    ros_domain_id=domains["hardware"],
                    mode="monitor",
                )
                if err:
                    self._write_digital_twin_status(
                        target,
                        {
                            "state": "partial",
                            "direction": self._digital_twin_direction(target),
                            "message": err,
                            "last_error": err,
                        },
                    )
                    self._write_digital_twin_sync_status(
                        target,
                        cfg,
                        {
                            "state": "partial",
                            "direction": self._digital_twin_direction(target),
                            "message": err,
                            "last_error": err,
                        },
                    )
                    return err

            err = self._ensure_digital_twin_launch(
                gazebo_process,
                gazebo_name,
                ros_domain_id=domains["gazebo"],
            )
            if err:
                self._write_digital_twin_status(
                    target,
                    {
                        "state": "partial",
                        "direction": self._digital_twin_direction(target),
                        "message": err,
                        "last_error": err,
                    },
                )
                self._write_digital_twin_sync_status(
                    target,
                    cfg,
                    {
                        "state": "partial",
                        "direction": self._digital_twin_direction(target),
                        "message": err,
                        "last_error": err,
                    },
                )
                return err

            init_err = None
            if teach_mode or not dual_already_started:
                init_err = self._initialize_digital_twin_gazebo_from_hardware(
                    target,
                    cfg,
                    gazebo_process=gazebo_process,
                    domains=domains,
                )

            if teach_mode:
                if init_err:
                    self._write_digital_twin_status(
                        target,
                        {
                            "state": "waiting",
                            "direction": "gazebo -> hardware",
                            "message": init_err,
                        },
                    )
                    return init_err
                gazebo_moveit_process = str(cfg.get("gazebo_moveit_process") or "").strip()
                gazebo_moveit_launch = str(cfg.get("gazebo_moveit_launch") or "").strip()
                if gazebo_moveit_process and gazebo_moveit_launch:
                    err = self._ensure_digital_twin_launch(
                        gazebo_moveit_process,
                        gazebo_moveit_launch,
                        ros_domain_id=domains["gazebo"],
                    )
                    if err:
                        return err
                err = self._start_digital_twin_dual_drag_markers(
                    cfg,
                    ros_domain_id=domains["gazebo"],
                    mode="teach",
                )
                if err:
                    return err
                self._write_digital_twin_status(
                    target,
                    {
                        "state": "teach",
                        "direction": "gazebo -> hardware",
                        "message": (
                            "Teach: sim RViz controls Gazebo only; Replay in Twin commits saved sim waypoints through "
                            "/xarm6/xarm6_traj_controller/follow_joint_trajectory and "
                            "/execute_trajectory."
                        ),
                    },
                )
                return None
            if init_err:
                self._write_digital_twin_sync_status(
                    target,
                    cfg,
                    {
                        "state": "waiting",
                        "direction": self._digital_twin_direction(target),
                        "message": init_err,
                        "last_error": init_err,
                    },
                )

            return self._start_digital_twin_sync_when_ready(
                target,
                cfg,
                gazebo_process=gazebo_process,
                domains=domains,
            )

        err = self._start_digital_twin_launch(
            gazebo_process,
            gazebo_name,
            ros_domain_id=domains["gazebo"],
        )
        if err:
            self._write_digital_twin_status(
                target,
                {
                    "state": "partial",
                    "direction": self._digital_twin_direction(target),
                    "message": err,
                    "last_error": err,
                },
            )
            return err

        err = self._start_digital_twin_hardware_stack(
            target,
            cfg,
            ros_domain_id=domains["hardware"],
            domain_ids=domains,
        )
        if err:
            self._write_digital_twin_status(
                target,
                {
                    "state": "partial",
                    "direction": self._digital_twin_direction(target),
                    "message": err,
                    "last_error": err,
                },
            )
            return err

        init_err = self._initialize_digital_twin_gazebo_from_hardware(
            target,
            cfg,
            gazebo_process=gazebo_process,
            domains=domains,
        )

        # Teach mode is sim-leads: the operator drives the sim via MoveIt, so the live
        # hardware -> gazebo mirror must NOT run (it would overwrite MoveIt's motion every
        # cycle). Capture/replay use their own one-shot sync subprocesses, not this mirror.
        if self._digital_twin_sim_mode(target) == "teach":
            if init_err:
                self._write_digital_twin_status(
                    target,
                    {
                        "state": "waiting",
                        "direction": "gazebo -> hardware",
                        "message": init_err,
                    },
                )
                return init_err
            self._write_digital_twin_status(
                target,
                {
                    "state": "teach",
                    "direction": "gazebo -> hardware",
                    "message": (
                        "Teach: sim RViz controls Gazebo only; Replay in Twin commits "
                        "saved sim waypoints through hardware MoveIt."
                    ),
                },
            )
            return None
        if init_err:
            self._write_digital_twin_sync_status(
                target,
                cfg,
                {
                    "state": "waiting",
                    "direction": self._digital_twin_direction(target),
                    "message": init_err,
                    "last_error": init_err,
                },
            )

        err = self._start_digital_twin_sync_when_ready(
            target,
            cfg,
            gazebo_process=gazebo_process,
            domains=domains,
        )
        if err:
            self._write_digital_twin_sync_status(
                target,
                cfg,
                {
                    "state": "waiting",
                    "direction": self._digital_twin_direction(target),
                    "message": err,
                    "last_error": err,
                },
            )
        return err

    def digital_twin_stop(self, target: str) -> str | None:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return f"unknown digital twin target: {target}"

        for process_name in reversed(self._digital_twin_process_names(cfg)):
            self.ros2_stop(process_name, reason="digital_twin_stop")
        self._stop_teleop_server()
        # Force-kill stragglers (move_group/rviz2/sync, gazebo core) so the twin fully
        # shuts down and the sim mode can be switched without leftovers.
        self._force_kill_digital_twin_helpers()
        self._kill_stale_gazebo_helpers()
        self._force_kill_gazebo_core(reason="digital_twin_stop")
        self._write_digital_twin_status(
            target,
            {
                "state": "stopped",
                "direction": self._digital_twin_direction(target),
                "message": "digital twin stopped.",
            },
        )
        return None

    def digital_twin_apply_gazebo_to_hardware(self, target: str) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        if not bool(cfg.get("hardware_supported", False)):
            return {"success": False, "message": self._DUAL_HARDWARE_LIMITATION}
        if self._digital_twin_direction(target) != "gazebo -> hardware":
            return {
                "success": False,
                "message": "Set direction to gazebo -> hardware before applying to hardware.",
            }
        if self.ros2_proc_status(str(cfg.get("gazebo_process") or "")) != "running":
            return {"success": False, "message": f"{target} gazebo is not running."}
        domains = self._digital_twin_domain_ids()
        if self._digital_twin_hardware_status(cfg).get("overall") != "running":
            return {"success": False, "message": f"{target} hardware is not running."}

        args = [
            "python3.10",
            str(self._DIGITAL_TWIN_SYNC_SCRIPT),
            "--mode",
            "apply-gazebo-to-hardware",
            "--target",
            target,
            "--robot",
            str(cfg.get("robot") or ""),
            "--model-name",
            str(cfg.get("model_name") or ""),
            "--gazebo-domain-id",
            str(domains["gazebo"]),
            "--hardware-domain-id",
            str(domains["hardware"]),
            "--status-file",
            str(self._digital_twin_status_path(target)),
            "--direction-file",
            str(self._digital_twin_direction_path(target)),
            "--max-joint-delta-deg",
            str(self._DIGITAL_TWIN_MAX_JOINT_DELTA_DEG),
        ]
        command = " ".join(shlex.quote(part) for part in args)
        full_cmd = self._ROS2_ENV + command
        try:
            cp = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "gazebo -> hardware apply timed out."}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

        output = (cp.stdout or "").strip()
        result: dict[str, Any] = {}
        if output:
            try:
                result = json.loads(output.splitlines()[-1])
            except Exception:
                result = {}
        if not result:
            detail = self._tail_output(cp.stderr) or self._tail_output(cp.stdout)
            result = {
                "success": cp.returncode == 0,
                "message": detail or f"gazebo -> hardware exited with code {cp.returncode}",
            }
        if cp.returncode != 0:
            result["success"] = False
        return result

    # ── Manual record & replay (gazebo -> hardware) ──────────────────────────
    @staticmethod
    def _sync_arg_value(extra_args: list[str], name: str) -> str:
        try:
            index = extra_args.index(name)
            return str(extra_args[index + 1])
        except Exception:
            return ""

    def _run_digital_twin_sync(self, extra_args: list[str], timeout_sec: float = 60.0) -> dict[str, Any]:
        """Run the digital twin sync helper in a one-shot mode and parse its JSON output."""
        extra_args = list(extra_args)
        if "--ur5e-hardware-trajectory-action" not in extra_args:
            robot_arg = self._sync_arg_value(extra_args, "--robot")
            target_arg = self._sync_arg_value(extra_args, "--target")
            if str(robot_arg or "").strip().lower() == "ur5e" or str(target_arg or "").strip().lower() in {"ur5e only", "dual robots"}:
                extra_args.extend([
                    "--ur5e-hardware-trajectory-action",
                    _UR5E_RTDE_TRAJECTORY_ACTION,
                ])
        args = ["python3.10", str(self._DIGITAL_TWIN_SYNC_SCRIPT), *extra_args]
        command = " ".join(shlex.quote(part) for part in args)
        full_cmd = self._ROS2_ENV + command
        try:
            cp = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            mode = self._sync_arg_value(extra_args, "--mode")
            robot = self._sync_arg_value(extra_args, "--robot")
            detail = " ".join(part for part in (f"mode={mode}" if mode else "", f"robot={robot}" if robot else "") if part)
            suffix = f" ({detail})" if detail else ""
            return {
                "success": False,
                "message": f"digital twin sync helper timed out after {float(timeout_sec):.0f}s{suffix}.",
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

        output = (cp.stdout or "").strip()
        result: dict[str, Any] = {}
        if output:
            try:
                result = json.loads(output.splitlines()[-1])
            except Exception:
                result = {}
        if not result:
            detail = self._tail_output(cp.stderr) or self._tail_output(cp.stdout)
            result = {
                "success": cp.returncode == 0,
                "message": detail or f"helper exited with code {cp.returncode}",
            }
        if cp.returncode != 0:
            result["success"] = False
        return result

    def _digital_twin_recordings_dir(self) -> Path:
        path = _PROJECT_ROOT / "cais_spade_llm" / "monitor" / "digital_twin" / "recordings"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def digital_twin_function_names(cls) -> list[str]:
        return list(cls._ROBOT_FUNCTION_NAMES)

    def digital_twin_saved_function_names(self, target: str, robot: str) -> list[str]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return []
        storage_source = self._robot_function_storage_source(target, cfg)
        robot_key = str(robot or "").strip().lower()
        directory = _ROBOT_TAUGHT_FUNCTIONS_DIR / robot_key
        if not directory.is_dir():
            return []
        suffix = f"__{storage_source}.json"
        names: list[str] = []
        for function_dir in directory.iterdir():
            if not function_dir.is_dir() or function_dir.name.startswith("."):
                continue
            if (function_dir / f"default{suffix}").is_file():
                names.append(function_dir.name)
        return sorted(dict.fromkeys(names))

    @classmethod
    def digital_twin_function_template(cls, function_name: str) -> list[dict[str, str]]:
        return [
            dict(step)
            for step in cls._ROBOT_FUNCTION_TEMPLATES.get(str(function_name or "").strip(), [])
        ]

    def digital_twin_target_robots(self, target: str) -> list[str]:
        cfg = self._digital_twin_target(target) or {}
        if self._digital_twin_is_dual_robots(cfg):
            return self._digital_twin_dual_robot_keys(cfg)
        robot = str(cfg.get("robot") or "").strip().lower()
        return [robot] if robot in {"xarm6", "ur5e"} else []

    @classmethod
    def _robot_function_template_step(
        cls,
        function_name: str,
        step_name: str,
    ) -> dict[str, str] | None:
        return digital_twin.robot_function_template_step(
            cls._ROBOT_FUNCTION_TEMPLATES,
            function_name,
            step_name,
        )

    @staticmethod
    def _robot_function_safe_name(name: object) -> str:
        return digital_twin.robot_function_safe_name(name)

    @classmethod
    def _robot_function_safe_function_name(cls, function_name: object) -> str:
        return digital_twin.robot_function_safe_function_name(function_name)

    @staticmethod
    def _robot_function_buffer_key(
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> str:
        return digital_twin.robot_function_buffer_key(target, robot, function_name, name)

    @staticmethod
    def _robot_function_display_path(path: Path) -> str:
        return digital_twin.robot_function_display_path(path, _PROJECT_ROOT)

    @staticmethod
    def _robot_function_launch_mode_from_target(_target: str, _cfg: dict[str, Any]) -> str:
        return digital_twin.robot_function_launch_mode_from_target(_target, _cfg)

    @staticmethod
    def _robot_function_storage_source_for_launch(launch_mode: str) -> str:
        return digital_twin.robot_function_storage_source_for_launch(launch_mode)

    def _robot_function_storage_source(self, target: str, cfg: dict[str, Any]) -> str:
        launch_mode = self._robot_function_launch_mode_from_target(target, cfg)
        return self._robot_function_storage_source_for_launch(launch_mode)

    def _robot_function_capture_source(self, target: str, cfg: dict[str, Any]) -> str:
        # monitor = hardware leads, teach = gazebo leads.
        return "gazebo" if self._digital_twin_sim_mode(target) == "teach" else "hardware"

    def _robot_function_path(
        self,
        robot: str,
        function_name: str,
        name: str,
        storage_source: str,
    ) -> Path:
        return digital_twin.robot_function_path(
            _ROBOT_TAUGHT_FUNCTIONS_DIR,
            robot,
            function_name,
            name,
            storage_source,
        )

    def digital_twin_function_info(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        robot_key = str(robot or "").strip().lower()
        if robot_key not in {"xarm6", "ur5e"}:
            return {"success": False, "message": f"unknown robot: {robot}"}
        function_key = self._robot_function_safe_function_name(function_name)
        if not function_key:
            return {"success": False, "message": "function name is empty"}
        launch_mode = self._robot_function_launch_mode_from_target(target, cfg)
        storage_source = self._robot_function_storage_source_for_launch(launch_mode)
        capture_source = self._robot_function_capture_source(target, cfg)
        path = self._robot_function_path(robot_key, function_key, name, storage_source)
        return {
            "success": True,
            "launch_mode": launch_mode,
            "storage_source": storage_source,
            "capture_source": capture_source,
            "path": str(path),
            "display_path": self._robot_function_display_path(path),
        }

    def digital_twin_function_step_count(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> int:
        key = self._robot_function_buffer_key(
            target,
            robot,
            self._robot_function_safe_function_name(function_name),
            self._robot_function_safe_name(name),
        )
        with self._digital_twin_record_lock:
            return len(list(self._digital_twin_function_steps.get(key, [])))

    def digital_twin_list_function_buffer_steps(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> list[dict[str, Any]]:
        key = self._robot_function_buffer_key(
            target,
            robot,
            self._robot_function_safe_function_name(function_name),
            self._robot_function_safe_name(name),
        )
        with self._digital_twin_record_lock:
            steps = deepcopy(list(self._digital_twin_function_steps.get(key, [])))
        out: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            waypoint = dict(step.get("waypoint") or {})
            positions = waypoint.get("joint_positions") or waypoint.get("positions") or []
            out.append(
                {
                    "index": index,
                    "step_name": str(step.get("step_name") or ""),
                    "primitive": str(step.get("primitive") or ""),
                    "has_waypoint": bool(positions),
                    "joint_positions": [float(v) for v in positions],
                    "source": str(waypoint.get("source") or step.get("capture_source") or ""),
                }
            )
        return out

    def digital_twin_list_function_files(
        self,
        target: str,
        robot: str,
        function_name: str,
    ) -> list[str]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return []
        storage_source = self._robot_function_storage_source(target, cfg)
        robot_key = str(robot or "").strip().lower()
        function_key = self._robot_function_safe_function_name(function_name)
        suffix = f"__{storage_source}.json"
        directory = _ROBOT_TAUGHT_FUNCTIONS_DIR / robot_key / function_key
        if not directory.is_dir():
            return []
        return sorted(
            p.name[:-len(suffix)]
            for p in directory.glob(f"*{suffix}")
            if p.is_file() and p.name.endswith(suffix)
        )

    def digital_twin_list_function_file_steps(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> list[dict[str, Any]]:
        payload, _path, err = self._robot_function_file_payload(target, robot, function_name, name)
        if err or payload is None:
            return []
        out: list[dict[str, Any]] = []
        for index, step in enumerate(list(payload.get("steps") or [])):
            body = self._robot_function_step_waypoint(dict(step))
            out.append(
                {
                    "index": index,
                    "step_name": str(dict(step).get("step_name") or ""),
                    "primitive": str(dict(step).get("primitive") or ""),
                    "has_waypoint": body is not None,
                }
            )
        return out

    def digital_twin_clear_function_steps(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> None:
        key = self._robot_function_buffer_key(
            target,
            robot,
            self._robot_function_safe_function_name(function_name),
            self._robot_function_safe_name(name),
        )
        with self._digital_twin_record_lock:
            self._digital_twin_function_steps.pop(key, None)

    def digital_twin_delete_function_buffer_step(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
        index: int,
    ) -> dict[str, Any]:
        key = self._robot_function_buffer_key(
            target,
            robot,
            self._robot_function_safe_function_name(function_name),
            self._robot_function_safe_name(name),
        )
        try:
            step_index = int(index)
        except Exception:
            return {"success": False, "message": "Invalid step index."}
        with self._digital_twin_record_lock:
            steps = self._digital_twin_function_steps.get(key)
            if not steps:
                return {"success": False, "message": "No unsaved steps to delete."}
            if step_index < 0 or step_index >= len(steps):
                return {"success": False, "message": f"Step #{step_index + 1} does not exist."}
            removed = steps.pop(step_index)
            if not steps:
                self._digital_twin_function_steps.pop(key, None)
        step_name = str(dict(removed).get("step_name") or f"step {step_index + 1}")
        return {"success": True, "message": f"Deleted unsaved step: {step_name}"}

    def _robot_function_validate_request(
        self,
        target: str,
        robot: str,
        function_name: str,
    ) -> tuple[dict[str, Any] | None, str]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return None, f"unknown digital twin target: {target}"
        robot_key = str(robot or "").strip().lower()
        if robot_key not in {"xarm6", "ur5e"}:
            return None, f"unknown robot: {robot}"
        if robot_key not in tuple(str(r).strip().lower() for r in (cfg.get("hardware") or ())):
            return None, f"{robot_key} is not part of {target}."
        function_key = self._robot_function_safe_function_name(function_name)
        if not function_key:
            return None, "function name is empty"
        return cfg, ""

    def digital_twin_capture_function_step(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
        step_name: str,
        primitive: str = "move_cartesian",
    ) -> dict[str, Any]:
        cfg, err = self._robot_function_validate_request(target, robot, function_name)
        if err or cfg is None:
            return {"success": False, "message": err}
        step_key = self._robot_function_safe_name(step_name)
        primitive = str(primitive or "move_cartesian").strip()
        if primitive not in {
            "move_cartesian",
            "move_relative",
            "move_to_named_pose",
            *self._ROBOT_FUNCTION_EVENT_PRIMITIVES,
        }:
            return {"success": False, "message": f"unsupported step type: {primitive}"}
        robot_key = str(robot or "").strip().lower()
        source = self._robot_function_capture_source(target, cfg)
        waypoint = self._snapshot_robot_waypoint(robot_key, source=source)
        if "error" in waypoint:
            return {"success": False, "message": waypoint["error"]}
        step = {
            "step_name": step_key,
            "primitive": primitive,
            "capture_source": source,
            "captured_at": time.time(),
            "waypoint": {
                "joint_names": list(waypoint.get("joint_names") or []),
                "joint_positions": [float(v) for v in (waypoint.get("positions") or [])],
                "pose": None,
                "gripper_joint": waypoint.get("gripper_joint"),
                "gripper_position": waypoint.get("gripper"),
                "source": source,
            },
        }
        key = self._robot_function_buffer_key(
            target,
            robot_key,
            self._robot_function_safe_function_name(function_name),
            self._robot_function_safe_name(name),
        )
        with self._digital_twin_record_lock:
            steps = self._digital_twin_function_steps.setdefault(key, [])
            steps.append(step)
            count = len(steps)
        return {
            "success": True,
            "message": f"captured {step['step_name']} for {robot_key} {function_name}",
            "count": count,
        }

    def digital_twin_add_function_event(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
        step_name: str,
        primitive: str = "release_part",
    ) -> dict[str, Any]:
        cfg, err = self._robot_function_validate_request(target, robot, function_name)
        if err or cfg is None:
            return {"success": False, "message": err}
        step_key = self._robot_function_safe_name(step_name)
        primitive = str(primitive or "release_part").strip()
        if primitive not in self._ROBOT_FUNCTION_EVENT_PRIMITIVES:
            return {
                "success": False,
                "message": f"{step_key} is a waypoint step; use Capture Step.",
            }
        robot_key = str(robot or "").strip().lower()
        step = {
            "step_name": step_key,
            "primitive": primitive,
            "capture_source": self._robot_function_capture_source(target, cfg),
            "captured_at": time.time(),
        }
        key = self._robot_function_buffer_key(
            target,
            robot_key,
            self._robot_function_safe_function_name(function_name),
            self._robot_function_safe_name(name),
        )
        with self._digital_twin_record_lock:
            steps = self._digital_twin_function_steps.setdefault(key, [])
            steps.append(step)
            count = len(steps)
        return {
            "success": True,
            "message": f"added {step['step_name']} for {robot_key} {function_name}",
            "count": count,
        }

    def _robot_function_payload(
        self,
        target: str,
        cfg: dict[str, Any],
        robot: str,
        function_name: str,
        name: str,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        storage_source = self._robot_function_storage_source(target, cfg)
        replay_targets = ["gazebo"] if storage_source == "gazebo" else ["hardware", "digital_twin"]
        capture_source = self._robot_function_capture_source(target, cfg)
        for step in steps:
            waypoint = dict(step.get("waypoint") or {})
            if waypoint.get("source"):
                capture_source = str(waypoint.get("source") or capture_source)
                break
        return {
            "robot": str(robot or "").strip().lower(),
            "function_name": self._robot_function_safe_function_name(function_name),
            "name": str(name or "").strip() or "default",
            "capture_source": capture_source,
            "replay_targets": replay_targets,
            "steps": deepcopy(steps),
        }

    def digital_twin_save_function(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> dict[str, Any]:
        cfg, err = self._robot_function_validate_request(target, robot, function_name)
        if err or cfg is None:
            return {"success": False, "message": err}
        safe_name = self._robot_function_safe_name(name)
        function_key = self._robot_function_safe_function_name(function_name)
        key = self._robot_function_buffer_key(target, robot, function_key, safe_name)
        with self._digital_twin_record_lock:
            unsaved_steps = deepcopy(list(self._digital_twin_function_steps.get(key, [])))
        storage_source = self._robot_function_storage_source(target, cfg)
        path = self._robot_function_path(robot, function_key, safe_name, storage_source)
        saved_steps: list[dict[str, Any]] = []
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    saved_steps = deepcopy(list(existing.get("steps") or []))
            except Exception:
                saved_steps = []
        steps = [*saved_steps, *unsaved_steps]
        if not steps:
            return {"success": False, "message": "no function steps captured."}
        payload = self._robot_function_payload(target, cfg, robot, function_key, safe_name, steps)
        atomic_json_write(path, payload)
        with self._digital_twin_record_lock:
            self._digital_twin_function_steps.pop(key, None)
        appended_text = (
            f" ({len(saved_steps)} saved + {len(unsaved_steps)} new)"
            if saved_steps and unsaved_steps
            else ""
        )
        return {
            "success": True,
            "message": f"saved {len(steps)} steps{appended_text} to {self._robot_function_display_path(path)}",
            "file": str(path),
            "display_path": self._robot_function_display_path(path),
            "saved_steps": len(steps),
            "unsaved_steps": 0,
        }

    def digital_twin_delete_function(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> dict[str, Any]:
        cfg, err = self._robot_function_validate_request(target, robot, function_name)
        if err or cfg is None:
            return {"success": False, "message": err}
        storage_source = self._robot_function_storage_source(target, cfg)
        path = self._robot_function_path(robot, function_name, name, storage_source)
        if not path.is_file():
            return {"success": False, "message": f"taught function file not found: {path.name}"}
        try:
            path.unlink()
        except Exception as exc:
            return {"success": False, "message": str(exc)}
        return {
            "success": True,
            "message": f"deleted {self._robot_function_display_path(path)}",
        }

    @staticmethod
    def _robot_function_step_waypoint(step: dict[str, Any]) -> dict[str, Any] | None:
        return digital_twin.robot_function_step_waypoint(step)

    def _recording_from_function_payload(
        self,
        target: str,
        payload: dict[str, Any],
        *,
        step_index: int | None = None,
    ) -> dict[str, Any] | None:
        robot = str(payload.get("robot") or "").strip().lower()
        steps = list(payload.get("steps") or [])
        if step_index is not None:
            if not (0 <= int(step_index) < len(steps)):
                return None
            steps = [steps[int(step_index)]]
        waypoints = [
            body
            for body in (self._robot_function_step_waypoint(dict(step)) for step in steps)
            if body is not None
        ]
        if not waypoints:
            return None
        first = dict(waypoints[0])
        return {
            "target": target,
            "robot": robot,
            "recording_type": "single_robot",
            "saved_steps": len(steps),
            "joint_names": list(first.get("joint_names") or []),
            "gripper_joint": first.get("gripper_joint"),
            "recovery_metadata": self._digital_twin_recovery_metadata(
                target,
                recording_type="single_robot",
            ),
            "waypoints": [
                {
                    "positions": list(body.get("positions") or []),
                    **(
                        {"gripper": body.get("gripper")}
                        if body.get("gripper") is not None
                        else {}
                    ),
                }
                for body in waypoints
            ],
        }

    def _robot_function_file_payload(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
    ) -> tuple[dict[str, Any] | None, Path | None, str]:
        cfg, err = self._robot_function_validate_request(target, robot, function_name)
        if err or cfg is None:
            return None, None, err
        storage_source = self._robot_function_storage_source(target, cfg)
        path = self._robot_function_path(robot, function_name, name, storage_source)
        payload = self._read_json_file(path)
        if not payload:
            return None, path, f"taught function file not found: {self._robot_function_display_path(path)}"
        return payload, path, ""

    def _single_robot_replay_cfg(self, cfg: dict[str, Any], robot: str) -> dict[str, Any]:
        return digital_twin.single_robot_replay_cfg(cfg, robot)

    def digital_twin_replay_function(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
        *,
        replay_target: str = "twin",
        repeat_count: int = 1,
    ) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        payload, path, err = self._robot_function_file_payload(target, robot, function_name, name)
        if err or payload is None or path is None:
            return {"success": False, "message": err}
        recording = self._recording_from_function_payload(target, payload)
        if recording is None:
            return {"success": False, "message": "function has no waypoint steps to replay."}
        temp_path = Path("/tmp") / (
            f"cais_digital_twin_function_{self._robot_function_safe_name(robot)}_"
            f"{self._robot_function_safe_name(function_name)}_"
            f"{self._robot_function_safe_name(name)}.json"
        )
        atomic_json_write(temp_path, recording)
        replay_cfg = self._single_robot_replay_cfg(cfg, robot)
        return self._replay_recording_file(
            target,
            replay_cfg,
            temp_path,
            replay_target,
            source=f"function_{robot}_{function_name}_{name}",
            repeat_count=repeat_count,
        )

    def digital_twin_replay_function_step(
        self,
        target: str,
        robot: str,
        function_name: str,
        name: str,
        step_index: int,
        *,
        replay_target: str = "twin",
    ) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        payload, path, err = self._robot_function_file_payload(target, robot, function_name, name)
        if err or payload is None or path is None:
            return {"success": False, "message": err}
        recording = self._recording_from_function_payload(target, payload, step_index=step_index)
        if recording is None:
            return {"success": False, "message": "selected step has no waypoint to replay."}
        temp_path = Path("/tmp") / (
            f"cais_digital_twin_function_step_{self._robot_function_safe_name(robot)}_"
            f"{self._robot_function_safe_name(function_name)}_"
            f"{self._robot_function_safe_name(name)}_{int(step_index)}.json"
        )
        atomic_json_write(temp_path, recording)
        replay_cfg = self._single_robot_replay_cfg(cfg, robot)
        return self._replay_recording_file(
            target,
            replay_cfg,
            temp_path,
            replay_target,
            source=f"function_step_{robot}_{function_name}_{name}_{int(step_index)}",
        )

    def _paired_recording_from_function_payloads(
        self,
        target: str,
        xarm6_payload: dict[str, Any],
        ur5e_payload: dict[str, Any],
        *,
        xarm6_step_index: int | None = None,
        ur5e_step_index: int | None = None,
    ) -> dict[str, Any] | None:
        def _waypoints(payload: dict[str, Any], step_index: int | None) -> list[dict[str, Any]]:
            steps = list(payload.get("steps") or [])
            if step_index is not None:
                if not (0 <= int(step_index) < len(steps)):
                    return []
                steps = [steps[int(step_index)]]
            return [
                body
                for body in (self._robot_function_step_waypoint(dict(step)) for step in steps)
                if body is not None
            ]

        xarm6_waypoints = _waypoints(xarm6_payload, xarm6_step_index)
        ur5e_waypoints = _waypoints(ur5e_payload, ur5e_step_index)
        if not xarm6_waypoints or not ur5e_waypoints:
            return None
        count = max(len(xarm6_waypoints), len(ur5e_waypoints))
        first_xarm6 = dict(xarm6_waypoints[0])
        first_ur5e = dict(ur5e_waypoints[0])
        return {
            "target": target,
            "robot": "dual robots",
            "recording_type": "paired_dual_robots",
            "robots": {
                "xarm6": {
                    "joint_names": list(first_xarm6.get("joint_names") or []),
                    "gripper_joint": first_xarm6.get("gripper_joint"),
                },
                "ur5e": {
                    "joint_names": list(first_ur5e.get("joint_names") or []),
                    "gripper_joint": first_ur5e.get("gripper_joint"),
                },
            },
            "recovery_metadata": self._digital_twin_recovery_metadata(
                target,
                recording_type="paired_dual_robots",
            ),
            "waypoints": [
                {
                    "robots": {
                        "xarm6": {
                            "positions": list(xarm6_waypoints[min(i, len(xarm6_waypoints) - 1)].get("positions") or []),
                            **(
                                {"gripper": xarm6_waypoints[min(i, len(xarm6_waypoints) - 1)].get("gripper")}
                                if xarm6_waypoints[min(i, len(xarm6_waypoints) - 1)].get("gripper") is not None
                                else {}
                            ),
                        },
                        "ur5e": {
                            "positions": list(ur5e_waypoints[min(i, len(ur5e_waypoints) - 1)].get("positions") or []),
                            **(
                                {"gripper": ur5e_waypoints[min(i, len(ur5e_waypoints) - 1)].get("gripper")}
                                if ur5e_waypoints[min(i, len(ur5e_waypoints) - 1)].get("gripper") is not None
                                else {}
                            ),
                        },
                    }
                }
                for i in range(count)
            ],
        }

    def _load_dual_function_payloads(
        self,
        target: str,
        xarm6_function_name: str,
        xarm6_name: str,
        ur5e_function_name: str,
        ur5e_name: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
        xarm6_payload, _xarm6_path, xarm6_err = self._robot_function_file_payload(
            target,
            "xarm6",
            xarm6_function_name,
            xarm6_name,
        )
        if xarm6_err or xarm6_payload is None:
            return None, None, f"xarm6: {xarm6_err}"
        ur5e_payload, _ur5e_path, ur5e_err = self._robot_function_file_payload(
            target,
            "ur5e",
            ur5e_function_name,
            ur5e_name,
        )
        if ur5e_err or ur5e_payload is None:
            return None, None, f"ur5e: {ur5e_err}"
        return xarm6_payload, ur5e_payload, ""

    def digital_twin_replay_dual_function(
        self,
        target: str,
        xarm6_function_name: str,
        xarm6_name: str,
        ur5e_function_name: str,
        ur5e_name: str,
        *,
        replay_target: str = "twin",
        repeat_count: int = 1,
    ) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        if not self._digital_twin_is_dual_robots(cfg):
            return {"success": False, "message": "Replay Dual Taught Function requires dual robots."}
        xarm6_payload, ur5e_payload, err = self._load_dual_function_payloads(
            target,
            xarm6_function_name,
            xarm6_name,
            ur5e_function_name,
            ur5e_name,
        )
        if err or xarm6_payload is None or ur5e_payload is None:
            return {"success": False, "message": err}
        recording = self._paired_recording_from_function_payloads(
            target,
            xarm6_payload,
            ur5e_payload,
        )
        if recording is None:
            return {"success": False, "message": "both taught function files need at least one waypoint step."}
        temp_path = Path("/tmp") / (
            f"cais_digital_twin_dual_function_"
            f"{self._robot_function_safe_name(xarm6_function_name)}_{self._robot_function_safe_name(xarm6_name)}__"
            f"{self._robot_function_safe_name(ur5e_function_name)}_{self._robot_function_safe_name(ur5e_name)}.json"
        )
        atomic_json_write(temp_path, recording)
        return self._replay_recording_file(
            target,
            cfg,
            temp_path,
            replay_target,
            source=f"dual_function_{xarm6_function_name}_{xarm6_name}_{ur5e_function_name}_{ur5e_name}",
            repeat_count=repeat_count,
        )

    def digital_twin_replay_dual_step(
        self,
        target: str,
        xarm6_function_name: str,
        xarm6_name: str,
        xarm6_step_index: int,
        ur5e_function_name: str,
        ur5e_name: str,
        ur5e_step_index: int,
        *,
        replay_target: str = "twin",
    ) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        if not self._digital_twin_is_dual_robots(cfg):
            return {"success": False, "message": "Replay Dual Selected Step requires dual robots."}
        xarm6_payload, ur5e_payload, err = self._load_dual_function_payloads(
            target,
            xarm6_function_name,
            xarm6_name,
            ur5e_function_name,
            ur5e_name,
        )
        if err or xarm6_payload is None or ur5e_payload is None:
            return {"success": False, "message": err}
        recording = self._paired_recording_from_function_payloads(
            target,
            xarm6_payload,
            ur5e_payload,
            xarm6_step_index=xarm6_step_index,
            ur5e_step_index=ur5e_step_index,
        )
        if recording is None:
            return {"success": False, "message": "selected dual steps both need waypoints."}
        temp_path = Path("/tmp") / (
            f"cais_digital_twin_dual_function_step_"
            f"{self._robot_function_safe_name(xarm6_function_name)}_{self._robot_function_safe_name(xarm6_name)}_{int(xarm6_step_index)}__"
            f"{self._robot_function_safe_name(ur5e_function_name)}_{self._robot_function_safe_name(ur5e_name)}_{int(ur5e_step_index)}.json"
        )
        atomic_json_write(temp_path, recording)
        return self._replay_recording_file(
            target,
            cfg,
            temp_path,
            replay_target,
            source=(
                f"dual_function_step_{xarm6_function_name}_{xarm6_name}_{int(xarm6_step_index)}_"
                f"{ur5e_function_name}_{ur5e_name}_{int(ur5e_step_index)}"
            ),
        )

    def _digital_twin_buffer_path(self, target: str, cfg: dict[str, Any]) -> Path:
        slug = self._digital_twin_slug(cfg)
        return Path("/tmp") / f"cais_digital_twin_{slug}_buffer.json"

    @staticmethod
    def _digital_twin_recording_hash(recording: dict[str, Any]) -> str:
        return digital_twin.digital_twin_recording_hash(recording)

    def _digital_twin_prepared_replay_path(
        self,
        target: str,
        cfg: dict[str, Any],
        recording: dict[str, Any],
        replay_target: str,
        source: str,
    ) -> Path:
        return digital_twin.digital_twin_prepared_replay_path(
            cfg,
            recording,
            replay_target,
            source,
            Path("/tmp"),
        )

    @staticmethod
    def _digital_twin_is_dual_robots(cfg: dict[str, Any]) -> bool:
        return digital_twin.digital_twin_is_dual_robots(cfg)

    @staticmethod
    def _digital_twin_dual_robot_keys(cfg: dict[str, Any]) -> list[str]:
        return digital_twin.digital_twin_dual_robot_keys(cfg)

    @staticmethod
    def _digital_twin_recovery_metadata(target: str, *, recording_type: str) -> dict[str, Any]:
        return digital_twin.digital_twin_recovery_metadata(
            target,
            recording_type=recording_type,
        )

    def _waypoints_from_recording(
        self,
        cfg: dict[str, Any],
        recording: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return digital_twin.waypoints_from_recording(cfg, recording)

    def _load_digital_twin_buffer_waypoints(
        self,
        target: str,
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        path = self._digital_twin_buffer_path(target, cfg)
        data = self._read_json_file(path)
        if not data:
            return []
        return self._waypoints_from_recording(cfg, data)

    def _digital_twin_buffer_waypoints(
        self,
        target: str,
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        with self._digital_twin_record_lock:
            buf = list(self._digital_twin_waypoints.get(target, []))
        if buf:
            return buf

        persisted = self._load_digital_twin_buffer_waypoints(target, cfg)
        if persisted:
            with self._digital_twin_record_lock:
                if not self._digital_twin_waypoints.get(target):
                    self._digital_twin_waypoints[target] = list(persisted)
                buf = list(self._digital_twin_waypoints.get(target, []))
            return buf
        return []

    def _persist_digital_twin_buffer(
        self,
        target: str,
        cfg: dict[str, Any],
        waypoints: list[dict[str, Any]],
    ) -> None:
        path = self._digital_twin_buffer_path(target, cfg)
        if waypoints:
            atomic_json_write(path, self._recording_from_waypoints(target, cfg, waypoints))
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("failed to clear digital twin buffer %s", path)

    def _snapshot_robot_waypoint(self, robot: str, *, source: str = "gazebo") -> dict[str, Any]:
        """Snapshot one robot's current gazebo or hardware pose into a waypoint body."""
        source_key = str(source or "gazebo").strip().lower()
        if source_key not in {"gazebo", "hardware"}:
            source_key = "gazebo"
        domains = self._digital_twin_domain_ids()
        result = self._run_digital_twin_sync(
            [
                "--mode", "snapshot",
                "--robot", robot,
                "--source", source_key,
                "--gazebo-domain-id", str(domains["gazebo"]),
                "--hardware-domain-id", str(domains["hardware"]),
            ],
            timeout_sec=20.0,
        )
        if not result.get("success"):
            return {"error": str(result.get("message") or "snapshot failed")}
        return {
            "positions": [float(v) for v in (result.get("positions") or [])],
            "gripper": result.get("gripper_position"),
            "joint_names": list(result.get("joint_names") or []),
            "gripper_joint": result.get("gripper_joint"),
            "source": source_key,
        }

    def _snapshot_waypoint(self, target: str, cfg: dict[str, Any]) -> dict[str, Any]:
        """Snapshot the current gazebo pose into a waypoint dict (or {'error': msg})."""
        if self.ros2_proc_status(str(cfg.get("gazebo_process") or "")) != "running":
            return {"error": f"{target} gazebo is not running."}

        if self._digital_twin_is_dual_robots(cfg):
            robots: dict[str, dict[str, Any]] = {}
            for robot in self._digital_twin_dual_robot_keys(cfg):
                waypoint = self._snapshot_robot_waypoint(robot)
                if "error" in waypoint:
                    return {"error": f"{robot}: {waypoint['error']}"}
                robots[robot] = waypoint
            if not robots:
                return {"error": f"{target} has no paired robots configured."}
            return {
                "robots": robots,
                "t": time.time(),
            }

        robot_waypoint = self._snapshot_robot_waypoint(str(cfg.get("robot") or ""))
        if "error" in robot_waypoint:
            return robot_waypoint
        robot_waypoint["t"] = time.time()
        return robot_waypoint

    def digital_twin_capture_waypoint(self, target: str) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        waypoint = self._snapshot_waypoint(target, cfg)
        if "error" in waypoint:
            return {"success": False, "message": waypoint["error"]}
        with self._digital_twin_record_lock:
            buf = self._digital_twin_waypoints.setdefault(target, [])
            buf.append(waypoint)
            persisted_buf = list(buf)
            count = len(buf)
        self._persist_digital_twin_buffer(target, cfg, persisted_buf)
        return {"success": True, "message": f"captured waypoint {count}", "count": count}

    def digital_twin_list_waypoints(self, target: str) -> list[dict[str, Any]]:
        cfg = self._digital_twin_target(target) or {}
        buf = self._digital_twin_buffer_waypoints(target, cfg) if cfg else []
        return [
            {
                "index": i,
                "positions": [float(v) for v in (wp.get("positions") or [])],
                "gripper": wp.get("gripper"),
                "robots": {
                    str(robot): {
                        "positions": [float(v) for v in (body.get("positions") or [])],
                        "gripper": body.get("gripper"),
                    }
                    for robot, body in dict(wp.get("robots") or {}).items()
                    if isinstance(body, dict)
                },
            }
            for i, wp in enumerate(buf)
        ]

    def digital_twin_delete_waypoint(self, target: str, index: int) -> dict[str, Any]:
        cfg = self._digital_twin_target(target) or {}
        if cfg:
            self._digital_twin_buffer_waypoints(target, cfg)
        with self._digital_twin_record_lock:
            buf = self._digital_twin_waypoints.get(target, [])
            if not (0 <= index < len(buf)):
                return {"success": False, "message": "waypoint index out of range."}
            buf.pop(index)
            persisted_buf = list(buf)
            count = len(buf)
        if cfg:
            self._persist_digital_twin_buffer(target, cfg, persisted_buf)
        return {"success": True, "message": f"deleted waypoint {index + 1}", "count": count}

    def digital_twin_move_waypoint(self, target: str, index: int, delta: int) -> dict[str, Any]:
        cfg = self._digital_twin_target(target) or {}
        if cfg:
            self._digital_twin_buffer_waypoints(target, cfg)
        with self._digital_twin_record_lock:
            buf = self._digital_twin_waypoints.get(target, [])
            new_index = index + (1 if delta > 0 else -1)
            if not (0 <= index < len(buf)) or not (0 <= new_index < len(buf)):
                return {"success": False, "message": "cannot move waypoint."}
            buf[index], buf[new_index] = buf[new_index], buf[index]
            persisted_buf = list(buf)
        if cfg:
            self._persist_digital_twin_buffer(target, cfg, persisted_buf)
        return {"success": True, "message": "reordered waypoints."}

    def digital_twin_overwrite_waypoint(self, target: str, index: int) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        self._digital_twin_buffer_waypoints(target, cfg)
        with self._digital_twin_record_lock:
            if not (0 <= index < len(self._digital_twin_waypoints.get(target, []))):
                return {"success": False, "message": "waypoint index out of range."}
        waypoint = self._snapshot_waypoint(target, cfg)
        if "error" in waypoint:
            return {"success": False, "message": waypoint["error"]}
        with self._digital_twin_record_lock:
            buf = self._digital_twin_waypoints.get(target, [])
            if not (0 <= index < len(buf)):
                return {"success": False, "message": "waypoint index out of range."}
            buf[index] = waypoint
            persisted_buf = list(buf)
        self._persist_digital_twin_buffer(target, cfg, persisted_buf)
        return {"success": True, "message": f"updated waypoint {index + 1} to current sim pose"}

    def digital_twin_waypoint_count(self, target: str) -> int:
        cfg = self._digital_twin_target(target) or {}
        return len(self._digital_twin_buffer_waypoints(target, cfg)) if cfg else 0

    def digital_twin_clear_waypoints(self, target: str) -> None:
        with self._digital_twin_record_lock:
            self._digital_twin_waypoints.pop(target, None)
        cfg = self._digital_twin_target(target) or {}
        if cfg:
            self._persist_digital_twin_buffer(target, cfg, [])

    def digital_twin_save_recording(self, target: str, name: str) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name).strip())
        if not safe:
            return {"success": False, "message": "recording name is empty."}
        buf = self._digital_twin_buffer_waypoints(target, cfg)
        if not buf:
            return {"success": False, "message": "no waypoints captured."}

        recording = self._recording_from_waypoints(target, cfg, buf)
        slug = self._digital_twin_slug(cfg)
        path = self._digital_twin_recordings_dir() / f"{slug}__{safe}.json"
        atomic_json_write(path, recording)
        return {"success": True, "message": f"saved {len(buf)} waypoints to {path.name}", "file": str(path)}

    def digital_twin_list_recordings(self, target: str) -> list[str]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return []
        slug = self._digital_twin_slug(cfg)
        prefix = f"{slug}__"
        return sorted(
            p.name[len(prefix):-len(".json")]
            for p in self._digital_twin_recordings_dir().glob(f"{prefix}*.json")
        )

    def digital_twin_delete_recording(self, target: str, name: str) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        safe = str(name or "").strip()
        if not safe:
            return {"success": False, "message": "no recording selected."}
        slug = self._digital_twin_slug(cfg)
        path = self._digital_twin_recordings_dir() / f"{slug}__{safe}.json"
        if not path.is_file():
            return {"success": False, "message": f"recording not found: {safe}"}
        try:
            path.unlink()
        except Exception as exc:
            return {"success": False, "message": str(exc)}
        return {"success": True, "message": f"deleted recording '{safe}'."}

    def _build_buffer_recording(self, target: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
        """Assemble a recording dict from the in-memory capture buffer (or None if empty)."""
        buf = self._digital_twin_buffer_waypoints(target, cfg)
        if not buf:
            return None
        return self._recording_from_waypoints(target, cfg, buf)

    def _recording_from_waypoints(
        self,
        target: str,
        cfg: dict[str, Any],
        waypoints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._digital_twin_is_dual_robots(cfg):
            robot_meta: dict[str, dict[str, Any]] = {}
            first_robots = dict((waypoints[0] or {}).get("robots") or {})
            for robot in self._digital_twin_dual_robot_keys(cfg):
                body = dict(first_robots.get(robot) or {})
                robot_meta[robot] = {
                    "joint_names": list(body.get("joint_names") or []),
                    "gripper_joint": body.get("gripper_joint"),
                }
            return {
                "target": target,
                "robot": "dual robots",
                "recording_type": "paired_dual_robots",
                "robots": robot_meta,
                "recovery_metadata": self._digital_twin_recovery_metadata(
                    target,
                    recording_type="paired_dual_robots",
                ),
                "waypoints": [
                    {
                        "robots": {
                            robot: {
                                "positions": list(dict((wp.get("robots") or {}).get(robot) or {}).get("positions") or []),
                                "gripper": dict((wp.get("robots") or {}).get(robot) or {}).get("gripper"),
                            }
                            for robot in self._digital_twin_dual_robot_keys(cfg)
                        }
                    }
                    for wp in waypoints
                ],
            }

        return {
            "target": target,
            "robot": str(cfg.get("robot") or ""),
            "recording_type": "single_robot",
            "joint_names": list(waypoints[0].get("joint_names") or []),
            "gripper_joint": waypoints[0].get("gripper_joint"),
            "recovery_metadata": self._digital_twin_recovery_metadata(
                target,
                recording_type="single_robot",
            ),
            "waypoints": [
                {"positions": list(wp.get("positions") or []), "gripper": wp.get("gripper")}
                for wp in waypoints
            ],
        }

    def _prepare_replay_recording_file(
        self,
        target: str,
        cfg: dict[str, Any],
        recording_path: Path,
        replay_target: str,
        *,
        source: str,
    ) -> dict[str, Any]:
        if replay_target not in ("twin", "gazebo", "hardware"):
            return {"success": False, "message": f"unknown replay target: {replay_target}"}
        if not self._digital_twin_is_dual_robots(cfg):
            return {"success": False, "message": "prepare replay only supports dual robots."}
        recording = self._read_json_file(recording_path)
        if not recording:
            return {"success": False, "message": f"recording not found: {recording_path.name}"}
        sync_target = "both" if replay_target == "twin" else replay_target
        needs_hardware = replay_target in ("twin", "hardware")
        domains = self._digital_twin_domain_ids()
        if needs_hardware:
            err = self._wait_for_digital_twin_dual_robots_hardware_ready(
                cfg,
                ros_domain_id=int(domains["hardware"]),
                require_moveit=False,
                require_ur5e_trajectory_controller=False,
            )
            if err:
                message = f"{target} hardware is not ready: {err}"
                self._write_digital_twin_status(
                    target,
                    {
                        "state": "blocked",
                        "direction": self._digital_twin_direction(target),
                        "message": message,
                        "last_error": message,
                    },
                )
                return {"success": False, "message": message}
        gazebo_process = str(cfg.get("gazebo_process") or "")
        if replay_target in ("twin", "gazebo") and self.ros2_proc_status(gazebo_process) != "running":
            return {"success": False, "message": f"{target} gazebo is not running."}

        prepared_path = self._digital_twin_prepared_replay_path(
            target,
            cfg,
            recording,
            replay_target,
            source,
        )
        with self._digital_twin_prepare_lock:
            self._write_digital_twin_status(
                target,
                {
                    "state": "preparing",
                    "direction": self._digital_twin_direction(target),
                    "message": "preparing replay without moving gazebo or hardware.",
                    "last_error": "",
                },
            )
            result = self._run_digital_twin_sync(
                [
                    "--mode", "prepare-replay",
                    "--robot", str(cfg.get("robot") or ""),
                    "--replay-target", sync_target,
                    "--recording-file", str(recording_path),
                    "--prepared-file", str(prepared_path),
                    "--gazebo-domain-id", str(domains["gazebo"]),
                    "--hardware-domain-id", str(domains["hardware"]),
                    "--max-joint-delta-deg", str(self._DIGITAL_TWIN_MAX_JOINT_DELTA_DEG),
                    "--status-file", str(self._digital_twin_status_path(target)),
                    "--direction-file", str(self._digital_twin_direction_path(target)),
                ],
                timeout_sec=120.0,
            )
        result["prepared_file"] = str(prepared_path)
        result["recording_hash"] = self._digital_twin_recording_hash(recording)
        return result

    def _replay_recording_file(
        self,
        target: str,
        cfg: dict[str, Any],
        recording_path: Path,
        replay_target: str,
        *,
        source: str = "recording",
        repeat_count: int = 1,
    ) -> dict[str, Any]:
        if replay_target not in ("twin", "gazebo", "hardware"):
            return {"success": False, "message": f"unknown replay target: {replay_target}"}
        try:
            repeat_total = int(repeat_count)
        except Exception:
            repeat_total = 1
        repeat_total = max(1, min(999, repeat_total))
        replay_label = "Replay Dual Function" if str(source).startswith("dual_function") else "Replay Function"
        # "twin" drives gazebo + hardware together; "gazebo" is a sim-only dry run.
        sync_target = "both" if replay_target == "twin" else replay_target
        needs_hardware = replay_target in ("twin", "hardware")
        domains = self._digital_twin_domain_ids()
        dual_hardware_replay = needs_hardware and self._digital_twin_is_dual_robots(cfg)

        def _check_dual_replay_ready(message: str) -> str | None:
            self._write_digital_twin_status(
                target,
                {
                    "state": "checking",
                    "direction": "hardware -> gazebo" if replay_target == "twin" else self._digital_twin_direction(target),
                    "message": message,
                    "last_error": "",
                },
            )
            return self._wait_for_digital_twin_dual_robots_replay_ready(
                target,
                cfg,
                ros_domain_id=int(domains["hardware"]),
            )

        def _blocked_dual_replay_result(message: str, **extra: Any) -> dict[str, Any]:
            self._write_digital_twin_status(
                target,
                {
                    "state": "blocked",
                    "direction": "hardware -> gazebo" if replay_target == "twin" else self._digital_twin_direction(target),
                    "message": message,
                    "last_error": message,
                },
            )
            out: dict[str, Any] = {"success": False, "message": message}
            out.update(extra)
            return out

        if dual_hardware_replay:
            self._write_digital_twin_status(
                target,
                {
                    "state": "checking",
                    "direction": "hardware -> gazebo" if replay_target == "twin" else self._digital_twin_direction(target),
                    "message": "checking replay readiness.",
                    "last_error": "",
                },
            )
            err = self._wait_for_digital_twin_dual_robots_replay_ready(
                target,
                cfg,
                ros_domain_id=int(domains["hardware"]),
            )
            if err:
                message = f"{target} hardware is not ready: {err}"
                self._write_digital_twin_status(
                    target,
                    {
                        "state": "blocked",
                        "direction": "hardware -> gazebo" if replay_target == "twin" else self._digital_twin_direction(target),
                        "message": message,
                        "last_error": message,
                    },
                )
                return {"success": False, "message": message}
        elif needs_hardware and self._digital_twin_hardware_status(cfg).get("overall") != "running":
            return {"success": False, "message": f"{target} hardware is not running."}
        if self.ros2_proc_status(str(cfg.get("gazebo_process") or "")) != "running":
            return {"success": False, "message": f"{target} gazebo is not running."}

        prepared_args: list[str] = []
        if replay_target == "twin" and self._digital_twin_is_dual_robots(cfg):
            recording = self._read_json_file(recording_path)
            if recording:
                prepared_path = self._digital_twin_prepared_replay_path(
                    target,
                    cfg,
                    recording,
                    replay_target,
                    source,
                )
                if prepared_path.is_file():
                    prepared_data = self._read_json_file(prepared_path)
                    metadata = dict(prepared_data.get("metadata") or {})
                    expected = {
                        "version": self._DIGITAL_TWIN_PREPARED_REPLAY_VERSION,
                        "recording_hash": self._digital_twin_recording_hash(recording),
                        "replay_target": sync_target,
                        "gazebo_domain_id": int(domains["gazebo"]),
                        "hardware_domain_id": int(domains["hardware"]),
                        "waypoint_count": len(list(recording.get("waypoints") or [])),
                        "robot": str(recording.get("robot") or ""),
                        "recording_type": str(recording.get("recording_type") or ""),
                    }
                    if all(metadata.get(key) == value for key, value in expected.items()):
                        prepared_args = ["--prepared-file", str(prepared_path)]
        gazebo_initialization = "not_required"
        init_before_replay = replay_target == "twin" or (
            replay_target == "gazebo"
            and self._digital_twin_is_dual_robots(cfg)
            and self._digital_twin_sim_mode(target) == "teach"
        )
        if init_before_replay:
            sync_snapshot = self._digital_twin_sync_status_snapshot(target, cfg, time.time())
            status_data = dict(sync_snapshot.get("status_data") or {})
            status_age_ms = sync_snapshot.get("status_age_ms")
            sync_is_healthy = (
                replay_target == "twin"
                and str(sync_snapshot.get("process_status") or "") == "running"
                and str(status_data.get("state") or "") == "mirroring"
                and status_age_ms is not None
                and float(status_age_ms) <= 3000.0
                and self._digital_twin_direction(target) == "hardware -> gazebo"
            )
            if sync_is_healthy:
                init_before_replay = False
                gazebo_initialization = "skipped"
        if init_before_replay:
            init_label = "Replay in Twin" if replay_target == "twin" else "Preview in Gazebo"
            init_failure_word = "replay" if replay_target == "twin" else "preview"
            self._write_digital_twin_status(
                target,
                {
                    "state": "replaying",
                    "direction": "hardware -> gazebo",
                    "message": f"{init_label}: initializing gazebo from hardware.",
                    "last_error": "",
                },
            )
            init_err = self._initialize_digital_twin_gazebo_from_hardware(
                target,
                cfg,
                gazebo_process=str(cfg.get("gazebo_process") or ""),
                domains=domains,
            )
            if init_err:
                return {
                    "success": False,
                    "message": f"could not initialize gazebo from hardware before {init_failure_word}: {init_err}",
                    "gazebo_initialization": "failed",
                }
            gazebo_initialization = "ran"
        if dual_hardware_replay:
            err = _check_dual_replay_ready("checking final replay readiness.")
            if err:
                message = f"{target} hardware is not ready: {err}"
                return _blocked_dual_replay_result(
                    message,
                    gazebo_initialization=gazebo_initialization,
                )
        self._write_digital_twin_status(
            target,
            {
                "state": "replaying",
                "direction": "hardware -> gazebo" if replay_target == "twin" else self._digital_twin_direction(target),
                "message": (
                    "Replay in Twin: committing prepared sim waypoint through /xarm6/xarm6_traj_controller/follow_joint_trajectory and /execute_trajectory."
                    if (
                        prepared_args
                        and replay_target == "twin"
                        and self._digital_twin_is_dual_robots(cfg)
                    )
                    else (
                        "Replay in Twin: committing saved sim waypoint through /xarm6/xarm6_traj_controller/follow_joint_trajectory and /execute_trajectory."
                        if replay_target == "twin" and self._digital_twin_is_dual_robots(cfg)
                        else (
                            "Replay in Twin: committing prepared sim waypoint through hardware MoveIt."
                            if prepared_args and replay_target == "twin"
                            else (
                                "Replay in Twin: committing saved sim waypoint through hardware MoveIt."
                                if replay_target == "twin"
                                else "Preview in Gazebo: publishing to Gazebo only."
                            )
                        )
                    )
                ),
                "last_error": "",
            },
        )
        stopped_mirror_workers = False
        if replay_target == "twin" and self._digital_twin_is_dual_robots(cfg):
            self._stop_digital_twin_mirror_workers_for_target(target, cfg)
            stopped_mirror_workers = True
            self._write_digital_twin_status(
                target,
                {
                    "state": "replaying",
                    "direction": "hardware -> gazebo",
                    "message": "executing dual replay.",
                    "last_error": "",
                },
            )
        sync_args = [
            "--mode", "replay",
            "--robot", str(cfg.get("robot") or ""),
            "--replay-target", sync_target,
            "--recording-file", str(recording_path),
            *prepared_args,
            "--gazebo-domain-id", str(domains["gazebo"]),
            "--hardware-domain-id", str(domains["hardware"]),
            "--max-joint-delta-deg", str(self._DIGITAL_TWIN_MAX_JOINT_DELTA_DEG),
            "--waypoint-duration-sec", f"{self._DIGITAL_TWIN_REPLAY_WAYPOINT_DURATION_SEC:.6f}",
            "--max-joint-vel-deg-s", f"{self._DIGITAL_TWIN_REPLAY_MAX_JOINT_VEL_DEG_S:.6f}",
            "--status-file", str(self._digital_twin_status_path(target)),
            "--direction-file", str(self._digital_twin_direction_path(target)),
        ]
        result: dict[str, Any] = {}
        for repeat_index in range(repeat_total):
            if repeat_total > 1:
                self._write_digital_twin_status(
                    target,
                    {
                        "state": "replaying",
                        "direction": "hardware -> gazebo" if replay_target == "twin" else self._digital_twin_direction(target),
                        "message": f"executing replay repeat {repeat_index + 1}/{repeat_total}.",
                        "last_error": "",
                    },
                )
            if dual_hardware_replay and repeat_index > 0:
                err = _check_dual_replay_ready(
                    f"checking replay readiness for repeat {repeat_index + 1}/{repeat_total}."
                )
                if err:
                    message = f"{target} hardware is not ready: {err}"
                    result = {
                        "success": False,
                        "message": (
                            f"{replay_label} failed on repeat {repeat_index + 1}/{repeat_total}: {message}"
                            if repeat_total > 1
                            else message
                        ),
                        "repeat_count": repeat_total,
                        "repeat_iteration": repeat_index + 1,
                    }
                    break
            result = self._run_digital_twin_sync(sync_args, timeout_sec=120.0)
            result["repeat_count"] = repeat_total
            result["repeat_iteration"] = repeat_index + 1
            if not result.get("success"):
                if repeat_total > 1:
                    result["message"] = (
                        f"{replay_label} failed on repeat {repeat_index + 1}/{repeat_total}: "
                        f"{str(result.get('message') or '')}"
                    )
                break
        if repeat_total > 1 and result.get("success"):
            result["message"] = (
                f"{replay_label} repeated {repeat_total} times. "
                f"{str(result.get('message') or '')}"
            )
        result["gazebo_initialization"] = gazebo_initialization
        if replay_target == "twin" and (result.get("success") or stopped_mirror_workers):
            self._write_digital_twin_status(
                target,
                {
                    "state": "resuming",
                    "direction": "hardware -> gazebo",
                    "message": "resuming hardware -> gazebo mirror.",
                    "last_error": "" if result.get("success") else str(result.get("message") or ""),
                },
            )
            sync_err = self._start_digital_twin_sync_when_ready(
                target,
                cfg,
                gazebo_process=str(cfg.get("gazebo_process") or ""),
                domains=domains,
                direction="hardware -> gazebo",
            )
            result["sync_resumed"] = not bool(sync_err)
            base_message = str(result.get("message") or "")
            if sync_err:
                result["sync_resume_error"] = sync_err
                result["message"] = (
                    f"{base_message}; hardware -> Gazebo sync resume failed: {sync_err}"
                    if base_message
                    else f"hardware -> Gazebo sync resume failed: {sync_err}"
                )
            elif result.get("success"):
                settle_sec = max(0.0, float(self._DIGITAL_TWIN_MIRROR_STABILIZATION_SEC))
                if settle_sec > 0.0:
                    self._write_digital_twin_status(
                        target,
                        {
                            "state": "resuming",
                            "direction": "hardware -> gazebo",
                            "message": (
                                "waiting for hardware -> gazebo mirror to stabilize; "
                                f"mirror_stabilization_sec={settle_sec:.2f}."
                            ),
                            "last_error": "",
                        },
                    )
                    time.sleep(settle_sec)
                result["message"] = (
                    f"{base_message}; hardware -> Gazebo sync resumed."
                    if base_message
                    else "hardware -> Gazebo sync resumed."
                )
            else:
                result["message"] = (
                    f"{base_message}; hardware -> Gazebo sync resumed after failed replay."
                    if base_message
                    else "hardware -> Gazebo sync resumed after failed replay."
                )
        detail_parts: list[str] = []
        if result.get("saved_steps") is not None:
            detail_parts.append(f"saved_steps={int(result.get('saved_steps') or 0)}")
        if result.get("waypoints") is not None:
            detail_parts.append(f"waypoints={int(result.get('waypoints') or 0)}")
        if result.get("approach_time") is not None:
            detail_parts.append(f"approach_time={float(result.get('approach_time') or 0.0):.3f}s")
        if result.get("max_joint_delta_deg") is not None:
            detail_parts.append(f"max_joint_delta_deg={float(result.get('max_joint_delta_deg') or 0.0):.3f}")
        if result.get("max_joint_delta_joint"):
            detail_parts.append(f"max_joint_delta_joint={str(result.get('max_joint_delta_joint') or '')}")
        if gazebo_initialization != "not_required":
            detail_parts.append(f"gazebo_initialization={gazebo_initialization}")
        if result.get("used_prepared_file") is not None:
            detail_parts.append(f"used_prepared_file={bool(result.get('used_prepared_file'))}")
        if detail_parts:
            base_message = str(result.get("message") or "").rstrip(". ")
            result["message"] = (
                f"{base_message}; {'; '.join(detail_parts)}."
                if base_message
                else f"{'; '.join(detail_parts)}."
            )
        message = str(result.get("message") or "")
        self._write_digital_twin_status(
            target,
            {
                "state": "replayed" if result.get("success") else "blocked",
                "direction": "hardware -> gazebo" if replay_target == "twin" else self._digital_twin_direction(target),
                "message": message or "replay finished.",
                "last_error": "" if result.get("success") else message,
            },
        )
        return result

    def digital_twin_replay(self, target: str, name: str, *, replay_target: str = "twin") -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        slug = self._digital_twin_slug(cfg)
        path = self._digital_twin_recordings_dir() / f"{slug}__{name}.json"
        if not path.is_file():
            return {"success": False, "message": f"recording not found: {name}"}
        return self._replay_recording_file(target, cfg, path, replay_target, source=f"saved_{name}")

    def digital_twin_replay_buffer(self, target: str, *, replay_target: str = "twin") -> dict[str, Any]:
        """Replay the just-captured (unsaved) waypoint buffer, without forcing a Save."""
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        recording = self._build_buffer_recording(target, cfg)
        if recording is None:
            return {"success": False, "message": "No captured waypoints to replay."}
        slug = self._digital_twin_slug(cfg)
        path = Path("/tmp") / f"cais_digital_twin_{slug}_buffer.json"
        atomic_json_write(path, recording)
        return self._replay_recording_file(target, cfg, path, replay_target, source="buffer")

    def digital_twin_prepare_replay(
        self,
        target: str,
        name: str,
        *,
        replay_target: str = "twin",
    ) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        slug = self._digital_twin_slug(cfg)
        path = self._digital_twin_recordings_dir() / f"{slug}__{name}.json"
        if not path.is_file():
            return {"success": False, "message": f"recording not found: {name}"}
        return self._prepare_replay_recording_file(
            target,
            cfg,
            path,
            replay_target,
            source=f"saved_{name}",
        )

    def digital_twin_prepare_replay_buffer(
        self,
        target: str,
        *,
        replay_target: str = "twin",
    ) -> dict[str, Any]:
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        recording = self._build_buffer_recording(target, cfg)
        if recording is None:
            return {"success": False, "message": "No captured waypoints to prepare."}
        slug = self._digital_twin_slug(cfg)
        path = Path("/tmp") / f"cais_digital_twin_{slug}_buffer.json"
        atomic_json_write(path, recording)
        return self._prepare_replay_recording_file(
            target,
            cfg,
            path,
            replay_target,
            source="buffer",
        )

    def digital_twin_go_home(self, target: str, *, replay_target: str = "twin") -> dict[str, Any]:
        """Move the arm to its configured home/initial pose (sim + hardware by default)."""
        cfg = self._digital_twin_target(target)
        if not cfg:
            return {"success": False, "message": f"unknown digital twin target: {target}"}
        robot = str(cfg.get("robot") or "")
        home = self._DIGITAL_TWIN_HOME.get(robot)
        if not home:
            return {"success": False, "message": f"no home pose configured for {robot}."}
        # Omit joint_names so the sync defaults to ROBOTS[robot]["gazebo_joints"].
        recording = {"target": target, "robot": robot, "waypoints": [{"positions": list(home)}]}
        slug = self._digital_twin_slug(cfg)
        path = Path("/tmp") / f"cais_digital_twin_{slug}_home.json"
        atomic_json_write(path, recording)
        return self._replay_recording_file(target, cfg, path, replay_target, source="home")

    def ros2_start_hardware_stack(self, robot: str) -> str | None:
        key = str(robot).strip().lower()
        stack = self._hardware_stack_for_robot(key)
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

        driver_name = stack[0]
        gripper_names = tuple(name for name in stack[1:-1] if "rg2_gripper" in name)
        moveit_name = stack[-1]
        if key == "ur5e":
            err = self._start_ur5e_rtde_trajectory_server(driver_name)
            if err:
                return f"{key} RTDE trajectory server is not ready: {err}"
            if self.ros2_proc_status(moveit_name) != "running":
                err = self.ros2_start(moveit_name)
                if err:
                    return err
            err = self._wait_with_ros2_daemon_retry(
                f"{key} MoveIt execute trajectory",
                lambda: self._wait_for_ros_action(
                    "/execute_trajectory",
                    timeout_sec=22.0,
                    process_name=moveit_name,
                ),
            )
            if err:
                return (
                    f"{key} MoveIt is not ready: {err}. "
                    "MoveIt may have failed to launch correctly."
                )
            err = self._wait_with_ros2_daemon_retry(
                f"{key} RTDE /joint_states publisher",
                lambda: self._wait_for_ros_topic_publisher(
                    "/joint_states",
                    timeout_sec=20.0,
                    process_name=driver_name,
                ),
            )
            if err:
                return f"{key} RTDE trajectory server is not publishing /joint_states: {err}"
            for gripper_name in gripper_names:
                if self.ros2_proc_status(gripper_name) != "running":
                    err = self.ros2_start(gripper_name)
                    if err:
                        return err
                err = self._wait_with_ros2_daemon_retry(
                    f"{key} RG2 gripper bridge",
                    lambda: self._wait_for_ros_action(
                        _UR5E_RG2_GRIPPER_ACTION,
                        timeout_sec=12.0,
                        process_name=gripper_name,
                    ),
                )
                if err:
                    return f"{key} RG2 gripper bridge is not ready: {err}"
            return None
        if self.ros2_proc_status(driver_name) != "running":
            err = self.ros2_start(driver_name)
            if err:
                return err

        err = self._wait_with_ros2_daemon_retry(
            f"{key} driver ready",
            lambda: self._wait_for_driver_ready(key, timeout_sec=18.0),
        )
        if err:
            return err
        err = self._wait_with_ros2_daemon_retry(
            f"{key} /joint_states publisher",
            lambda: self._wait_for_ros_topic_publisher(
                "/joint_states",
                timeout_sec=40.0,
                process_name=driver_name,
            ),
        )
        if err:
            return f"{key} driver is not publishing /joint_states: {err}"

        for gripper_name in gripper_names:
            if self.ros2_proc_status(gripper_name) != "running":
                err = self.ros2_start(gripper_name)
                if err:
                    return err
            err = self._wait_with_ros2_daemon_retry(
                f"{key} RG2 gripper bridge",
                lambda: self._wait_for_ros_action(
                    _UR5E_RG2_GRIPPER_ACTION,
                    timeout_sec=12.0,
                    process_name=gripper_name,
                ),
            )
            if err:
                return f"{key} RG2 gripper bridge is not ready: {err}"

        if self.ros2_proc_status(moveit_name) != "running":
            err = self.ros2_start(moveit_name)
            if err:
                return err
        err = self._wait_with_ros2_daemon_retry(
            f"{key} MoveIt execute trajectory",
            lambda: self._wait_for_ros_action(
                "/execute_trajectory",
                timeout_sec=22.0,
                process_name=moveit_name,
            ),
        )
        if err:
            return (
                f"{key} MoveIt is not ready: {err}. "
                "MoveIt may have failed to launch correctly."
            )
        return None

    def ros2_stop_hardware_stack(self, robot: str) -> str | None:
        key = str(robot).strip().lower()
        stack = self._hardware_stack_for_robot(key)
        if not stack:
            return f"unknown hardware robot: {robot}"
        if len(stack) == 1:
            self.ros2_stop(stack[0])
            # Also stop potential legacy standalone driver if user launched it.
            self.ros2_stop("hardware_xarm6_driver")
        else:
            for process_name in reversed(stack):
                self.ros2_stop(process_name)
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
        keep_controller_alive = str(
            os.environ.get("CAIS_KEEP_GAZEBO_PREWARM_CONTROLLERS", "0")
        ).strip().lower() not in {"0", "false", "no", "off"}

        with self._gazebo_prewarm_lock:
            existing = self._gazebo_prewarm_controllers.get(robot_key)
        if existing is not None:
            if keep_controller_alive:
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
            from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
                UR5E_JOINT_NAMES,
                UR5E_JOINT_STATES_TOPIC,
                XARM6_JOINT_NAMES,
                XARM6_JOINT_STATES_TOPIC,
                GazeboPickPlaceController,
            )
        except Exception:
            log.exception("Gazebo prewarm failed importing controller modules for %s", robot_key)
            return False, time.monotonic() - start_ts, "controller import failure"

        controller = None
        detail = ""
        try:
            if robot_key == "xarm6":
                prewarm_node = f"xarm6_prewarm_controller_{os.getpid()}_{int(time.monotonic() * 1000) % 1000000}"
                controller = GazeboPickPlaceController(
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
                controller = GazeboPickPlaceController(
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
                if keep_controller_alive:
                    with self._gazebo_prewarm_lock:
                        old = self._gazebo_prewarm_controllers.pop(robot_key, None)
                        self._gazebo_prewarm_controllers[robot_key] = controller
                    if old is not None and old is not controller:
                        try:
                            old.shutdown()
                        except Exception:
                            pass
                    controller = None  # kept alive for reuse; cleaned up when Gazebo stops
                else:
                    try:
                        controller.shutdown()
                    except Exception:
                        log.exception("Gazebo prewarm cleanup failed for %s", robot_key)
                    controller = None
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
    def _force_kill_digital_twin_helpers() -> None:
        """Hard-kill digital-twin stragglers that survive ros2 launch shutdown.

        Teach mode spawns sim + hardware MoveIt (two move_group + two rviz2); killing the
        tracked launch process group does not always take these down. Safe to pattern-kill
        here because a running digital twin blocks all other stacks (see
        _digital_twin_blocked_reason), so no unrelated move_group/rviz2 is expected.
        """
        for cmd in [
            "pkill -9 -f move_group 2>/dev/null",
            "pkill -9 -f rviz2 2>/dev/null",
            "pkill -9 -f dual_drag_markers.py 2>/dev/null",
            "pkill -9 -f digital_twin_sync.py 2>/dev/null",
            "pkill -9 -f xarm6_hardware_driver.launch.py 2>/dev/null",
            "pkill -9 -f XArm6JointStateRelay 2>/dev/null",
            "pkill -9 -f dual_robots_hardware_moveit.launch.py 2>/dev/null",
            "pkill -9 -f ur5e_rg2_hardware_moveit.launch.py 2>/dev/null",
            "pkill -9 -f xarm6_moveit_realmove.launch.py 2>/dev/null",
            "pkill -9 -f ur5e_rg2_rtde_gripper.py 2>/dev/null",
            "pkill -9 -f ur5e_rtde_trajectory_server.py 2>/dev/null",
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

    def ros2_start(
        self,
        name: str,
        *,
        fast_forward_simulation: bool | None = None,
    ) -> str | None:
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

        if name == "hardware_ur5e_moveit" or name == "hardware_dual_robots_moveit":
            err = self._start_ur5e_rtde_trajectory_server("hardware_ur5e_rtde_trajectory_server")
            if err:
                return f"UR5e RTDE trajectory server is not ready: {err}"

        cmd = self._ROS2_ENV + self._render_ros2_launch_cmd(
            name,
            fast_forward_simulation=fast_forward_simulation,
        )
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
            "pkill -9 -f dual_drag_markers.py 2>/dev/null",
            "pkill -9 -f digital_twin_sync.py 2>/dev/null",
            "pkill -9 -f xarm6_hardware_driver.launch.py 2>/dev/null",
            "pkill -9 -f XArm6JointStateRelay 2>/dev/null",
            "pkill -9 -f ur5e_rg2_rtde_gripper.py 2>/dev/null",
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
            "pkill -9 -f dual_drag_markers.py 2>/dev/null",
            "pkill -9 -f spawn_entity.py 2>/dev/null",
            "pkill -9 -f xarm_driver_node 2>/dev/null",
            "pkill -9 -f controller_manager 2>/dev/null",
            "pkill -9 -f 'spawner' 2>/dev/null",
            "pkill -9 -f xarm6_hardware_driver.launch.py 2>/dev/null",
            "pkill -9 -f XArm6JointStateRelay 2>/dev/null",
            "pkill -9 -f xarm6_moveit_realmove.launch.py 2>/dev/null",
            "pkill -9 -f ur_moveit.launch.py 2>/dev/null",
            "pkill -9 -f ur5e_rg2_hardware_moveit.launch.py 2>/dev/null",
            "pkill -9 -f ur5e_rg2_rtde_gripper.py 2>/dev/null",
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
            from cais_spade_llm.resources.robot.gazebo_pick_place_controller import (
                UR5E_JOINT_NAMES,
                UR5E_JOINT_STATES_TOPIC,
                XARM6_JOINT_NAMES,
                XARM6_JOINT_STATES_TOPIC,
                GazeboPickPlaceController,
            )
        except Exception as exc:
            return None, False, f"{robot_key}: controller import failed ({exc})"

        try:
            if robot_key == "xarm6":
                controller = GazeboPickPlaceController(
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
                controller = GazeboPickPlaceController(
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
            self._teleop_server_ros_domain_id = None
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
        self._teleop_server_ros_domain_id = None

    def _stop_teleop_server(self) -> None:
        with self._teleop_server_lock:
            self._stop_teleop_server_locked()

    def _teleop_stderr_excerpt_locked(self, max_chars: int = 2000) -> str:
        proc = self._teleop_server_proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            fd = proc.stderr.fileno()
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return ""
            data = os.read(fd, max(1, int(max_chars)))
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def _read_teleop_response_locked(self, timeout_sec: float) -> tuple[bool, str, dict[str, Any]]:
        proc = self._teleop_server_proc
        if proc is None or proc.stdout is None:
            return False, "teleop server not running", {}
        if proc.poll() is not None:
            code = proc.returncode
            stderr = self._teleop_stderr_excerpt_locked()
            self._stop_teleop_server_locked()
            detail = f"teleop server exited ({code})"
            if stderr:
                detail = f"{detail}: {stderr}"
            return False, detail, {}

        ready, _, _ = select.select([proc.stdout.fileno()], [], [], max(0.1, timeout_sec))
        if not ready:
            stderr = self._teleop_stderr_excerpt_locked()
            detail = f"teleop response timeout after {timeout_sec:.1f}s"
            if stderr:
                detail = f"{detail}: {stderr}"
            return False, detail, {}

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

    def _ensure_teleop_server_locked(self, ros_domain_id: int | None) -> str | None:
        try:
            resolved_domain_id = int(ros_domain_id) if ros_domain_id is not None else self._default_ros_domain_id()
        except (TypeError, ValueError):
            resolved_domain_id = self._default_ros_domain_id()
        proc = self._teleop_server_proc
        if (
            proc is not None
            and proc.poll() is None
            and self._teleop_server_ros_domain_id == resolved_domain_id
        ):
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
                ["bash", "-c", self._ROS2_ENV + self._ros2_domain_export(resolved_domain_id) + cmd],
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
        self._teleop_server_ros_domain_id = resolved_domain_id
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
        ros_domain_id: int | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        with self._teleop_server_lock:
            err = self._ensure_teleop_server_locked(ros_domain_id)
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
                err = self._ensure_teleop_server_locked(ros_domain_id)
                if err:
                    return False, err, {}
                proc = self._teleop_server_proc
                if proc is None or proc.stdin is None:
                    return False, "teleop server unavailable after restart", {}
                proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                return self._read_teleop_response_locked(timeout_sec=timeout_sec)
            return ok, msg, response_payload

    def _teleop_preflight(self, robot: str, op: str) -> dict[str, Any]:
        target = self.teleop_target(robot, op)
        warning = str(target.get("warning") or "")
        if warning:
            return target
        return target

    def _teleop_request(self, payload: dict[str, Any], timeout_sec: float) -> tuple[bool, str]:
        robot = str(payload.get("robot") or "").strip().lower()
        op = str(payload.get("op") or "").strip().lower()
        target = self._teleop_preflight(robot, op)
        warning = str(target.get("warning") or "")
        if warning:
            return False, warning
        ok, msg, _payload = self._teleop_request_payload(
            payload=payload,
            timeout_sec=timeout_sec,
            ros_domain_id=target.get("ros_domain_id"),
        )
        return ok, msg

    def _run_teleop_once(
        self,
        args: list[str],
        timeout_sec: float,
        ros_domain_id: int | None = None,
    ) -> tuple[bool, str]:
        quoted_script = shlex.quote(self._TELEOP_SCRIPT)
        quoted_args = " ".join(shlex.quote(str(arg)) for arg in args)
        cmd = f"python3.10 {quoted_script} {quoted_args}".strip()
        return self.ros2_exec(
            self._ros2_domain_export(ros_domain_id) + cmd,
            timeout_sec=timeout_sec,
        )

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
        target = self._teleop_preflight(robot, "gripper")
        warning = str(target.get("warning") or "")
        if warning:
            return False, warning
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
        return self._run_teleop_once(
            once_args,
            timeout_sec=10.0,
            ros_domain_id=target.get("ros_domain_id"),
        )

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
        env = str(self.teleop_target(robot).get("environment") or self.teleop_target_environment())
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
        target = self._teleop_preflight(robot, "save_position")
        warning = str(target.get("warning") or "")
        if warning:
            return False, warning
        return self._teleop_request(
            payload={
                "op": "save_position",
                "robot": robot,
                "name": position_name,
                "env": str(target.get("environment") or self.teleop_target_environment()),
            },
            timeout_sec=6.0,
        )

    def teleop_state(self, robot: str) -> tuple[bool, str, dict[str, Any]]:
        robot = str(robot).strip().lower()
        if robot not in {"xarm6", "ur5e"}:
            return False, f"unknown robot: {robot}", {}
        target = self._teleop_preflight(robot, "state")
        warning = str(target.get("warning") or "")
        if warning:
            return False, warning, {}
        ok, msg, payload = self._teleop_request_payload(
            payload={
                "op": "state",
                "robot": robot,
            },
            timeout_sec=4.0,
            ros_domain_id=target.get("ros_domain_id"),
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

    def get_current_task_dag_nodes(self) -> list[dict[str, Any]]:
        nodes = []
        for pa in self.product_agents:
            pp = getattr(pa, "process_planner", None)
            if not pp:
                continue
            task_states = getattr(pa, "task_states", {})
            if not isinstance(task_states, dict):
                task_states = {}
            runtime = getattr(pp, "product_order_runtime", {})
            if (
                isinstance(runtime, dict)
                and runtime.get("enabled")
            ):
                committed_parts = {
                    str(part or "").strip()
                    for part in (runtime.get("committed_product_order_parts") or [])
                    if str(part or "").strip()
                }
                completed_parts = {
                    str(part or "").strip()
                    for part in (runtime.get("completed_product_order_parts") or [])
                    if str(part or "").strip()
                }
                visible_parts = committed_parts - completed_parts
                visible_ids: set[str] = set()
                visible_nodes: list[dict[str, Any]] = []
                for node in getattr(pp, "nodes", []) or []:
                    if not isinstance(node, dict) or node.get("type") != "task":
                        continue
                    part_name = str(node.get("product_order_part") or "").strip()
                    if part_name not in visible_parts:
                        continue
                    node_id = str(node.get("id") or node.get("task_id") or "").strip()
                    node_copy = deepcopy(node)
                    if node_id:
                        visible_ids.add(node_id)
                        if node_id in task_states:
                            node_copy["status"] = str(task_states.get(node_id) or node_copy.get("status") or "pending")
                    visible_nodes.append(node_copy)
                for node in visible_nodes:
                    node["predecessors"] = [
                        pred
                        for pred in (node.get("predecessors") or [])
                        if str(pred or "").strip() in visible_ids
                    ]
                    node["successors"] = [
                        succ
                        for succ in (node.get("successors") or [])
                        if str(succ or "").strip() in visible_ids
                    ]
                nodes.extend(visible_nodes)
                continue
            if hasattr(pp, "nodes"):
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
