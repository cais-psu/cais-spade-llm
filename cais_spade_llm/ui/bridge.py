"""SystemBridge: in-process bridge between SPADE agents and the NiceGUI operator console."""

from __future__ import annotations

import asyncio
import json
import logging
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("ui.bridge")

# Filesystem locations (mirror spade_main.py constants).
_BASE = Path(__file__).resolve().parent.parent          # cais_spade_llm/
_PROJECT_ROOT = _BASE.parent                             # repo root
_PRODUCT_DIR = _BASE / "initialization" / "products"
_RESOURCE_DIR = _BASE / "initialization" / "resources"
_TOOLS_OUT = _BASE / "initialization" / "tools.json"
_CCA_INIT = _BASE / "initialization" / "cca.json"
_MONITOR = _BASE / "monitor"
_LOG_DIR = _BASE / "log"
_XARM6_RESOURCE = _RESOURCE_DIR / "robot_xarm6.json"
_UR5E_RESOURCE = _RESOURCE_DIR / "robot_ur5e.json"


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
    _GAZEBO_PREWARM_START_DELAY_S = 2.0
    _GAZEBO_PREWARM_READY_WAIT_S = 60.0

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

        # Lifecycle flags.
        self.system_running: bool = False
        self._starting: bool = False
        self._stopping: bool = False
        self.last_error: Optional[str] = None

        # Configuration (set from UI before start).
        self.execution_mode: str = "simulation"
        self.robot_env: str = "gazebo"
        self.selected_product: str = ""

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
        self.hardware_ips = self._load_hardware_ips()
        self._hw_ping_cache: dict[str, dict[str, Any]] = {}
        self._hw_ping_last_ts: float = 0.0
        self._startup_seq: int = 0
        self._startup_phase: str = "idle"
        self._startup_phase_ts: float = time.monotonic()
        self._agent_creator_cached: Any | None = None
        self._agent_creator_prefetch_started: bool = False
        self._agent_creator_prefetch_lock = threading.Lock()
        self._agent_creator_prefetch_thread: Optional[threading.Thread] = None
        self._maybe_start_agent_creator_prefetch()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _diag_emit(self, message: str) -> None:
        """Emit high-signal startup diagnostics to stdout and logger."""
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
            )
            self._diag_emit(
                f"startup#{startup_id} agents created resources={len(self.resource_agents)} "
                f"products={len(self.product_agents)} in {time.monotonic() - startup_t0:.2f}s"
            )

            # Build tools catalogue.
            self._set_startup_phase("build_tools_catalogue")
            await asyncio.to_thread(
                FunctionAnalyzer.build_tools_catalogue,
                self.product_agents + self.resource_agents,
                ac.ALLOWED_FUNCS,
                str(_TOOLS_OUT),
            )
            self._diag_emit(
                f"startup#{startup_id} tools catalogue done in {time.monotonic() - startup_t0:.2f}s"
            )

            # Start agents in order: resources → CCA → user → products.
            self._set_startup_phase("start_resource_agents")
            for ra in self.resource_agents:
                name = getattr(ra, "agent_name", str(getattr(ra, "jid", "?")))
                t_ra = time.monotonic()
                self._diag_emit(f"startup#{startup_id} starting resource {name}")
                await ra.start(auto_register=True)
                self._diag_emit(
                    f"startup#{startup_id} resource {name} started in {time.monotonic() - t_ra:.2f}s"
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
            for i, pa in enumerate(self.product_agents):
                await asyncio.sleep(0.2 * i)
                name = getattr(pa, "agent_name", str(getattr(pa, "jid", "?")))
                t_pa = time.monotonic()
                self._diag_emit(f"startup#{startup_id} starting product {name}")
                await pa.start(auto_register=True)
                self._diag_emit(
                    f"startup#{startup_id} product {name} started in {time.monotonic() - t_pa:.2f}s"
                )

            self.system_running = True
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

    @staticmethod
    def _archive_monitors() -> None:
        for sub, pattern in [
            ("history", "*.jsonl"),
            ("plan", "*.json"),
            ("state", "*.json"),
            ("debug", "*.md"),
        ]:
            d = _MONITOR / sub
            if not d.exists():
                continue
            files = [p for p in d.glob(pattern) if p.is_file()]
            if not files:
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive = d / "archive" / stamp
            archive.mkdir(parents=True, exist_ok=True)
            for p in files:
                try:
                    shutil.move(str(p), str(archive / p.name))
                except Exception:
                    pass

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
    ) -> tuple[Any, list[Any], list[Any], Any]:
        user_agent = agent_creator_module.create_user()
        resource_agents = agent_creator_module.create_resource_agents(
            res_files,
            cca_init_file,
            prewarmed_controllers=prewarmed_controllers,
        )
        product_agents = agent_creator_module.create_product_agents(
            prod_files,
            resource_agents,
            cca_init_file,
        )
        cca = agent_creator_module.create_central_controller(
            cca_init_file,
            resource_agents,
        )
        return user_agent, resource_agents, product_agents, cca

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
        """Return whether Gazebo+MoveIt simulation startup is fully ready for agent start."""
        now = time.monotonic()
        if not self._any_running(self._GAZEBO_PROCESS_NAMES):
            result = (False, "Gazebo stack is not running. Launch Gazebo + MoveIt first.")
            self._sim_ready_cache_ts = now
            self._sim_ready_cache = result
            return result

        # If prewarm completed successfully, services were confirmed ready.
        if self._gazebo_prewarm_done.is_set():
            result = (True, "")
            self._sim_ready_cache = result
            self._sim_ready_cache_ts = now
            if force:
                return result
            return result

        if force:
            result = self._probe_sim_services(timeout_sec=6.0)
            self._sim_ready_cache_ts = now
            self._sim_ready_cache = result
            return result

        # Non-force path is called from a 1s UI timer. Keep it cheap and avoid
        # repeatedly spawning `ros2 service list`, which can be expensive.
        with self._gazebo_prewarm_lock:
            prewarm_inflight = bool(
                (self._gazebo_prewarm_thread and self._gazebo_prewarm_thread.is_alive())
                or self._gazebo_prewarm_pending
            )

        if prewarm_inflight:
            result = (False, "Simulation startup is still initializing ROS services. Please wait...")
            self._sim_ready_cache_ts = now
            self._sim_ready_cache = result
            return result

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
        required_services = ("/compute_cartesian_path", "/detect_all")
        missing = [
            svc
            for svc in required_services
            if not any(name == svc or name.endswith(svc) for name in services)
        ]
        if missing:
            return (
                False,
                "Simulation startup is not done yet. Waiting for services: "
                + ", ".join(missing),
            )
        return (True, "")

    def _probe_sim_services_worker(self) -> None:
        try:
            result = self._probe_sim_services(timeout_sec=3.0)
            self._sim_ready_cache = result
            self._sim_ready_cache_ts = time.monotonic()
        finally:
            with self._gazebo_prewarm_lock:
                self._sim_ready_probe_inflight = False

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
        target = str(service_name).strip()
        if not target:
            return "service name is empty"

        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        while time.monotonic() < deadline:
            if cancel_event and cancel_event.is_set():
                return f"{target} wait cancelled"

            if process_name and self.ros2_proc_status(process_name) != "running":
                return f"{process_name} exited before {target} became available"

            ok, out = self._ros2_command_output(
                "ros2 service list",
                timeout_sec=3.0,
                emit_slow_diag=False,
                emit_failure_diag=False,
                emit_timeout_diag=False,
            )
            if ok:
                services = [line.strip() for line in out.splitlines() if line.strip()]
                if any(s == target or s.endswith(target) for s in services):
                    return None
            time.sleep(1.5)

        if process_name and self.ros2_proc_status(process_name) != "running":
            return f"{process_name} exited before {target} became available"
        return f"{target} not available within {timeout_sec:.0f}s"

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

    def _prewarm_gazebo_controller(self, robot: str) -> None:
        robot_key = str(robot or "").strip().lower()
        settings = self._load_gazebo_controller_settings(robot_key)
        if settings is None:
            return
        controller_cfg, named_positions = settings

        with self._gazebo_prewarm_lock:
            existing = self._gazebo_prewarm_controllers.get(robot_key)
        if existing is not None:
            try:
                if existing.wait_for_services(timeout_sec=1.0):
                    log.info("Gazebo prewarm already ready for %s", robot_key)
                    return
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
                TRAJECTORY_TOPIC as UR5E_TRAJECTORY_TOPIC,
            )
            from cais_spade_llm.resources.robot.xarm6_controller import (
                JOINT_NAMES as XARM6_JOINT_NAMES,
                JOINT_STATES_TOPIC as XARM6_JOINT_STATES_TOPIC,
            )
        except Exception:
            log.exception("Gazebo prewarm failed importing controller modules for %s", robot_key)
            return

        controller = None
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
                    arm_trajectory_topic=UR5E_TRAJECTORY_TOPIC,
                    joint_states_topic=UR5E_JOINT_STATES_TOPIC,
                )
            else:
                return

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
            else:
                log.warning("Gazebo prewarm not ready for %s after %.2fs", robot_key, elapsed)
        except Exception:
            log.exception("Gazebo prewarm exception for %s", robot_key)
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

                # Phase 1: Wait until MoveIt + perception services appear via
                # lightweight shell probe (`ros2 service list`).  This avoids
                # creating heavyweight ROS2 controllers before the simulation
                # stack is actually ready.
                for svc in ("/compute_cartesian_path", "/detect_all"):
                    wait_err = self._wait_for_ros_service(
                        svc,
                        timeout_sec=self._GAZEBO_PREWARM_READY_WAIT_S,
                        cancel_event=self._gazebo_prewarm_cancel,
                    )
                    if wait_err:
                        log.info("Gazebo prewarm skipped (service not ready): %s", wait_err)
                        return

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
                for robot_key in targets:
                    if self._gazebo_prewarm_cancel.is_set():
                        return
                    self._prewarm_gazebo_controller(robot_key)

                # Signal that prewarm is done so simulation_start_ready() unblocks.
                self._gazebo_prewarm_done.set()
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

    @staticmethod
    def _kill_stale_gazebo_helpers() -> None:
        """Best-effort cleanup for helper nodes that can survive ros2 launch shutdown."""
        for cmd in [
            "pkill -9 -f auto_link_attacher_node.py 2>/dev/null",
            "pkill -9 -f gazebo_camera_detector.py 2>/dev/null",
        ]:
            subprocess.run(["bash", "-c", cmd], capture_output=True)

    def ros2_start(self, name: str) -> str | None:
        """Start a ROS2 process by name. Returns error string or None on success."""
        if name not in self.ROS2_LAUNCH_CMDS:
            return f"Unknown process: {name}"
        if self.ros2_proc_status(name) == "running":
            return f"{name} is already running"
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
            if not self._any_running(self._GAZEBO_PROCESS_NAMES):
                self._kill_stale_gazebo_helpers()

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
                self._queue_gazebo_prewarm(name)
            return None
        except Exception as exc:
            return str(exc)

    def ros2_stop(self, name: str) -> str | None:
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
        return None

    def ros2_stop_all(self) -> None:
        """Stop all tracked ROS2 processes."""
        self._stop_teleop_server()
        for name in list(self._ros2_procs):
            self.ros2_stop(name)
        self._shutdown_gazebo_prewarm_controllers()
        self._kill_stale_gazebo_helpers()

    def ros2_kill_gazebo(self) -> None:
        """Kill any orphan Gazebo / ROS2 processes (cleanup helper)."""
        self._stop_teleop_server()
        self._shutdown_gazebo_prewarm_controllers()
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

    def ros2_cleanup_processes(self) -> None:
        """Aggressively clean stale ROS2/MoveIt/driver processes without killing the UI."""
        self._stop_teleop_server()
        self._shutdown_gazebo_prewarm_controllers()
        for name in list(self._ros2_procs):
            self.ros2_stop(name)

        # Kill common stale processes that often block hardware reconnection.
        for cmd in [
            "killall -9 gzserver gzclient 2>/dev/null",
            "killall -9 move_group rviz2 robot_state_publisher joint_state_publisher static_transform_publisher ros2_control_node 2>/dev/null",
            "pkill -9 -f keyboard_teleop.py 2>/dev/null",
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

        self._ros2_procs = {k: p for k, p in self._ros2_procs.items() if p.poll() is None}

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

    def ros2_reset_gazebo_environment(self) -> tuple[bool, str]:
        """
        Reset Gazebo world/simulation to initial state without relaunching ROS2.

        This is intended for simulation workflows where the operator wants a clean
        scene between runs.
        """
        if not self._any_running(self._GAZEBO_PROCESS_NAMES):
            return False, "Gazebo is not running."
        if self.system_running or self._starting:
            return (
                False,
                "Stop the agent system before resetting Gazebo to avoid state mismatch.",
            )

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

        call_ok, call_msg = self.ros2_exec(
            f'ros2 service call {selected} std_srvs/srv/Empty "{{}}"',
            timeout_sec=10.0,
        )
        if not call_ok:
            return False, f"Gazebo reset failed via {selected}: {call_msg}"

        return True, f"Gazebo environment reset via {selected}."

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
            ts = getattr(pa, "task_states", {})
            merged.update(ts)
        return merged

    def get_execution_timeline(self) -> list[dict[str, Any]]:
        timeline = []
        for pa in self.product_agents:
            tl = getattr(pa, "execution_timeline", [])
            timeline.extend(tl)
        return timeline

    def get_part_tracker(self) -> dict[str, dict[str, Any]]:
        merged = {}
        for pa in self.product_agents:
            pt = getattr(pa, "part_tracker", {})
            merged.update(pt)
        return merged

    def get_safety_rules(self) -> list[dict[str, Any]]:
        if self.cca and hasattr(self.cca, "safety_rules"):
            return self.cca.safety_rules or []
        return []

    def get_safety_state(self) -> dict[str, Any]:
        if not self.cca:
            return {}
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
        return result

    def get_log_paths(self) -> dict[str, str]:
        paths = {}
        if _LOG_DIR.exists():
            for p in sorted(_LOG_DIR.glob("*.log")):
                paths[p.stem] = str(p)
        return paths
