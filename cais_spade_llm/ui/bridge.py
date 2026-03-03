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
import subprocess
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

        # SPADE XMPP server task.
        self._xmpp_server = None
        self._xmpp_task: Optional[asyncio.Task] = None

        # Lifecycle flags.
        self.system_running: bool = False
        self._starting: bool = False
        self._stopping: bool = False
        self.last_error: Optional[str] = None

        # Configuration (set from UI before start).
        self.execution_mode: str = "ros2"
        self.robot_env: str = "gazebo"
        self.selected_product: str = ""

        # ROS2 subprocess tracking.
        self._ros2_procs: dict[str, subprocess.Popen] = {}
        self._teleop_server_proc: Optional[subprocess.Popen] = None
        self._teleop_server_lock = threading.Lock()
        self.hardware_ips = self._load_hardware_ips()
        self._hw_ping_cache: dict[str, dict[str, Any]] = {}
        self._hw_ping_last_ts: float = 0.0

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
    async def _ensure_xmpp_server(self) -> None:
        """Start the embedded XMPP server if it isn't already running."""
        if self._xmpp_task is not None and not self._xmpp_task.done():
            return
        import loguru
        from pyjabber.server import Server
        from pyjabber.server_parameters import Parameters

        loguru.logger.remove()
        self._xmpp_server = Server(Parameters(host="localhost", database_in_memory=True))
        self._xmpp_task = asyncio.create_task(self._xmpp_server.start())
        await self._xmpp_server.ready.wait()
        log.info("Embedded XMPP server started on localhost:5222")

    async def _stop_xmpp_server(self) -> None:
        if self._xmpp_task is not None:
            self._xmpp_task.cancel()
            try:
                await self._xmpp_task
            except asyncio.CancelledError:
                pass
            self._xmpp_task = None
            self._xmpp_server = None

    # ------------------------------------------------------------------
    # System lifecycle
    # ------------------------------------------------------------------
    async def start_system(self) -> None:
        """Start the SPADE agent system (mirrors spade_main.spade_main)."""
        if self.system_running or self._starting:
            return
        self._starting = True
        self.last_error = None

        try:
            # Archive previous monitor outputs.
            self._archive_monitors()

            # Set environment variables that agent_creator reads.
            os.environ["ROBOT_ENV"] = self.robot_env
            if self.robot_env == "gazebo":
                os.environ["USE_ROS2_CAMERA"] = "0"
            else:
                os.environ.pop("USE_ROS2_CAMERA", None)

            # Ensure XMPP server is up.
            await self._ensure_xmpp_server()

            # Ensure the SPADE Container uses the current running loop.
            from spade.container import Container
            container = Container()
            container.loop = asyncio.get_running_loop()

            # Import agent_creator from the project (uses sys.path set by ui_main).
            import importlib
            import agent_creator as ac
            importlib.reload(ac)  # Re-read env vars.
            from function_analyzer import FunctionAnalyzer
            import utils

            # Collect init files.
            prod_files = utils.get_init_files(str(_PRODUCT_DIR))
            res_files = utils.get_init_files(str(_RESOURCE_DIR))

            # Create agents.
            self.user_agent = ac.create_user()
            self.resource_agents = ac.create_resource_agents(res_files, str(_CCA_INIT))
            self.product_agents = ac.create_product_agents(prod_files, self.resource_agents, str(_CCA_INIT))
            self.cca = ac.create_central_controller(str(_CCA_INIT), self.resource_agents)

            # Build tools catalogue.
            FunctionAnalyzer.build_tools_catalogue(
                agents=self.product_agents + self.resource_agents,
                allowed=ac.ALLOWED_FUNCS,
                outfile=str(_TOOLS_OUT),
            )

            # Start agents in order: resources → CCA → user → products.
            for ra in self.resource_agents:
                await ra.start(auto_register=True)

            if self.cca:
                await self.cca.start(auto_register=True)

            if self.user_agent:
                await self.user_agent.start(auto_register=True)

            for i, pa in enumerate(self.product_agents):
                await asyncio.sleep(0.2 * i)
                await pa.start(auto_register=True)

            self.system_running = True
            log.info("All agents started successfully.")

        except Exception as exc:
            self.last_error = str(exc)
            log.exception("Failed to start system")
            await self._cleanup_agents()
        finally:
            self._starting = False

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

    def _ros2_command_output(self, command: str, timeout_sec: float = 8.0) -> tuple[bool, str]:
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
        return True, cp.stdout or ""

    def _wait_for_ros_service(
        self,
        service_name: str,
        timeout_sec: float = 20.0,
        process_name: str | None = None,
    ) -> str | None:
        target = str(service_name).strip()
        if not target:
            return "service name is empty"

        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        while time.monotonic() < deadline:
            if process_name and self.ros2_proc_status(process_name) != "running":
                return f"{process_name} exited before {target} became available"

            ok, out = self._ros2_command_output("ros2 service list", timeout_sec=3.0)
            if ok:
                services = [line.strip() for line in out.splitlines() if line.strip()]
                if any(s == target or s.endswith(target) for s in services):
                    return None
            time.sleep(0.7)

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

    def ros2_start(self, name: str) -> str | None:
        """Start a ROS2 process by name. Returns error string or None on success."""
        if name not in self.ROS2_LAUNCH_CMDS:
            return f"Unknown process: {name}"
        if self.ros2_proc_status(name) == "running":
            return f"{name} is already running"

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
            return None
        except Exception as exc:
            return str(exc)

    def ros2_stop(self, name: str) -> str | None:
        """Stop a tracked ROS2 process. Returns error string or None on success."""
        proc = self._ros2_procs.get(name)
        if proc is None or proc.poll() is not None:
            self._ros2_procs.pop(name, None)
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
        return None

    def ros2_stop_all(self) -> None:
        """Stop all tracked ROS2 processes."""
        self._stop_teleop_server()
        for name in list(self._ros2_procs):
            self.ros2_stop(name)

    def ros2_kill_gazebo(self) -> None:
        """Kill any orphan Gazebo / ROS2 processes (cleanup helper)."""
        self._stop_teleop_server()
        for cmd in [
            (
                "killall -9 gzserver gzclient robot_state_publisher spawner spawn_entity.py "
                "rviz2 move_group static_transform_publisher ros2 2>/dev/null"
            ),
            "pkill -9 -f gazebo 2>/dev/null",
            "pkill -9 -f keyboard_teleop.py 2>/dev/null",
        ]:
            subprocess.run(["bash", "-c", cmd], capture_output=True)

    def ros2_cleanup_processes(self) -> None:
        """Aggressively clean stale ROS2/MoveIt/driver processes without killing the UI."""
        self._stop_teleop_server()
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
