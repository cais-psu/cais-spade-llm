"""SystemBridge: in-process bridge between SPADE agents and the NiceGUI operator console."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
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


class SystemBridge:
    """Singleton that owns the SPADE lifecycle and exposes agent state to the UI."""

    _instance: Optional[SystemBridge] = None

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
        self.execution_mode: str = "simulate"
        self.robot_env: str = "gazebo"
        self.selected_product: str = ""

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
