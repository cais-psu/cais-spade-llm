"""Connect the home Gazebo viewer to requests from the recovery-framework UI."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from starlette.responses import FileResponse

if TYPE_CHECKING:
    from cais_spade_llm.ui.bridge import SystemBridge

_SERVER_ID = uuid.uuid4().hex
_revision = 0
_last_poll: float | None = None
_LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "home_gazebo_viewer.py"


def home_gazebo_viewer_connected(client_ip: str) -> bool:
    """Return whether this browser shares the connected home launcher's address."""
    home_ip = os.environ.get("CAIS_HOME_GAZEBO_IP", "")
    return bool(
        home_ip
        and client_ip == home_ip
        and _last_poll is not None
        and time.monotonic() - _last_poll < 10.0
    )


def request_home_gazebo_viewer(client_ip: str) -> bool:
    """Ask the connected home launcher to open its Gazebo viewer once."""
    global _revision
    if not home_gazebo_viewer_connected(client_ip):
        return False
    _revision += 1
    return True


def register_home_gazebo_viewer_routes(bridge: SystemBridge) -> None:
    """Register polling and installation routes for the configured home address."""
    from nicegui import app

    def _require_home(request: Request) -> None:
        home_ip = os.environ.get("CAIS_HOME_GAZEBO_IP", "")
        if not home_ip or request.client is None or request.client.host != home_ip:
            raise HTTPException(status_code=403, detail="Home Gazebo viewer is not configured here.")

    @app.get("/home-gazebo-viewer/request")
    async def home_gazebo_viewer_request(request: Request) -> dict[str, object]:
        """Report the home request and simulation status without starting processes."""
        global _last_poll
        _require_home(request)
        _last_poll = time.monotonic()
        running = await asyncio.to_thread(bridge.simulation_environment_running)
        return {
            "server_id": _SERVER_ID,
            "revision": _revision,
            "simulation_running": running,
        }

    @app.get("/home-gazebo-viewer/launcher.py")
    def home_gazebo_viewer_launcher(request: Request) -> FileResponse:
        """Serve the standalone launcher for one installation in home WSL."""
        _require_home(request)
        return FileResponse(_LAUNCHER, media_type="text/x-python", filename=_LAUNCHER.name)
