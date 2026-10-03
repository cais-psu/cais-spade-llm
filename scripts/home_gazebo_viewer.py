"""Open home gzclient when Start Simulation is clicked in the school CAIS UI."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

logger = logging.getLogger(__name__)
_CACHE = Path.home() / ".cache" / "cais-spade-llm" / "home-gazebo-viewer"


class HomeGazeboViewer:
    """Poll the school UI and own at most one home Gazebo viewer process."""

    def __init__(self, ui_url: str, master_uri: str, home_ip: str) -> None:
        """Configure the school UI, Gazebo master, and home Tailscale address."""
        self.ui_url = ui_url.rstrip("/")
        self.master_uri = master_uri
        self.home_ip = home_ip
        self._opener = build_opener(ProxyHandler({}))
        self._last_request: tuple[str, int] | None = None
        self._last_error: str | None = None
        self._viewer: subprocess.Popen[bytes] | None = None

    def _status(self) -> dict[str, object]:
        with self._opener.open(self.ui_url + "/home-gazebo-viewer/request", timeout=5) as reply:
            data = json.loads(reply.read(65536))
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("server_id"), str)
            or not data["server_id"]
            or type(data.get("revision")) is not int
            or data["revision"] < 0
            or type(data.get("simulation_running")) is not bool
        ):
            raise ValueError("Invalid response from the school CAIS UI")
        return data

    def _existing_viewer(self) -> bool:
        if self._viewer is not None and self._viewer.poll() is None:
            return True
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                if (entry / "comm").read_text().strip() != "gzclient":
                    continue
                env = (entry / "environ").read_bytes().split(b"\0")
                if ("GAZEBO_MASTER_URI=" + self.master_uri).encode() in env:
                    return True
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return False

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        ros = Path.home() / "projects" / "cais-spade-llm" / "ros2"
        model_paths = [str(ros), str(ros / "cais_lab_robotics"), str(ros / "cais_lab_robotics/models")]
        if env.get("GAZEBO_MODEL_PATH"):
            model_paths.append(env["GAZEBO_MODEL_PATH"])
        plugin_paths = [str(Path.home() / "ros2_ws/install/cais_lab_robotics/lib")]
        if env.get("GAZEBO_PLUGIN_PATH"):
            plugin_paths.append(env["GAZEBO_PLUGIN_PATH"])
        env.update(
            GAZEBO_MASTER_URI=self.master_uri,
            GAZEBO_IP=self.home_ip,
            GAZEBO_MODEL_PATH=":".join(model_paths),
            GAZEBO_PLUGIN_PATH=":".join(plugin_paths),
            LIBGL_ALWAYS_SOFTWARE="1",
        )
        # The home address must take precedence over inherited hostname settings.
        env.pop("GAZEBO_HOSTNAME", None)
        env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        env.pop("QT_QPA_FONTDIR", None)
        return env

    def _start_viewer(self) -> None:
        _CACHE.mkdir(parents=True, exist_ok=True)
        with (_CACHE / "gzclient.log").open("wb") as output:
            self._viewer = subprocess.Popen(
                ["/usr/bin/gzclient", "--verbose"],
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        logger.info("Opened home Gazebo; viewer log: %s", _CACHE / "gzclient.log")

    def _stop_owned_viewer(self) -> None:
        if self._viewer is not None and self._viewer.poll() is None:
            self._viewer.terminate()
            try:
                self._viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._viewer.kill()
                self._viewer.wait(timeout=3)
        self._viewer = None

    def poll_once(self) -> None:
        """Handle one school request, preserving existing home viewers."""
        data = self._status()
        if self._last_error is not None:
            logger.info("Connected to the school CAIS UI")
            self._last_error = None
        if not data["simulation_running"]:
            self._stop_owned_viewer()
            return
        if data["revision"] == 0:
            return
        request = (str(data["server_id"]), int(data["revision"]))
        if request == self._last_request:
            return
        if not self._existing_viewer():
            self._start_viewer()
        self._last_request = request

    def run(self) -> None:
        """Wait for clicks in the home browser without opening inbound ports."""
        _CACHE.mkdir(parents=True, exist_ok=True)
        with (_CACHE / "launcher.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info("Home Gazebo launcher is already running")
                return
            logger.info("Waiting for Start Simulation at %s", self.ui_url)
            while True:
                try:
                    self.poll_once()
                except (URLError, OSError, ValueError) as error:
                    message = str(error)
                    if message != self._last_error:
                        logger.warning("Waiting for the school CAIS UI: %s", message)
                        self._last_error = message
                time.sleep(2)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _install(ui_url: str, master_uri: str, home_ip: str) -> None:
    if socket.gethostname() != "home-pc":
        raise ValueError("Run --install in home Ubuntu (home-pc).")
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        raise ValueError("Run --install in home WSL.")
    destination = Path.home() / ".local/bin/cais_home_gazebo_viewer.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if Path(__file__).resolve() != destination.resolve():
        shutil.copyfile(__file__, destination)
    arguments = [
        "-d", distro, "--exec", "python3", str(destination),
        "--ui-url", ui_url, "--master-uri", master_uri, "--home-ip", home_ip,
    ]
    command = "& \"$env:WINDIR\\System32\\wsl.exe\" " + " ".join(map(_ps_quote, arguments))
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    install_command = (
        "$caisShell = New-Object -ComObject WScript.Shell; "
        "$caisPath = Join-Path ([Environment]::GetFolderPath('Startup')) 'CAIS Gazebo viewer.lnk'; "
        "$caisShortcut = $caisShell.CreateShortcut($caisPath); "
        "$caisShortcut.TargetPath = \"$env:WINDIR\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\"; "
        "$caisShortcut.Arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand "
        + encoded + "'; "
        "$caisShortcut.WindowStyle = 7; $caisShortcut.Save()"
    )
    subprocess.run(
        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile",
         "-NonInteractive", "-Command", install_command],
        check=True, timeout=20, stdout=subprocess.DEVNULL,
    )
    _CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [sys.executable, str(destination), "--ui-url", ui_url,
         "--master-uri", master_uri, "--home-ip", home_ip],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info("Installed home launcher and Windows Startup shortcut")
    logger.info("Use the home browser at %s and click Start Simulation", ui_url)


def main() -> None:
    """Run the home viewer launcher or install it once for Windows login."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-url", default="http://100.111.96.116:8080")
    parser.add_argument("--master-uri", default="http://100.111.96.116:11345")
    parser.add_argument("--home-ip", default="100.108.30.10")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    _CACHE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(_CACHE / "launcher.log")],
    )
    if args.install:
        _install(args.ui_url, args.master_uri, args.home_ip)
    else:
        HomeGazeboViewer(args.ui_url, args.master_uri, args.home_ip).run()


if __name__ == "__main__":
    main()
