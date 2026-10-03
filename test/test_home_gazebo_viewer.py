"""No-motion coverage for Start Simulation opening the home Gazebo viewer."""

from __future__ import annotations

import base64
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import nicegui
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cais_spade_llm.ui import home_gazebo_viewer as school

_HOME_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/home_gazebo_viewer.py"
_HOME_SPEC = importlib.util.spec_from_file_location("cais_home_gazebo_viewer", _HOME_SCRIPT)
home = importlib.util.module_from_spec(_HOME_SPEC)
_HOME_SPEC.loader.exec_module(home)

HOME_IP = "100.108.30.10"
UI_URL = "http://100.111.96.116:8080"
MASTER_URI = "http://100.111.96.116:11345"


@pytest.fixture
def school_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()
    monkeypatch.setattr(nicegui, "app", app)
    monkeypatch.setenv("CAIS_HOME_GAZEBO_IP", HOME_IP)
    monkeypatch.setattr(school, "_revision", 0)
    monkeypatch.setattr(school, "_last_poll", None)
    school.register_home_gazebo_viewer_routes(
        SimpleNamespace(simulation_environment_running=lambda: True)
    )
    return app


def test_only_a_connected_home_browser_requests_its_viewer(school_app: FastAPI) -> None:
    assert not school.request_home_gazebo_viewer(HOME_IP)
    with TestClient(school_app, client=(HOME_IP, 40000)) as client:
        status = client.get("/home-gazebo-viewer/request").json()
        assert status["revision"] == 0
        assert status["simulation_running"] is True
        assert school.home_gazebo_viewer_connected(HOME_IP)
        assert not school.request_home_gazebo_viewer("127.0.0.1")
        assert not school.request_home_gazebo_viewer("100.111.96.116")
        assert school.request_home_gazebo_viewer(HOME_IP)
        assert client.get("/home-gazebo-viewer/request").json()["revision"] == 1


def test_other_addresses_cannot_poll_or_download(school_app: FastAPI) -> None:
    with TestClient(school_app, client=("192.168.1.230", 40000)) as client:
        assert client.get("/home-gazebo-viewer/request").status_code == 403
        assert client.get("/home-gazebo-viewer/launcher.py").status_code == 403
    assert school._last_poll is None
    assert school._revision == 0


def test_home_can_download_the_standalone_installer(school_app: FastAPI) -> None:
    with TestClient(school_app, client=(HOME_IP, 40000)) as client:
        reply = client.get("/home-gazebo-viewer/launcher.py")
    assert reply.status_code == 200
    assert "class HomeGazeboViewer" in reply.text
    assert "from cais_spade_llm" not in reply.text
    assert school._revision == 0


def test_offline_launcher_does_not_enable_home_requests(
    school_app: FastAPI, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(school, "_last_poll", school.time.monotonic() - 11)
    assert not school.home_gazebo_viewer_connected(HOME_IP)
    assert not school.request_home_gazebo_viewer(HOME_IP)


def test_polling_with_no_home_configuration_fails_closed(
    school_app: FastAPI, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAIS_HOME_GAZEBO_IP")
    with TestClient(school_app, client=(HOME_IP, 40000)) as client:
        assert client.get("/home-gazebo-viewer/request").status_code == 403
    assert not school.request_home_gazebo_viewer(HOME_IP)


def test_new_click_opens_once_and_reopens_a_closed_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = home.HomeGazeboViewer(UI_URL, MASTER_URI, HOME_IP)
    status = {"server_id": "school", "revision": 0, "simulation_running": True}
    open_now = {"value": False}
    launches = []
    monkeypatch.setattr(viewer, "_status", lambda: status)
    monkeypatch.setattr(viewer, "_existing_viewer", lambda: open_now["value"])

    def _start() -> None:
        launches.append(status["revision"])
        open_now["value"] = True

    monkeypatch.setattr(viewer, "_start_viewer", _start)
    viewer.poll_once()
    assert launches == []
    status["revision"] = 1
    viewer.poll_once()
    viewer.poll_once()
    status["revision"] = 2
    viewer.poll_once()
    assert launches == [1]
    open_now["value"] = False
    viewer.poll_once()
    assert launches == [1]
    status["revision"] = 3
    viewer.poll_once()
    assert launches == [1, 3]


def test_stopped_simulation_closes_only_the_launchers_own_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = home.HomeGazeboViewer(UI_URL, MASTER_URI, HOME_IP)
    monkeypatch.setattr(viewer, "_status", lambda: {
        "server_id": "school", "revision": 2, "simulation_running": False,
    })
    launch = MagicMock()
    monkeypatch.setattr(viewer, "_start_viewer", launch)
    viewer.poll_once()
    launch.assert_not_called()
    owned_process = MagicMock()
    owned_process.poll.return_value = None
    viewer._viewer = owned_process
    viewer.poll_once()
    owned_process.terminate.assert_called_once()
    owned_process.wait.assert_called_once_with(timeout=3)
    assert viewer._viewer is None


def test_ui_restart_still_accepts_a_new_home_click(monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = home.HomeGazeboViewer(UI_URL, MASTER_URI, HOME_IP)
    status = {"server_id": "old_ui", "revision": 1, "simulation_running": True}
    monkeypatch.setattr(viewer, "_status", lambda: status)
    monkeypatch.setattr(viewer, "_existing_viewer", lambda: False)
    launch = MagicMock()
    monkeypatch.setattr(viewer, "_start_viewer", launch)
    viewer.poll_once()
    status.update(server_id="new_ui", revision=0)
    viewer.poll_once()
    assert launch.call_count == 1
    status["revision"] = 1
    viewer.poll_once()
    assert launch.call_count == 2


@pytest.mark.parametrize("response", [
    [],
    {"server_id": "school", "revision": True, "simulation_running": True},
    {"server_id": "school", "revision": -1, "simulation_running": True},
    {"server_id": "school", "revision": 1, "simulation_running": "true"},
    {"server_id": "", "revision": 1, "simulation_running": True},
])
def test_malformed_school_responses_never_launch_a_process(response: object) -> None:
    viewer = home.HomeGazeboViewer(UI_URL, MASTER_URI, HOME_IP)
    reply = MagicMock()
    reply.__enter__.return_value.read.return_value = json.dumps(response).encode()
    viewer._opener = MagicMock()
    viewer._opener.open.return_value = reply
    viewer._start_viewer = MagicMock()
    with pytest.raises(ValueError, match="Invalid response"):
        viewer.poll_once()
    viewer._start_viewer.assert_not_called()


def test_spawn_preserves_model_uris_and_loads_the_installed_viewer_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(home.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(home, "_CACHE", tmp_path / "logs")
    monkeypatch.setenv("GAZEBO_HOSTNAME", "old-host")
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/opencv/qt")
    monkeypatch.setenv("GAZEBO_MODEL_PATH", "/existing/models")
    monkeypatch.setenv("GAZEBO_PLUGIN_PATH", "/existing/plugins")
    monkeypatch.setenv("DISPLAY", ":0")
    process = MagicMock()
    monkeypatch.setattr(home.subprocess, "Popen", process)
    uris = [
        "cais_lab_robotics/models/KMR/meshes/lbr_iiwa_14_r820/visual/base_link.dae",
        "cad_models/KET4_Square_4mm.STL",
    ]
    ros = tmp_path / "projects/cais-spade-llm/ros2"
    for uri in uris:
        asset = ros / uri if uri.startswith("cais_lab_robotics/") else ros / "cais_lab_robotics" / uri
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.touch()
    viewer = home.HomeGazeboViewer(UI_URL, MASTER_URI, HOME_IP)
    viewer._start_viewer()
    args, kwargs = process.call_args
    assert args[0] == ["/usr/bin/gzclient", "--verbose"]
    env = kwargs["env"]
    assert env["GAZEBO_MASTER_URI"] == MASTER_URI
    assert env["GAZEBO_IP"] == HOME_IP
    assert env["DISPLAY"] == ":0"
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert "GAZEBO_HOSTNAME" not in env
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in env
    paths = [Path(path) for path in env["GAZEBO_MODEL_PATH"].split(":")]
    assert all(any((path / uri).is_file() for path in paths) for uri in uris)
    assert env["GAZEBO_PLUGIN_PATH"].split(":") == [
        str(tmp_path / "ros2_ws/install/cais_lab_robotics/lib"), "/existing/plugins",
    ]
    assert "/existing/models" in env["GAZEBO_MODEL_PATH"].split(":")


def test_installer_refuses_the_school_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(home.socket, "gethostname", lambda: "Jonghan")
    run = MagicMock()
    monkeypatch.setattr(home.subprocess, "run", run)
    with pytest.raises(ValueError, match="home-pc"):
        home._install(UI_URL, MASTER_URI, HOME_IP)
    run.assert_not_called()


def test_installer_creates_only_the_home_startup_shortcut_and_background_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(home.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(home, "_CACHE", tmp_path / "logs")
    monkeypatch.setattr(home.socket, "gethostname", lambda: "home-pc")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    run, spawn = MagicMock(), MagicMock()
    monkeypatch.setattr(home.subprocess, "run", run)
    monkeypatch.setattr(home.subprocess, "Popen", spawn)
    home._install(UI_URL, MASTER_URI, HOME_IP)
    destination = tmp_path / ".local/bin/cais_home_gazebo_viewer.py"
    assert destination.is_file()
    command = run.call_args.args[0][-1]
    assert "GetFolderPath('Startup')" in command
    assert "CAIS Gazebo viewer.lnk" in command
    encoded = re.search(r"-EncodedCommand ([A-Za-z0-9+/=]+)", command).group(1)
    startup = base64.b64decode(encoded).decode("utf-16-le")
    assert "'Ubuntu-22.04'" in startup
    assert str(destination) in startup
    assert UI_URL in startup
    assert MASTER_URI in startup
    assert HOME_IP in startup
    assert spawn.call_args.kwargs["start_new_session"] is True
