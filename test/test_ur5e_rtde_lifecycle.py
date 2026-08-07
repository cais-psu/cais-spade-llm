"""Focused lifecycle diagnostics for the UI-owned UR5e RTDE trajectory server."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui.bridge import SystemBridge


def _bridge_with_exited_process(process_name: str, return_code: int) -> SystemBridge:
    bridge = object.__new__(SystemBridge)
    bridge._ros2_procs = {
        process_name: SimpleNamespace(pid=3549, poll=lambda: return_code),
    }
    bridge._ros2_intentional_stops = set()
    bridge._unregister_ui_process = lambda _name, _proc: None
    return bridge


def test_main_rtde_command_retains_fatal_signal_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process_name = "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server"
    bridge = object.__new__(SystemBridge)
    started: list[tuple[str, str, int | None]] = []
    stopped: list[tuple[str, str]] = []
    monkeypatch.setattr(bridge_module, "_LOG_DIR", tmp_path)
    bridge.ros2_stop = lambda name, reason="explicit_stop": stopped.append((name, reason))
    bridge.ros2_proc_status = lambda _name: "stopped"
    bridge._ros2_launch_prereq_error = lambda _name: None
    bridge._render_ros2_launch_cmd = lambda _name: "/venv/python ur5e_rtde_trajectory_server.py"
    bridge._start_tracked_ros2_command = (
        lambda name, command, ros_domain_id=None: started.append(
            (name, command, ros_domain_id)
        )
        or None
    )
    bridge._wait_with_ros2_daemon_retry = lambda *_args, **_kwargs: None

    assert bridge._start_ur5e_rtde_trajectory_server(process_name, ros_domain_id=42) is None

    log_path = tmp_path / f"{process_name}.log"
    assert log_path.parent.is_dir()
    assert started == [
        (
            process_name,
            "PYTHONFAULTHANDLER=1 /venv/python ur5e_rtde_trajectory_server.py "
            f">> {shlex.quote(str(log_path))} 2>&1",
            42,
        )
    ]
    assert set(stopped) == {
        ("ur5e_calibration_rtde_monitor", "ur5e_control_stack_start"),
        ("ur5e_calibration_state_publisher", "ur5e_control_stack_start"),
    }


def test_unexpected_main_rtde_exit_replaces_stale_ready_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process_name = "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server"
    status_path = tmp_path / "cais_ur5e_rtde_trajectory_status.json"
    log_dir = tmp_path / "log"
    status_path.write_text(
        json.dumps(
            {
                "state": "ready",
                "process_id": 3549,
                "ros_domain_id": 42,
                "updated_at": 1.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge_module, "_LOG_DIR", log_dir)
    monkeypatch.setattr(bridge_module, "_UR5E_RTDE_TRAJECTORY_STATUS", status_path)
    bridge = _bridge_with_exited_process(process_name, 134)
    logged: list[str] = []
    monkeypatch.setattr(
        bridge_module.log,
        "error",
        lambda _format, message: logged.append(str(message)),
    )

    assert bridge.ros2_proc_status(process_name) == "stopped"

    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected_log_path = log_dir / f"{process_name}.log"
    expected_message = (
        f"UR5e RTDE trajectory server process {process_name} exited with return code 134. "
        f"Log: {expected_log_path}"
    )
    assert bridge._ros2_procs == {}
    assert status["state"] == "failed"
    assert status["blocked_reason"] == expected_message
    assert status["message"] == expected_message
    assert status["process_name"] == process_name
    assert status["process_return_code"] == 134
    assert status["process_log_path"] == str(expected_log_path)
    assert status["process_id"] == 3549
    assert status["ros_domain_id"] == 42
    assert status["rtde_receive_connected"] is False
    assert status["rtde_control_connected"] is False
    assert status["joint_states_fresh"] is False
    assert status["updated_at"] > 1.0
    assert logged == [expected_message]


def test_intentional_main_rtde_stop_is_not_reclassified_as_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process_name = "hardware_ur5e_rtde_trajectory_server"
    status_path = tmp_path / "cais_ur5e_rtde_trajectory_status.json"
    original_status: dict[str, Any] = {
        "state": "stopped",
        "message": "UR5e RTDE trajectory server stopped",
        "updated_at": 10.0,
    }
    status_path.write_text(json.dumps(original_status), encoding="utf-8")
    monkeypatch.setattr(bridge_module, "_UR5E_RTDE_TRAJECTORY_STATUS", status_path)
    bridge = _bridge_with_exited_process(process_name, -2)
    bridge._ros2_intentional_stops.add(process_name)

    assert bridge.ros2_proc_status(process_name) == "stopped"

    assert json.loads(status_path.read_text(encoding="utf-8")) == original_status


def test_repair_records_rtde_process_that_already_exited_before_status_poll(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process_name = "digital_twin_dual_robots_hardware_ur5e_rtde_trajectory_server"
    status_path = tmp_path / "cais_ur5e_rtde_trajectory_status.json"
    status_path.write_text(json.dumps({"state": "ready", "updated_at": 1.0}), encoding="utf-8")
    monkeypatch.setattr(bridge_module, "_LOG_DIR", tmp_path / "log")
    monkeypatch.setattr(bridge_module, "_UR5E_RTDE_TRAJECTORY_STATUS", status_path)
    bridge = _bridge_with_exited_process(process_name, 139)

    assert bridge.ros2_stop(process_name, reason="digital_twin_repair") is None

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["process_name"] == process_name
    assert status["process_return_code"] == 139
    assert status["process_log_path"].endswith(f"/{process_name}.log")


def test_unexpected_rtde_exit_stops_only_its_ur5e_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process_name = "digital_twin_ur5e_only_hardware_ur5e_rtde_trajectory_server"
    sync_process = "digital_twin_ur5e_only_sync"
    status_path = tmp_path / "cais_ur5e_rtde_trajectory_status.json"
    mirror_status_path = tmp_path / "cais_digital_twin_ur5e_only.json"
    status_path.write_text(json.dumps({"state": "ready", "updated_at": 1.0}), encoding="utf-8")
    monkeypatch.setattr(bridge_module, "_LOG_DIR", tmp_path / "log")
    monkeypatch.setattr(bridge_module, "_UR5E_RTDE_TRAJECTORY_STATUS", status_path)
    bridge = _bridge_with_exited_process(process_name, 139)
    bridge._ros2_procs[sync_process] = SimpleNamespace(pid=3550, poll=lambda: None)
    bridge._digital_twin_status_path = lambda _target: mirror_status_path
    bridge._digital_twin_direction = lambda _target: "hardware -> gazebo"
    stopped: list[tuple[str, str]] = []

    def _stop(name: str, *, reason: str = "explicit_stop") -> None:
        stopped.append((name, reason))
        bridge._ros2_procs.pop(name, None)

    bridge.ros2_stop = _stop

    assert bridge.ros2_proc_status(process_name) == "stopped"

    assert stopped == [(sync_process, "ur5e_rtde_process_exit")]
    mirror_status = json.loads(mirror_status_path.read_text(encoding="utf-8"))
    assert mirror_status["state"] == "failed"
    assert mirror_status["robot"] == "ur5e"
    assert "mirror stopped" in mirror_status["message"]
