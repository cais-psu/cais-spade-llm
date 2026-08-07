"""Focused tests for cleanup of process groups left by a previous UI."""

from __future__ import annotations

import json
from pathlib import Path

from cais_spade_llm.ui.process_registry import UIProcessRegistry


def _write_registry(path: Path, processes: dict) -> None:
    path.write_text(
        json.dumps({"version": 1, "processes": processes}),
        encoding="utf-8",
    )


def test_cleanup_stops_verified_group_from_dead_previous_ui(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "ui_processes.json"
    registry = UIProcessRegistry(path)
    _write_registry(
        path,
        {
            "hardware_ur5e_rtde_trajectory_server": {
                "owner_pid": 111,
                "process_group": 222,
                "command": "ur5e_rtde_trajectory_server.py",
            }
        },
    )
    stopped: list[int] = []
    monkeypatch.setattr(registry, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(registry, "_process_group_alive", lambda _group: True)
    monkeypatch.setattr(registry, "_verified_cais_group", lambda _group: True)
    monkeypatch.setattr(
        registry,
        "_terminate_process_group",
        lambda group: stopped.append(group),
    )

    result = registry.cleanup_previous()

    assert result["stopped"] == ["hardware_ur5e_rtde_trajectory_server"]
    assert stopped == [222]
    assert json.loads(path.read_text(encoding="utf-8"))["processes"] == {}


def test_cleanup_preserves_processes_when_previous_ui_is_still_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "ui_processes.json"
    registry = UIProcessRegistry(path)
    _write_registry(
        path,
        {
            "gazebo_dual": {
                "owner_pid": 111,
                "process_group": 222,
                "command": "ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py",
            }
        },
    )
    monkeypatch.setattr(registry, "_pid_alive", lambda _pid: True)

    result = registry.cleanup_previous()

    assert result["active_owner"] == ["gazebo_dual"]
    assert "gazebo_dual" in json.loads(path.read_text(encoding="utf-8"))["processes"]


def test_cleanup_never_kills_an_unverified_process_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "ui_processes.json"
    registry = UIProcessRegistry(path)
    _write_registry(
        path,
        {
            "stale": {
                "owner_pid": 111,
                "process_group": 222,
                "command": "unknown",
            }
        },
    )
    monkeypatch.setattr(registry, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(registry, "_process_group_alive", lambda _group: True)
    monkeypatch.setattr(registry, "_verified_cais_group", lambda _group: False)

    result = registry.cleanup_previous()

    assert result["discarded"] == ["stale"]
    assert json.loads(path.read_text(encoding="utf-8"))["processes"] == {}


def test_register_and_unregister_are_process_group_specific(tmp_path: Path) -> None:
    path = tmp_path / "ui_processes.json"
    registry = UIProcessRegistry(path)

    registry.register("ur5e_camera", 321, "ros2 launch cais_lab_robotics")
    registry.unregister("ur5e_camera", process_group=999)
    assert "ur5e_camera" in json.loads(path.read_text(encoding="utf-8"))["processes"]

    registry.unregister("ur5e_camera", process_group=321)
    assert json.loads(path.read_text(encoding="utf-8"))["processes"] == {}


def test_exited_launcher_cleanup_stops_surviving_owned_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "ui_processes.json"
    registry = UIProcessRegistry(path)
    registry.register("digital_twin_ur5e_only_sync", 321, "digital_twin_sync.py")
    stopped: list[int] = []
    monkeypatch.setattr(registry, "_process_group_alive", lambda _group: True)
    monkeypatch.setattr(registry, "_verified_cais_group", lambda _group: True)
    monkeypatch.setattr(
        registry,
        "_terminate_process_group",
        lambda group: stopped.append(group),
    )

    error = registry.cleanup_exited_process(
        "digital_twin_ur5e_only_sync",
        process_group=321,
    )

    assert error is None
    assert stopped == [321]
    assert json.loads(path.read_text(encoding="utf-8"))["processes"] == {}


def test_exited_launcher_cleanup_preserves_unverified_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "ui_processes.json"
    registry = UIProcessRegistry(path)
    registry.register("digital_twin_ur5e_only_sync", 321, "digital_twin_sync.py")
    monkeypatch.setattr(registry, "_process_group_alive", lambda _group: True)
    monkeypatch.setattr(registry, "_verified_cais_group", lambda _group: False)

    error = registry.cleanup_exited_process(
        "digital_twin_ur5e_only_sync",
        process_group=321,
    )

    assert error == "refused to stop unverified process group 321"
    assert "digital_twin_ur5e_only_sync" in json.loads(
        path.read_text(encoding="utf-8")
    )["processes"]


def test_app_runs_previous_ui_cleanup_before_watchdogs() -> None:
    app_source = Path("cais_spade_llm/ui/app.py").read_text(encoding="utf-8")
    cleanup = app_source.index("bridge.cleanup_previous_ui_processes")
    watchdog = app_source.index("watchdog_task = asyncio.create_task", cleanup)

    assert cleanup < watchdog
    assert "async def _ros2_process_watchdog()" in app_source
    assert "await asyncio.to_thread(bridge.ros2_all_statuses)" in app_source


def test_ui_main_hard_cleanup_includes_twin_and_camera_workers() -> None:
    source = Path("cais_spade_llm/ui_main.py").read_text(encoding="utf-8")

    for process_marker in (
        "digital_twin_sync.py",
        "physical_part_twin_sync.py",
        "ur5e_rtde_trajectory_server.py",
        "realsense_roboflow_node.py",
        "realsense_preview_node.py",
        "realsense2_camera_node",
    ):
        assert process_marker in source
