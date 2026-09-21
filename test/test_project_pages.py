"""Project navigation and read-only recovery evidence contracts."""

from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from nicegui import context, core, ui
from nicegui.client import Client

from cais_spade_llm.ui import app as ui_app
from cais_spade_llm.ui import recovery_results
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.pages import dashboard, recovery_framework, recovery_run


def _write(root: Path, name: str, payload: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _bridge() -> SimpleNamespace:
    return SimpleNamespace(
        system_running=False,
        execution_mode="simulation",
        selected_product="assembly_board-v1",
        _starting=False,
        _stopping=False,
        selected_product_order_file="order.json",
        selected_safety_file="safety.txt",
        runtime_recovery_mode="pre_ran",
        runtime_recovery_validation_policy="validated",
        runtime_recovery_archive_label="",
        last_error="",
        get_agent_statuses=Mock(return_value=[]),
        get_robot_states=Mock(return_value={"ur5e-1": {"held_part": None}}),
        get_task_states=Mock(return_value={"task_1": "failed"}),
        get_safety_state=Mock(return_value={"blocked_tasks": {"task_1": "unavailable"}}),
        get_execution_timeline=Mock(return_value=[{"task_id": "task_1", "status": "failed"}]),
    )


def test_sidebar_projects_follow_dashboard_with_existing_routes():
    client = Client(context.client.page)
    with client:
        ui_app._sidebar()
    drawer = next(
        element for element in client.elements.values() if isinstance(element, ui.left_drawer)
    )
    items = [
        element
        for element in drawer.default_slot.children
        if isinstance(element, (ui.link, ui.expansion))
    ]
    assert [
        element.text if isinstance(element, ui.expansion) else element._props["href"]
        for element in items
    ] == ["/", "projects", "/products", "/resources", "/safety", "/control", "/perception"]
    projects = items[1]
    assert projects.value is True
    assert [
        element._props["href"]
        for element in projects.default_slot.children[0].default_slot.children
    ] == ["/recovery-framework", "/spec2primitives"]
    assert [
        element.text for element in client.elements.values() if isinstance(element, ui.label)
    ] == [
        "Navigation", "dashboard", "recovery-framework", "spec2primitives", "products",
        "resources", "safety", "control", "perception",
    ]
    client.delete()


def test_recovery_sessions_include_failures_and_keep_revisions_in_one_row(tmp_path):
    payload = {"session_id": "exact Session-A", "turn_index": 1, "status": "need_revision"}
    _write(tmp_path, "multi_turn_turn01_outline_response_20260921T120000.txt", payload)
    payload = {
        "session_id": "exact Session-A",
        "turn_index": 2,
        "status": "ready_for_primitive_generation",
        "primitive_program_complete": False,
        "elapsed_sec": 0,
    }
    _write(tmp_path, "multi_turn_turn02_final_output_response_20260921T120100.txt", payload)
    _write(tmp_path, "multi_turn_turn02_final_output_response_latest.txt", payload)
    _write(
        tmp_path, "multi_turn_session_failed.json", {"session_id": "Failed-B", "status": "failed"}
    )
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}

    rows = recovery_results.read_sessions(tmp_path)

    assert {row["session_id"] for row in rows} == {"exact Session-A", "Failed-B"}
    row = next(row for row in rows if row["session_id"] == "exact Session-A")
    assert len(row["artifacts"]) == 3
    assert row["status"] == "ready_for_primitive_generation"
    assert row["primitive_program_complete"] is False
    assert row["execution_status"] == "not recorded"
    assert row["physical_outcome"] == "not recorded"
    exported = list(csv.DictReader(io.StringIO(recovery_results.results_csv([row]))))[0]
    assert exported["elapsed_sec"] == "0"
    assert exported["primitive_program_complete"] == "False"
    assert exported["source"] == row["source"]
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir()}


def test_recovery_unknown_sessions_deduplicate_only_latest_copies(tmp_path):
    payload = {"decision": "need_revision", "turn_index": 1}
    _write(tmp_path, "multi_turn_turn01_outline_result_20260921T120000.json", payload)
    _write(tmp_path, "multi_turn_turn01_outline_result_latest.json", payload)
    _write(tmp_path, "multi_turn_turn01_outline_result_20260921T130000.json", payload)
    rows = recovery_results.read_sessions(tmp_path)
    assert len(rows) == 2
    assert all(row["session_id"] == "not recorded" for row in rows)
    assert sorted(len(row["artifacts"]) for row in rows) == [1, 2]


def test_recovery_reader_retains_bad_records_and_contains_paths(tmp_path):
    assert recovery_results.read_sessions(tmp_path / "missing") == []
    path = _write(tmp_path, "multi_turn_session_bad.json", [])
    rows = recovery_results.read_sessions(tmp_path)
    assert len(rows) == 1
    assert "Expected a JSON object" in rows[0]["record_error"]
    assert recovery_results.read_artifact(tmp_path, path.name)["raw_text"] == "[]"
    assert "record_error" in recovery_results.read_artifact(tmp_path, "../outside.json")
    assert path.read_text() == "[]"


def test_dashboard_reads_status_without_controls_or_configuration_writes(monkeypatch):
    bridge = _bridge()
    before = vars(bridge).copy()
    timer = Mock()
    callbacks = []
    monkeypatch.setattr(ui, "timer", lambda interval, callback: callbacks.append(callback) or timer)
    client = Client(context.client.page)
    with client:
        dashboard.render(bridge)
    assert len(callbacks) == 1
    callbacks[0]()
    assert vars(bridge) == before
    assert any(
        isinstance(element, ui.table)
        and element.rows == [{"task_id": "task_1", "status": "failed"}]
        for element in client.elements.values()
    )
    client.delete()
    timer.cancel.assert_called_once()


def test_recovery_tabs_default_to_run_and_build_run_once(tmp_path, monkeypatch):
    bridge = _bridge()
    before = vars(bridge).copy()
    calls = []
    polling = []
    monkeypatch.setattr(recovery_framework, "_ROOT", tmp_path)
    monkeypatch.setattr(recovery_framework, "_PAPER", tmp_path / "missing.md")
    monkeypatch.setattr(
        recovery_framework, "render_results", lambda: recovery_results.render_results(tmp_path)
    )

    def render_run(value, *, is_active):
        calls.append(value)
        polling.append(is_active)

    monkeypatch.setattr(recovery_run, "render", render_run)
    client = Client(context.client.page)
    with client:
        recovery_framework.render(bridge)
        tabs = next(element for element in client.elements.values() if isinstance(element, ui.tabs))
        assert tabs.value == "run"
        assert [tab._props["name"] for tab in tabs.default_slot.children] == [
            "run", "setup", "results",
        ]
        assert calls == [bridge]
        assert polling[0]() is True
        tabs.value = "results"
        assert polling[0]() is False
        assert calls == [bridge]
        labels = [
            element._props["label"] for element in client.elements.values()
            if isinstance(element, ui.expansion)
        ]
        assert "Gazebo delivery runs" in labels
        assert "Offline nominal runs" not in labels
        assert not any(
            isinstance(element, ui.link) and element.text in {
                "Offline product plans and history", "Offline resource transitions",
            }
            for element in client.elements.values()
        )
        assert vars(bridge) == before
        tabs.value = "run"
        assert polling[0]() is True
        tabs.value = "setup"
        assert polling[0]() is False
        tabs.value = "run"
        assert calls == [bridge]
        assert vars(bridge) == before
        assert list(tmp_path.iterdir()) == []
    client.delete()


def test_recovery_results_search_export_and_artifact_inspection(tmp_path, monkeypatch):
    _write(tmp_path, "multi_turn_session_A.json", {"session_id": "Session A", "status": "failed"})
    _write(
        tmp_path,
        "multi_turn_session_B.json",
        {"session_id": "Session B", "status": "needs_context"},
    )
    callbacks = {}
    original_click = ui.button.on_click
    original_select = ui.table.on_select

    def capture_click(button, callback):
        callbacks[button.text] = callback
        return original_click(button, callback)

    def capture_select(table, callback):
        callbacks["select"] = callback
        return original_select(table, callback)

    download = Mock()
    monkeypatch.setattr(ui.button, "on_click", capture_click)
    monkeypatch.setattr(ui.table, "on_select", capture_select)
    monkeypatch.setattr(ui.download, "content", download)
    client = Client(context.client.page)
    with client:
        recovery_results.render_results(tmp_path)
        search = next(
            element for element in client.elements.values() if isinstance(element, ui.input)
        )
        table = next(
            element for element in client.elements.values() if isinstance(element, ui.table)
        )
        search.value = "Session B"
        assert [row["session_id"] for row in table.rows] == ["Session B"]
        callbacks["Export CSV"]()
        exported = list(csv.DictReader(io.StringIO(download.call_args.args[0])))
        assert [row["session_id"] for row in exported] == ["Session B"]
        table.selected = [table.rows[0]]
        callbacks["select"]()
        code = next(element for element in client.elements.values() if isinstance(element, ui.code))
        assert json.loads(code.content) == {"session_id": "Session B", "status": "needs_context"}
    client.delete()


def test_recovery_gazebo_startup_callback_remains_explicit(monkeypatch):
    bridge = SimpleNamespace(
        system_running=False,
        _starting=False,
        simulation_environment_running=lambda: False,
        passive_digital_twin_environment_running=lambda: False,
    )
    launch = Mock()
    callbacks = {}
    original = ui.button.on_click

    def capture(button, callback):
        callbacks[button.text] = callback
        return original(button, callback)

    monkeypatch.setattr(ui.button, "on_click", capture)
    client = Client(context.client.page)
    with client:
        banner = ui.column()
        ready = recovery_run._check_prerequisites(
            bridge, "simulation", banner, launch_simulation=launch
        )
        assert ready is False
        launch.assert_not_called()
        callbacks["start simulation"]()
        launch.assert_called_once_with()
    client.delete()


@pytest.mark.parametrize("mode", ["dry_run", "Dry Run", "", "unsupported"])
@pytest.mark.parametrize("running", [False, True])
def test_recovery_rejects_unsupported_modes_before_runtime_checks(mode, running):
    bridge = MagicMock(spec=SystemBridge)
    bridge.system_running = running
    bridge._starting = False
    client = Client(context.client.page)
    with client:
        banner = ui.column()
        assert recovery_run._check_prerequisites(bridge, mode, banner) is False
        assert any(
            isinstance(element, ui.label) and element.text == "Select Simulation or Physical mode."
            for element in banner.default_slot.children
        )
        assert bridge.mock_calls == []
    client.delete()


def test_recovery_run_polling_does_not_write_mode_and_pauses_when_hidden(monkeypatch):
    bridge = MagicMock(spec=SystemBridge)
    for name, value in {
        "system_running": False,
        "_starting": False,
        "_stopping": False,
        "execution_mode": "dry_run",
        "robot_env": "original",
        "last_error": "",
        "selected_product": "",
        "selected_product_order_file": "",
        "selected_safety_file": "",
        "cca": None,
    }.items():
        setattr(bridge, name, value)
    for name in (
        "list_product_files",
        "list_product_order_files",
        "list_safety_requirement_files",
        "list_runtime_recovery_archives",
        "get_plan_nodes",
        "get_current_task_dag_nodes",
        "get_plan_safety_alerts",
        "get_runtime_recoveries",
        "get_agent_statuses",
        "get_execution_timeline",
        "get_safety_rules",
    ):
        getattr(bridge, name).return_value = []
    for name in ("get_robot_states", "get_task_states", "get_safety_state"):
        getattr(bridge, name).return_value = {}
    bridge.get_runtime_recovery_settings.return_value = {
        "mode": "manual",
        "validation_policy": "validated",
    }
    bridge.ros2_proc_status.return_value = "stopped"
    bridge.ros2_start.return_value = None
    bridge.simulation_environment_running.return_value = False
    bridge.passive_digital_twin_environment_running.return_value = False
    callbacks = []
    buttons = {}
    original_click = ui.button.on_click

    def capture_click(button, callback):
        buttons[button.text] = callback
        return original_click(button, callback)

    monkeypatch.setattr(ui.button, "on_click", capture_click)
    active = {"value": True}
    monkeypatch.setattr(
        ui, "timer", lambda interval, callback, **kwargs: callbacks.append(callback) or Mock()
    )
    client = Client(context.client.page)
    async def check_page():
        with client:
            recovery_run.render(bridge, is_active=lambda: active["value"])
            for callback in callbacks:
                await callback()
            assert bridge.execution_mode == "dry_run"
            assert bridge.robot_env == "original"
            assert any(
                isinstance(element, ui.table)
                and {"setting": "Mode", "value": "Simulation"} in element.rows
                for element in client.elements.values()
            )
            assert not any(
                isinstance(element, ui.select) and element._props.get("label") in {
                    "Mode", "Product", "Product Order JSON", "Safety Requirement (.txt)",
                    "Recovery Handoff Mode", "Recovery Safety Mode", "Archived Recovery Run",
                }
                for element in client.elements.values()
            )
            assert not any(
                "dry run" in str(getattr(element, "text", "")).lower()
                for element in client.elements.values()
            )
            bridge.ros2_start.assert_not_called()
            bridge.start_system.assert_not_called()
            bridge.run_order_dry_run.assert_not_called()
            await buttons["Start Simulation"]()
            bridge.ros2_start.assert_called_once_with("gazebo_dual")
            bridge.start_system.assert_not_called()
            assert bridge.execution_mode == "dry_run"
            assert bridge.robot_env == "original"
            bridge.execution_mode, bridge.robot_env = "physical", "real"
            for callback in callbacks:
                await callback()
            assert bridge.execution_mode == "physical"
            assert bridge.robot_env == "real"
            active["value"] = False
            bridge.reset_mock()
            for callback in callbacks:
                await callback()
            assert bridge.mock_calls == []
    asyncio.run(check_page())
    client.delete()


def test_page_refresh_pauses_for_visibility_and_disconnect_and_cancels_on_delete(monkeypatch):
    from cais_spade_llm.ui.refresh import PageRefresh
    from starlette.requests import Request

    callbacks = []
    timer = Mock()
    monkeypatch.setattr(ui, "timer", lambda interval, callback, **kwargs: callbacks.append(callback) or timer)
    client = Client(context.client.page, request=Request({
        "type": "http", "path": "/", "headers": [], "query_string": b"",
        "scheme": "http", "server": ("testserver", 80),
    }))
    reads = Mock()

    async def scenario():
        monkeypatch.setattr(core, "loop", asyncio.get_running_loop())
        with client:
            polling = PageRefresh()
            polling.timer(1.0, reads)
            client.handle_handshake("socket", "document", None)
            await callbacks[0]()
            assert reads.call_count == 1
            client._cais_visible = False
            await callbacks[0]()
            assert reads.call_count == 1
            client._cais_visible = True
            client.handle_disconnect("socket")
            await callbacks[0]()
            assert reads.call_count == 1
            client.handle_handshake("socket", "document", None)
            await callbacks[0]()
            assert reads.call_count == 2
        client.delete()
        await callbacks[0]()
        assert reads.call_count == 2
        timer.cancel.assert_called_once_with(with_current_invocation=True)

    asyncio.run(scenario())
