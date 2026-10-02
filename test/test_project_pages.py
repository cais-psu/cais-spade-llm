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
from starlette.requests import Request
from starlette.responses import RedirectResponse

from cais_spade_llm.ui import app as ui_app
from cais_spade_llm.ui import recovery_results
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.pages import recovery_framework, recovery_run


def _finish_reads(client) -> None:
    async def finish() -> None:
        while True:
            timers = [item for item in client.elements.values() if isinstance(item, ui.timer)
                      and item.callback and item.callback.__qualname__.startswith("BackgroundSection.load.")
                      and not item._is_canceled]
            if not timers:
                break
            for timer in timers:
                timer.cancel()
                await timer.callback()
    asyncio.run(finish())


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


def test_sidebar_project_folders_expose_existing_views_and_recovery_placeholder():
    """Show both project folders before the shared links, without a main dashboard."""
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
    ] == [
        "journal2-project", "jounral3-project", "/products", "/resources", "/safety",
        "/control", "/perception",
    ]
    for folder, paths in zip(items[:2], (
        [
            "/recovery-framework?tab=run", "/recovery-framework?tab=setup",
            "/recovery-framework?tab=results", "/recovery-framework?tab=recovery",
        ],
        [
            "/spec2primitives?tab=run", "/spec2primitives?tab=setup",
            "/spec2primitives?tab=results",
        ],
    ), strict=True):
        assert folder.value is True
        assert [
            element._props["href"]
            for element in folder.default_slot.children[0].default_slot.children
        ] == paths
    assert [
        element.text for element in client.elements.values() if isinstance(element, ui.label)
    ] == [
        "Navigation", "dashboard", "setup", "results", "recovery", "dashboard", "setup",
        "results", "products", "resources", "safety", "control", "perception",
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


def test_home_redirects_to_journal2_without_rendering_or_starting_work(tmp_path, monkeypatch):
    """The root route redirects without rendering controls or starting the runtime."""
    from cais_spade_llm.spec2primitives.adapters import ui_runtime

    pages = {}

    def register_page(path):
        def register(handler):
            pages[path] = handler
            return handler
        return register

    bridge = MagicMock(spec=SystemBridge)
    wrapper = Mock()
    monkeypatch.setattr(ui_app, "__file__", str(tmp_path / "ui/app.py"))
    monkeypatch.setattr(ui_app, "app", Mock())
    monkeypatch.setattr(ui_app, "_patch_nicegui_lifecycle", Mock())
    monkeypatch.setattr(ui_app, "_page_wrapper", wrapper)
    monkeypatch.setattr(SystemBridge, "instance", lambda: bridge)
    monkeypatch.setattr(ui_runtime, "create_spec2primitives_ui_runtime", Mock())
    monkeypatch.setattr(ui, "page", register_page)
    monkeypatch.setattr(ui, "run", Mock())
    ui_app.create_app()

    response = pages["/"]()

    assert isinstance(response, RedirectResponse)
    assert response.status_code == 307
    assert response.headers["location"] == "/recovery-framework?tab=run"
    wrapper.assert_not_called()
    assert bridge.mock_calls == []


def test_header_home_link_targets_journal2_without_configuration_writes(monkeypatch):
    """The persistent header links home and refreshes status without runtime writes."""
    bridge = _bridge()
    before = vars(bridge).copy()
    callbacks = []
    timers = []

    def timer(interval, callback, **kwargs):
        callbacks.append(callback)
        timers.append(Mock())
        return timers[-1]

    monkeypatch.setattr(ui, "timer", timer)
    client = Client(context.client.page)
    with client:
        ui_app._header(bridge)
    assert len(callbacks) == 2

    async def refresh():
        for callback in callbacks:
            await callback()

    asyncio.run(refresh())
    assert vars(bridge) == before
    assert [
        element._props["href"] for element in client.elements.values()
        if isinstance(element, ui.link)
    ] == ["/recovery-framework?tab=run"]
    assert any(
        isinstance(element, ui.label) and element.text == "Stopped"
        for element in client.elements.values()
    )
    client.delete()
    for timer in timers:
        timer.cancel.assert_called_once_with(with_current_invocation=True)


def test_header_deletion_cancels_actual_nicegui_timers_without_callback_error(monkeypatch):
    """Deleting the header cancels both pending timer invocations cleanly."""
    handle_exception = Mock()
    monkeypatch.setattr(core.app, "handle_exception", handle_exception)
    monkeypatch.setattr(core.app, "on_startup", Mock())
    client = Client(context.client.page)
    with client:
        ui_app._header(_bridge())
    timers = [
        element for element in client.elements.values() if isinstance(element, ui.timer)
    ]
    assert {timer.interval for timer in timers} == {1.0, 2.0}
    invocations = []
    for timer in timers:
        invocation = Mock()
        timer._current_invocation = invocation
        invocations.append(invocation)
    client.delete()
    for timer, invocation in zip(timers, invocations, strict=True):
        assert timer._is_canceled
        invocation.cancel.assert_called_once_with()
    handle_exception.assert_not_called()


@pytest.mark.parametrize(("query", "expected"), [
    (b"", "run"), (b"tab=run", "run"), (b"tab=setup", "setup"),
    (b"tab=results", "results"), (b"tab=recovery", "recovery"),
    (b"tab=", "run"), (b"tab=unknown", "run"),
])
def test_recovery_direct_links_build_only_the_selected_view(query, expected, monkeypatch):
    """Direct links select one view; the recovery viewer invokes no runtime."""
    bridge = MagicMock(spec=SystemBridge)
    render_run, render_setup, render_results, render_delivery = (Mock() for _ in range(4))
    monkeypatch.setattr(recovery_run, "render", render_run)
    monkeypatch.setattr(recovery_framework, "_render_setup", render_setup)
    monkeypatch.setattr(recovery_framework, "render_results", render_results)
    monkeypatch.setattr(recovery_framework, "render_gazebo_delivery_runs", render_delivery)
    request = Request({
        "type": "http", "path": "/recovery-framework", "headers": [], "query_string": query,
    })
    client = Client(context.client.page, request=request)
    with client:
        recovery_framework.render(bridge)
        tabs = next(element for element in client.elements.values() if isinstance(element, ui.tabs))
        assert tabs.value == expected
        assert render_run.call_count == (expected == "run")
        assert render_setup.call_count == (expected == "setup")
        assert render_results.call_count == (expected == "results")
        assert render_delivery.call_count == (2 if expected == "results" else 0)
        if expected == "recovery":
            assert any(
                isinstance(element, ui.select) and element._props.get("label") == "Saved example / test"
                for element in client.elements.values()
            )
            assert not any(isinstance(element, ui.badge) and element.text == "Placeholder"
                           for element in client.elements.values())
        assert bridge.mock_calls == []
    client.delete()


def test_recovery_tabs_default_to_run_and_build_run_once(tmp_path, monkeypatch):
    """Inspecting setup, results, and recovery never remounts the run controls."""
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
            "run", "setup", "results", "recovery",
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
        tabs.value = "recovery"
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
        _finish_reads(client)
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
        _finish_reads(client)
        text = next(element.text for element in client.elements.values()
                    if isinstance(element, ui.label) and element.text.startswith('{"session_id"'))
        assert json.loads(text) == {"session_id": "Session B", "status": "needs_context"}
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
    bridge.get_conveyor_fault.return_value = {"status": "disabled", "ready": False}
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


def test_evidence_previews_are_bounded_exact_and_cache_invalidates(tmp_path):
    from cais_spade_llm.ui.evidence import PREVIEW_BYTES, read_json_cached, read_text_page
    path = tmp_path / 'prompt.txt'
    original = 'Exact prompt: Δ🙂\n' * 12000
    path.write_text(original, encoding='utf-8')
    offset, pages = 0, []
    while offset < path.stat().st_size:
        page, offset, size = read_text_page(path, offset)
        assert len(page.encode('utf-8')) <= PREVIEW_BYTES
        pages.append(page)
    assert ''.join(pages) == original
    record = _write(tmp_path, 'record.json', {'status': 'draft_ready'})
    assert read_json_cached(record)['status'] == 'draft_ready'
    _write(tmp_path, 'record.json', {'status': 'needs_context'})
    assert read_json_cached(record)['status'] == 'needs_context'


def test_recovery_examples_keep_provenance_and_missing_validation(tmp_path):
    from cais_spade_llm.ui.recovery_evidence import list_examples, read_stage_record, resolve_reference, stage_records
    outline = tmp_path / 'recovery_outline/worked_previous'
    request = outline / 'multi_turn_turn03_outline_request_20260802T145141.txt'
    path = _write(outline, 'multi_turn_turn03_outline_audit_20260802T145141.json', {
        'decision': 'need_next_task', 'artifact_paths': {'request_artifact_path': '/old/location/' + request.name},
        'candidate_evaluation_summary': [{'candidate_id': 'candidate_X', 'valid': False,
          'validation_stages': [{'status': 'failed', 'mocked': True, 'validation_category': 'physical_feasibility'}]}],
    })
    request.write_text('Exact recorded prompt Δ')
    record = read_stage_record(outline, path)
    assert record['prompts'] == [('/old/location/' + request.name, request)]
    assert record['validations'][0]['status'] == 'failed'
    assert record['validations'][0]['mocked'] is True
    request.unlink()
    assert read_stage_record(outline, path)['prompts'][0][1] is None
    outside = _write(tmp_path, 'outside.json', {'status': 'passed'})
    assert resolve_reference(outline, str(outside)) is None
    primitive = _write(tmp_path / 'recovery_primitves/1', 'multi_turn_RECOVERY_SEQ1_primitive_generation_turn02_response_20260426T202606.txt', {
        'outline_id': 'RECOVERY_SEQ1', 'decision': 'draft_ready', 'response': {'primitive_steps': []},
    })
    assert read_stage_record(primitive.parent, primitive)['validations'] == []
    safety = _write(tmp_path / 'recovery_safety/worked/5', 'recovery_safety_generation_result.json', {
        'ok': True, 'recovery_safety_status': 'ready', 'rules': [], 'dfa_dot_files': [],
        'all_rule_results': [{'rule_id': 'SAFE_2', 'status': 'not_involved', 'failure_reason': 'unsupported deterministic recovery safety family'}],
    })
    evidence = read_stage_record(safety.parent, safety)
    assert evidence['payload']['rules'] == []
    assert evidence['rules'][0]['status'] == 'not_involved'
    assert evidence['dfa'] == []
    _write(tmp_path / 'primitive_cleanup_acceptance', 'unrelated.json', {})
    assert len(list_examples(tmp_path)) == 3
    assert stage_records(outline, 'recovery_outline') == [path]


def test_large_gazebo_report_is_deferred_and_raw_content_is_bounded(tmp_path):
    from cais_spade_llm.ui.components.gazebo_delivery_run import render_gazebo_delivery_runs
    from cais_spade_llm.ui.evidence import PREVIEW_BYTES
    _write(tmp_path / 'latest', 'run.json', {'outcome': {'status': 'completed'}, 'models': {'detail': 'x' * 2_000_000}, 'transitions': []})
    client = Client(context.client.page)
    with client:
        render_gazebo_delivery_runs(tmp_path, environment=True)
        assert not any(isinstance(item, ui.timer) for item in client.elements.values())
        expansion = next(item for item in client.elements.values() if isinstance(item, ui.expansion))
        expansion.value = True
        _finish_reads(client)
        assert not any(isinstance(item, ui.code) for item in client.elements.values())
        details = next(item for item in client.elements.values() if isinstance(item, ui.expansion) and item.text == 'Recorded setup and observations')
        details.value = True
        _finish_reads(client)
        texts = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
        assert max(len(text.encode()) for text in texts) <= PREVIEW_BYTES
        assert any('of 2000' in text for text in texts)
    client.delete()


def test_background_read_yields_and_discards_stale_or_deleted_updates(monkeypatch):
    from cais_spade_llm.ui.evidence import BackgroundSection
    from threading import Event

    client = Client(context.client.page)
    async def exercise():
        started, release = Event(), Event()
        timers = []
        monkeypatch.setattr(ui, 'timer', lambda interval, callback, **kwargs: timers.append(callback) or Mock())
        shown = []
        with client:
            section = BackgroundSection()
            def slow():
                started.set()
                release.wait(3)
                return 'stale'
            section.load(slow, shown.append)
            pending = asyncio.create_task(timers[-1]())
            while not started.is_set():
                await asyncio.sleep(0.001)
            section.load(lambda: 'current', shown.append)
            await timers[-1]()
            assert shown == ['current']
            release.set()
            await pending
            assert shown == ['current']
            section.load(lambda: 'deleted', shown.append)
            client.delete()
            await timers[-1]()
            assert shown == ['current']
    asyncio.run(exercise())


def test_reused_outline_is_visible_and_grounding_is_not_safety(tmp_path):
    from cais_spade_llm.ui.recovery_evidence import read_stage_record, stage_records
    run = tmp_path / 'test_runs/selected'
    _write(run, 'run.json', {'mode': 'primitive', 'status': 'completed'})
    trace = [{'outline_id': 'RECOVERY_SEQ1', 'event_name': 'recover_to_home', 'resource_jid': 'xarm6@localhost'}]
    checkpoint = _write(run, 'outline_checkpoint.json', {
        'multi_turn_session': {'status': 'ready_for_primitive_generation', 'transition_trace': trace},
        'outline_source': '/source/outline_checkpoint.json',
    })
    assert stage_records(run / 'recovery_outline', 'recovery_outline') == [checkpoint]
    data = read_stage_record(run, checkpoint)
    assert data['trace'] == trace
    assert data['payload']['outline_source'] == '/source/outline_checkpoint.json'
    _write(run, 'multi_turn_turn01_grounding_response_20260416T155255.txt', {'phase': 'grounding'})
    assert stage_records(run, 'recovery_safety') == []


def test_request_only_failure_is_indexed_once_with_exact_attempts(tmp_path):
    from cais_spade_llm.ui.recovery_evidence import list_examples, stage_records, read_stage_record

    directory = tmp_path / "recovery_outline"
    request = _write(directory / "requests/call1", "round00_attempt01_request.json", {
        "kind": "provider_request", "tool_round": 0, "attempt": 1,
        "payload": {"messages": [{"role": "user", "content": "real request"}]},
    })
    examples = list_examples(tmp_path)
    assert len(examples) == 1
    assert examples[0]["path"] == directory
    assert set(examples[0]["stages"]) == {"recovery_outline"}
    assert stage_records(directory, "recovery_outline") == [request]
    assert read_stage_record(directory, request)["captures"][0]["record"]["payload"]["messages"][0]["content"] == "real request"
