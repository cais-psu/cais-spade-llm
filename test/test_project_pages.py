from __future__ import annotations

"""Project navigation and read-only recovery evidence contracts."""

import asyncio
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

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


def test_outline_draft_ui_shows_table_without_enabling_composition_or_execution():
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock(system_running=False)
    example = PartSlippageExample(bridge)
    example.source = "offline"
    state = example.snapshot()
    state.update(status="draft", message="Unvalidated draft", can_generate_outline_only=True, can_load_draft=True,
        outline_draft={"message": "Unvalidated draft", "provider_calls": 1, "transition_trace": [{
            "event_name": "Arbitrary task event", "resource_jid": "owner", "part_name": "KET4_Square_4mm",
            "expected_end_state": {"active_job": None}, "rationale": "Supplied UI test"}]})
    bridge.get_part_slippage_example.return_value = state
    bridge.list_recovery_observations.return_value = []
    bridge.list_safety_requirement_files.return_value = []
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    assert not any(isinstance(item, ui.button) and item.text in {"Generate outline only", "Load draft", "Start fresh"}
                   for item in client.elements.values())
    for item in client.elements.values():
        if isinstance(item, ui.button) and item.text in {"Continue", "Execute recovery", "Test in Gazebo"}:
            assert not item.enabled
    table = next(item for item in client.elements.values() if isinstance(item, ui.table)
                 and item.rows and item.rows[0].get("event_name") == "Arbitrary task event")
    assert table.rows[0]["end_state"] == "owner  active_job = none"
    assert table.visible
    assert any(isinstance(item, ui.label) and "Execution authorized: false" in item.text
               for item in client.elements.values())
    bridge.part_slippage_action.assert_not_called()
    client.delete()


def test_composition_review_shows_authored_resolved_effects_and_lazy_evidence(monkeypatch):
    from cais_spade_llm.ui.components.recovery_turns import render_composition_programs

    monkeypatch.setattr(ui, "run_javascript", lambda *_: None)

    evidence = {"feasibility_status": "FEASIBLE", "evidence_fingerprint": "bound",
        "trace": [{"resolved_params": {"count": 3}}], "derived_end_state": {"quality": "checked"},
        "findings": [], "reason": "All declared postconditions established", "private_detail": "EXPAND_ONLY"}
    program = [{"outline_id": "event", "resource_jid": "unfamiliar_owner",
        "primitive_steps": [{"primitive": "qualify", "params": {"count": {"context_ref": "/limit"}}}],
        "composition_validation": evidence}]
    client = Client(context.client.page)
    with client:
        render_composition_programs(program)
    table = next(item for item in client.elements.values() if isinstance(item, ui.table))
    assert json.loads(table.rows[0]["authored"])["count"] == {"context_ref": "/limit"}
    assert json.loads(table.rows[0]["resolved"])["count"] == 3
    labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
    assert "Achieved postconditions" in labels and "RA validation findings" in labels
    assert not any("EXPAND_ONLY" in text for text in labels)
    with client:
        next(item for item in client.elements.values() if isinstance(item, ui.expansion)
             and item.text == "Complete RA evidence").value = True
    _finish_reads(client)
    assert any(isinstance(item, ui.label) and "EXPAND_ONLY" in item.text for item in client.elements.values())
    client.delete()


def _write(root: Path, name: str, payload: object) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _nominal_proof_ui_case(directory: Path) -> tuple[Path, dict, dict]:
    import hashlib

    proof = {"complete": False, "status": "NEEDS_CONTEXT", "reason": "Recorded missing fact",
             "execution_authorized": False, "tasks": [{"task_id": "environment_8", "validation_stages": []}]}
    path = _write(directory, "turns/turn_00004_validation_0001.json", {
        "turn_index": 4, "phase": "outline", "decision": "need_context",
        "nominal_continuation_evidence": proof,
        "nominal_continuation_artifact_version": 1, "nominal_continuation_context_binding": "a" * 64,
    })
    summary = {key: proof[key] for key in ("complete", "status", "reason", "execution_authorized")}
    summary["artifact_reference"] = {"version": 1, "path": "turns/turn_00004_validation_0001.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "turn_index": 4, "phase": "outline",
        "context_binding": "a" * 64}
    return path, summary, proof


def test_nominal_proof_expanded_turn_loads_once_and_keeps_original_records(tmp_path, monkeypatch):
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import recovery_artifacts
    from cais_spade_llm.ui import recovery_evidence

    directory = tmp_path / "recovery_outline"
    full, summary, proof = _nominal_proof_ui_case(directory)
    first = _write(directory, "turns/turn_00004_outline.json", {
        "turn_index": 4, "phase": "outline", "nominal_continuation_evidence": summary})
    second = _write(directory, "multi_turn_turn04_outline_audit.json", {
        "turn_index": 4, "phase": "outline", "validation_evidence": {"nominal_continuation_evidence": summary}})
    original = {path: path.read_bytes() for path in (full, first, second)}
    load = recovery_artifacts.json.load
    reads = []

    def count(stream, *args, **kwargs):
        reads.append(Path(stream.name))
        return load(stream, *args, **kwargs)

    monkeypatch.setattr(recovery_artifacts.json, "load", count)
    indexed = recovery_evidence.recovery_turns(directory, "recovery_outline")
    assert reads == []
    data = recovery_evidence.read_recovery_turn(directory, {"files": [str(full), str(first), str(second)]})
    assert reads == [full]
    displayed = [row["nominal_continuation_artifact"] for row in data["results"]]
    assert all(row is displayed[0] for row in displayed)
    assert displayed[0]["status"] == "verified" and displayed[0]["proof"] == proof
    assert all("tasks" not in row["payload"].get("nominal_continuation_evidence", {}) for row in data["results"])
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert indexed


@pytest.mark.parametrize("failure", ["missing", "tampered", "escape", "conflicting_summary", "null_reference"])
def test_nominal_proof_ui_reports_unavailable_instead_of_trusting_summary(tmp_path, failure):
    from cais_spade_llm.ui.recovery_evidence import read_recovery_turn

    directory = tmp_path / "recovery_outline"
    path, summary, _ = _nominal_proof_ui_case(directory)
    if failure == "missing":
        path.unlink()
    elif failure == "tampered":
        path.write_text(path.read_text().replace("Recorded missing fact", "Changed evidence"))
    elif failure == "escape":
        summary["artifact_reference"]["path"] = "../outside.json"
    elif failure == "conflicting_summary":
        summary["complete"] = True
    else:
        summary["artifact_reference"] = None
    saved = _write(directory, "turns/turn_00004_outline.json", {"nominal_continuation_evidence": summary})
    data = read_recovery_turn(directory, {"files": [str(saved)]})
    display = data["results"][0]["nominal_continuation_artifact"]
    assert display["status"] == "unavailable" and display["reason"]
    assert "proof" not in display


def test_nominal_proof_ui_reads_copied_generation_without_original_root(tmp_path):
    import shutil
    from cais_spade_llm.ui.recovery_evidence import read_stage_record

    directory = tmp_path / "generation_0001/recovery_outline"
    _, summary, proof = _nominal_proof_ui_case(directory)
    _write(directory, "turns/turn_00004_outline.json", {"nominal_continuation_evidence": summary})
    copied = tmp_path / "generation_0002/recovery_outline"
    shutil.copytree(directory, copied)
    shutil.rmtree(directory)
    data = read_stage_record(copied, copied / "turns/turn_00004_outline.json")
    assert data["nominal_continuation_artifact"]["proof"] == proof
    assert Path(data["nominal_continuation_artifact"]["path"]).is_relative_to(copied)


def test_nominal_proof_renders_once_with_download_and_remains_collapsed(tmp_path, monkeypatch, recovery_browser):
    from cais_spade_llm.ui import recovery_evidence
    from cais_spade_llm.ui.components import recovery_turns

    directory = tmp_path / "recovery_outline"
    full, summary, proof = _nominal_proof_ui_case(directory)
    saved = _write(directory, "turns/turn_00004_outline.json", {"nominal_continuation_evidence": summary})
    data = recovery_evidence.read_recovery_turn(directory, {"files": [str(full), str(saved)]})
    client = Client(context.client.page)
    downloads = []
    monkeypatch.setattr(recovery_turns, "_download", lambda path, label: downloads.append((path, label)))
    with client:
        recovery_turns.render_turn_content(data, "output")
    sections = [row for row in client.elements.values() if isinstance(row, ui.expansion)
                and row.text == "Nominal continuation proof"]
    assert len(sections) == 1 and not sections[0].value
    assert sum(label == "Download full nominal continuation proof" for _, label in downloads) == 1
    assert not any(isinstance(row, ui.label) and row.text == json.dumps(proof, indent=2)
                   for row in client.elements.values())
    with client:
        sections[0].value = True
    _finish_reads(client)
    assert sum(isinstance(row, ui.label) and row.text == json.dumps(proof, indent=2)
               for row in client.elements.values()) == 1
    client.delete()


def test_nominal_proof_ui_renders_missing_error_without_download(tmp_path, monkeypatch):
    from cais_spade_llm.ui import recovery_evidence
    from cais_spade_llm.ui.components import recovery_turns

    directory = tmp_path / "recovery_outline"
    full, summary, _ = _nominal_proof_ui_case(directory)
    full.unlink()
    saved = _write(directory, "turns/turn_00004_outline.json", {"nominal_continuation_evidence": summary})
    data = recovery_evidence.read_recovery_turn(directory, {"files": [str(saved)]})
    downloads = []
    monkeypatch.setattr(recovery_turns, "_download", lambda path, label: downloads.append((path, label)))
    client = Client(context.client.page)
    with client:
        recovery_turns.render_turn_content(data, "output")
    assert any(isinstance(row, ui.label) and row.text.startswith("Nominal continuation proof unavailable:")
               for row in client.elements.values())
    assert not any(label == "Download full nominal continuation proof" for _, label in downloads)
    client.delete()


def test_tampered_nominal_proof_cannot_reappear_through_its_indexed_file(tmp_path):
    from cais_spade_llm.ui.recovery_evidence import read_recovery_turn

    directory = tmp_path / "recovery_outline"
    full, summary, _ = _nominal_proof_ui_case(directory)
    full.write_text(full.read_text().replace("Recorded missing fact", "Changed evidence"))
    saved = _write(directory, "turns/turn_00004_outline.json", {"nominal_continuation_evidence": summary})
    data = read_recovery_turn(directory, {"files": [str(full), str(saved)]})
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert result["nominal_continuation_artifact"]["status"] == "unavailable"
        assert "proof" not in result["nominal_continuation_artifact"]
        assert result["payload"]["nominal_continuation_evidence"]["status"] == "unavailable"


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


def test_manual_recovery_shows_observations_before_a_request_is_sent(tmp_path):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    directory = tmp_path / "generation_0001"
    _write(directory, "input_preview.json", {"request_sent": False, "prompt": "Observed failure"})
    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state.update(status="outline_ready", outline_status="ready", outline_mode="manual", primitive_mode="manual",
                 artifact_directory=str(directory), outline_artifact_directory=str(directory / "recovery_outline"))
    bridge.get_part_slippage_example.return_value = state
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    labels = [element.text for element in client.elements.values() if isinstance(element, ui.expansion)]
    assert "Prepared input — not sent" in labels
    bridge.start_part_slippage_example.assert_not_called()
    bridge.part_slippage_action.assert_not_called()
    client.delete()


@pytest.mark.parametrize("source", ["gazebo", "offline"])
def test_one_recovery_mode_controls_both_stages_and_restores_without_starting_work(source):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    example = PartSlippageExample(bridge)
    example.source = source
    bridge.get_part_slippage_example.side_effect = example.snapshot
    bridge.set_part_slippage_modes.side_effect = example.set_modes
    bridge.list_safety_requirement_files.return_value = example.safety_files
    bridge.list_recovery_observations.return_value = []
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
        modes = [element for element in client.elements.values()
                 if isinstance(element, ui.toggle) and element.options == {"auto": "Auto", "manual": "Manual"}]
        assert len(modes) == 1 and modes[0].value == ("manual" if source == "offline" else "auto")
        bridge.set_part_slippage_modes.assert_not_called()
        if source == "offline":
            assert not modes[0].enabled
        else:
            assert modes[0].enabled
            modes[0].value = "manual"
            bridge.set_part_slippage_modes.assert_called_once_with("manual", "manual")
            assert example.snapshot()["outline_mode"] == example.snapshot()["primitive_mode"] == "manual"
            modes[0].value = "auto"
            bridge.set_part_slippage_modes.assert_called_with("auto", "auto")
            assert example.snapshot()["outline_mode"] == example.snapshot()["primitive_mode"] == "auto"
            modes[0].value = "manual"
    client.delete()
    bridge.set_part_slippage_modes.reset_mock()
    restored = Client(context.client.page)
    with restored:
        render_part_slippage(bridge)
        mode = next(element for element in restored.elements.values()
                    if isinstance(element, ui.toggle) and element.options == {"auto": "Auto", "manual": "Manual"})
        assert mode.value == "manual"
    bridge.set_part_slippage_modes.assert_not_called()
    bridge.start_failure_example.assert_not_called()
    bridge.start_offline_recovery.assert_not_called()
    bridge.part_slippage_action.assert_not_called()
    restored.delete()


def test_shared_mode_does_not_change_an_existing_run_with_different_stage_modes():
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state.update(status="outline_ready", outline_mode="manual", primitive_mode="auto")
    bridge.get_part_slippage_example.return_value = state
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    modes = [element for element in client.elements.values()
             if isinstance(element, ui.toggle) and element.options == {"auto": "Auto", "manual": "Manual"}]
    assert len(modes) == 1 and modes[0].value is None and not modes[0].enabled
    assert any(isinstance(element, ui.label) and element.text ==
               "Recorded modes: Recovery outline — manual; Primitive composition — auto."
               for element in client.elements.values())
    bridge.set_part_slippage_modes.assert_not_called()
    client.delete()


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_offline_generate_visible_in_both_modes_and_reconnection_does_not_start_work(mode):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state.update(source="offline", status="outline_ready", can_generate_outline=True, outline_generate_reason="",
                 outline_mode=mode, primitive_mode=mode)
    bridge.get_part_slippage_example.return_value = state
    for _ in range(2):
        client = Client(context.client.page)
        with client:
            render_part_slippage(bridge)
        buttons = [element for element in client.elements.values() if isinstance(element, ui.button)]
        assert "Start offline test" not in [element.text for element in buttons]
        outline_generate = next(element for element in buttons if element.text == "Generate")
        assert outline_generate.visible and outline_generate.enabled
        assert all(not element.visible for element in buttons if element.text == "Continue")
        toggle = next(element for element in client.elements.values()
                      if isinstance(element, ui.toggle) and element.options == {"auto": "Auto", "manual": "Manual"})
        assert not toggle.enabled and toggle.value == "manual"
        client.delete()
    bridge.prepare_offline_recovery.assert_not_called()
    bridge.start_offline_recovery.assert_not_called()
    bridge.part_slippage_action.assert_not_called()


@pytest.mark.parametrize("status,reason", [
    ("idle", "Select saved observations to prepare the recovery input."),
    ("preparing_outline", "Preparing input from saved observations."),
    ("blocked", "NEEDS_CONTEXT: prepared_recovery_request"),
    ("generating_outline", "Generation already running."),
])
def test_offline_disabled_generate_shows_the_actual_reason(status, reason):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state.update(source="offline", status=status, can_generate_outline=False, outline_generate_reason=reason,
                 busy=status in {"preparing_outline", "generating_outline"})
    bridge.get_part_slippage_example.return_value = state
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    generate = next(element for element in client.elements.values() if isinstance(element, ui.button) and element.text == "Generate")
    assert generate.visible and not generate.enabled
    assert any(isinstance(element, ui.label) and element.text == reason for element in client.elements.values())
    client.delete()


@pytest.mark.parametrize("stage", ["outline", "primitive"])
def test_stage_stop_is_enabled_only_for_the_running_stage(stage, monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state.update(status="generating_outline" if stage == "outline" else "generating_primitives",
                 busy=True, generating_stage=stage)
    bridge.get_part_slippage_example.return_value = state
    bridge.list_safety_requirement_files.return_value = []
    bridge.part_slippage_action = AsyncMock(return_value=state)
    callbacks = {}
    original_click = ui.button.on_click

    def capture_click(button, callback):
        callbacks[button.id] = callback
        return original_click(button, callback)

    monkeypatch.setattr(ui.button, "on_click", capture_click)
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    example_stop, outline_stop, primitive_stop = [item for item in client.elements.values()
                                                  if isinstance(item, ui.button) and item.text == "Stop"]
    assert example_stop.enabled
    running, waiting = (outline_stop, primitive_stop) if stage == "outline" else (primitive_stop, outline_stop)
    assert running.visible and running.enabled
    assert waiting.visible and not waiting.enabled

    async def exercise():
        with client:
            await callbacks[running.id]()

    asyncio.run(exercise())
    bridge.part_slippage_action.assert_awaited_once_with("stop_generation", feedback="")
    client.delete()


def test_selecting_saved_observations_only_defers_the_input():
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state["source"] = "offline"
    bridge.get_part_slippage_example.return_value = state
    bridge.set_failure_example_options.return_value = {**state, "selected_observation_path": "capture"}
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
        selected = next(element for element in client.elements.values()
                        if isinstance(element, ui.select) and element._props.get("label") == "Saved observations")
        selected.set_options({"capture": "Part slippage · source · captured time · Ready to prepare"}, value="capture")
    bridge.prepare_offline_recovery.assert_not_called()
    assert bridge.set_failure_example_options.call_args.kwargs["defer_preparation"] is True
    bridge.start_offline_recovery.assert_not_called()
    bridge.part_slippage_action.assert_not_called()
    client.delete()


def test_saved_result_stays_loadable_when_new_outline_preparation_is_blocked(monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state.update(source="offline", status="blocked", selected_observation_path="capture",
                 message="NEEDS_CONTEXT: owner descriptor missing")
    bridge.get_part_slippage_example.return_value = state
    bridge.list_recovery_observations.return_value = [{
        "source_path": "capture", "scenario": "Part slippage", "run_id": "source",
        "result_available": True, "offline_ready": True}]
    bridge.list_safety_requirement_files.return_value = []
    callbacks = {}
    original_click = ui.button.on_click

    def capture_click(button, callback):
        callbacks[button.text] = callback
        return original_click(button, callback)

    monkeypatch.setattr(ui.button, "on_click", capture_click)
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    load = next(item for item in client.elements.values() if isinstance(item, ui.button)
                and item.text == "Load saved result")
    assert load.visible and load.enabled
    bridge.load_offline_recovery.assert_not_called()
    bridge.prepare_offline_recovery.assert_not_called()

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()

        async def read_in_background(reader):
            assert reader is bridge.load_offline_recovery
            started.set()
            await release.wait()
            return reader()

        monkeypatch.setattr(asyncio, "to_thread", read_in_background)
        with client:
            operation = asyncio.create_task(callbacks["Load saved result"]())
            await started.wait()
            assert not load.enabled
            assert any(isinstance(item, ui.label) and item.text.startswith("Loading saved result")
                       for item in client.elements.values())
            await callbacks["Load saved result"]()
            release.set()
            await operation
        assert load.enabled
        bridge.load_offline_recovery.assert_called_once_with()

    asyncio.run(exercise())
    client.delete()


def test_saved_observation_selection_survives_an_older_poll(monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state["validation_checks_selected"] = True
    bridge.get_default_recovery_observations.return_value = {"Part slippage": "capture"}
    bridge.get_part_slippage_example.side_effect = lambda: dict(state)
    bridge.list_recovery_observations.return_value = [{
        "source_path": "capture", "scenario": "Part slippage", "run_id": "source",
        "captured_at": "2026-10-08T20:46:46+00:00", "offline_ready": True,
    }]
    bridge.list_safety_requirement_files.return_value = []

    def select(**options):
        state.update(source=options["source"], selected_observation_path=options["observation_path"],
                     can_generate_outline=bool(options["observation_path"]), defer_preparation=True,
                     selection_revision=state.get("selection_revision", 0) + 1)
        return dict(state)

    def prepare():
        state.update(status="outline_ready", can_generate_outline=True, outline_generate_reason="",
                     observation_path="attempt/observations.json")

    bridge.set_failure_example_options.side_effect = select
    bridge.prepare_offline_recovery.side_effect = prepare
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()

        async def read_files(reader):
            started.set()
            await release.wait()
            return reader()

        monkeypatch.setattr(asyncio, "to_thread", read_files)
        with client:
            source = next(item for item in client.elements.values()
                          if isinstance(item, ui.toggle) and "offline" in item.options)
            selected = next(item for item in client.elements.values()
                            if isinstance(item, ui.select) and item._props.get("label") == "Saved observations")
            timer = next(item for item in client.elements.values() if isinstance(item, ui.timer)
                         and item.callback.__qualname__.startswith("PageRefresh.timer."))
            pending = asyncio.create_task(timer.callback())
            await started.wait()
            source.value = "offline"
            selected.value = "capture"
            release.set()
            await pending
            assert source.value == "offline"
            assert selected.visible and selected.value == "capture"
            assert next(item for item in client.elements.values()
                        if isinstance(item, ui.button) and item.text == "Generate").enabled

    asyncio.run(exercise())
    bridge.prepare_offline_recovery.assert_not_called()
    bridge.part_slippage_action.assert_not_called()
    client.delete()


def test_saved_observation_refresh_restores_current_selection_without_preparing(monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components import part_slippage_recovery

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    state["source"] = "offline"
    state["selected_observation_path"] = "capture"
    state["defer_preparation"] = True
    bridge.get_default_recovery_observations.return_value = {"Part slippage": "capture"}
    bridge.get_part_slippage_example.side_effect = lambda: dict(state)
    reader = Mock()
    monkeypatch.setattr(part_slippage_recovery, "BackgroundSection", lambda: reader)
    captures = [{"source_path": "capture", "scenario": "Part slippage", "run_id": "source",
                 "related_observation_paths": ["capture", "attempt/observations.json"],
                 "offline_ready": True}]
    client = Client(context.client.page)
    with client:
        part_slippage_recovery.render_part_slippage(bridge)
        selected = next(item for item in client.elements.values()
                        if isinstance(item, ui.select) and item._props.get("label") == "Saved observations")
        show_options = reader.load.call_args.args[1]
        show_options([])
        assert selected.value == "capture"
        assert len(selected.options) == 1
        unchanged = Mock(wraps=selected.set_options)
        monkeypatch.setattr(selected, "set_options", unchanged)
        show_options([])
        unchanged.assert_not_called()
    bridge.set_failure_example_options.assert_not_called()
    bridge.prepare_offline_recovery.assert_not_called()
    client.delete()


def test_rapid_outline_checks_survive_pending_poll_without_preparing(monkeypatch):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage
    from cais_spade_llm.recovery_framework.outline_checks import CHECK_FIELDS

    bridge = MagicMock(system_running=False)
    example = PartSlippageExample(SimpleNamespace(system_running=False))
    bridge.get_part_slippage_example.side_effect = example.snapshot
    bridge.set_failure_example_options.side_effect = example.set_options
    bridge.get_default_recovery_observations.return_value = {"Part slippage": "capture"}
    bridge.list_safety_requirement_files.return_value = []
    bridge.list_recovery_observations.side_effect = AssertionError("A preference change scanned history")
    bridge.prepare_offline_recovery.side_effect = AssertionError("A preference change prepared an attempt")
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)

    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(reader):
            entered.set()
            await release.wait()
            return reader()

        monkeypatch.setattr(asyncio, "to_thread", delayed)
        with client:
            checks = [item for item in client.elements.values() if isinstance(item, ui.checkbox)]
            timer = next(item for item in client.elements.values() if isinstance(item, ui.timer)
                         and item.callback.__qualname__.startswith("PageRefresh.timer."))
            pending = asyncio.create_task(timer.callback())
            await entered.wait()
            for check in checks:
                check.value = True
            assert all(check.value is True for check in checks)
            release.set()
            await pending
            assert all(check.value is True and check.enabled for check in checks)
            assert example.validation_checks == dict.fromkeys(CHECK_FIELDS, True)
            assert example._offline is None and example._draft is None
            assert not example.snapshot().get("run_id")
    asyncio.run(exercise())
    bridge.list_recovery_observations.assert_not_called()
    bridge.prepare_offline_recovery.assert_not_called()
    bridge.part_slippage_action.assert_not_called()
    client.delete()


def test_recovery_page_orders_main_panels_and_restores_safety_selection(tmp_path):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.pages import recovery

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    safety = "cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt"
    state["selected_safety_files"] = [safety]
    bridge.get_part_slippage_example.return_value = state
    bridge.list_safety_requirement_files.return_value = [safety]
    bridge.list_recovery_observations.return_value = []
    client = Client(context.client.page)
    with client:
        recovery.render(bridge=bridge, root=tmp_path, debug_root=tmp_path)
    _finish_reads(client)
    names = ["Example", "Safety", "Recovery outline", "Primitive composition", "Execution"]
    labels = [element.text for element in client.elements.values() if isinstance(element, ui.label)]
    assert "Current observations" not in labels
    assert [labels.index(name) for name in names] == sorted(labels.index(name) for name in names)
    assert not any(isinstance(element, ui.expansion) and element.text == "Details" for element in client.elements.values())
    assert any(isinstance(element, ui.button) and element.text == "Start failure" for element in client.elements.values())
    selected = next(element for element in client.elements.values()
                    if isinstance(element, ui.select) and element._props.get("label") == "Safety specifications")
    assert selected.value == [safety]
    bridge.set_failure_example_options.assert_not_called()
    bridge.start_failure_example.assert_not_called()
    bridge.start_offline_recovery.assert_not_called()
    client.delete()


def test_selecting_part_slippage_updates_safety_defaults_without_starting_work():
    from cais_spade_llm.recovery_framework.recovery_case_safety import default_safety_files
    from cais_spade_llm.recovery_framework import ROOT
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock(system_running=False)
    example = PartSlippageExample(bridge)
    expected = example.snapshot()["selected_safety_files"]
    example.set_options(source="gazebo", scenario="Conveyor breakdown", safety_files=[])
    bridge.get_part_slippage_example.side_effect = example.snapshot
    bridge.set_failure_example_options.side_effect = example.set_options
    bridge.list_safety_requirement_files.return_value = [str(ROOT / path) for path in expected]
    bridge.list_recovery_observations.return_value = []
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    with client:
        scenario = next(element for element in client.elements.values()
                        if isinstance(element, ui.select) and element._props.get("label") == "Failure scenario")
        safety = next(element for element in client.elements.values()
                      if isinstance(element, ui.select) and element._props.get("label") == "Safety specifications")
        assert safety.value == default_safety_files("Conveyor breakdown")
        scenario.value = "Part slippage"
        assert safety.value == expected
        assert set(safety.options) == set(expected)
        safety.value = []
        assert example.snapshot()["selected_safety_files"] == []
    bridge.start_failure_example.assert_not_called()
    bridge.start_offline_recovery.assert_not_called()
    bridge.part_slippage_action.assert_not_called()
    client.delete()


@pytest.mark.parametrize("scenario", ["Conveyor breakdown", "ur5e-1 breakdown", "Machining breakdown during part processing"])
@pytest.mark.parametrize("ready", [True, False])
def test_offline_case_selection_filters_captures_and_uses_contract_readiness(scenario, ready):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock(system_running=False)
    example = PartSlippageExample(bridge)
    example.set_options(source="offline", scenario=scenario, safety_files=[])
    state = example.snapshot()
    state.update(status="outline_review", validation_contract_version=2, outline_status="completed", outline=[{"outline_id": "retained"}])
    state.update(selected_observation_path="selected", defer_preparation=True)
    bridge.get_default_recovery_observations.return_value = {scenario: "selected"}
    state["composition_readiness"] = {"ready": ready, "reasons": [] if ready else ["Owner contract is missing"]}
    bridge.get_part_slippage_example.return_value = state
    bridge.list_safety_requirement_files.return_value = []
    bridge.list_recovery_observations.return_value = [
        {"source_path": "selected", "scenario": scenario, "run_id": "case-run", "offline_ready": True},
        {"source_path": "other", "scenario": "Part slippage", "run_id": "other-run", "offline_ready": True},
    ]
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    selected = next(item for item in client.elements.values() if isinstance(item, ui.select)
                    and item._props.get("label") == "Saved observations")
    scenario_select = next(item for item in client.elements.values() if isinstance(item, ui.select)
                           and item._props.get("label") == "Failure scenario")
    assert set(selected.options) == {"selected"}
    assert scenario_select.visible
    for item in client.elements.values():
        if isinstance(item, ui.button) and item.text in {"Continue", "Execute recovery", "Test in Gazebo"}:
            assert not item.enabled
        if isinstance(item, ui.button) and item.text == "Generate" and item is not next(
                element for element in client.elements.values() if isinstance(element, ui.button) and element.text == "Generate"):
            assert item.enabled is ready
    labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
    assert "Capture observations → Recovery outline → Primitive composition → Review" in labels
    assert not any("not supported for" in label for label in labels)
    for item in client.elements.values():
        if isinstance(item, ui.button) and item.text == "Test in Gazebo":
            assert item.visible and item.parent_slot.parent.visible
    bridge.part_slippage_action.assert_not_called()
    client.delete()


def test_safety_panel_marks_relative_defaults_selected_with_absolute_bridge_options():
    from cais_spade_llm.recovery_framework import ROOT
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock()
    state = PartSlippageExample(bridge).snapshot()
    selected = state["selected_safety_files"]
    bridge.get_part_slippage_example.return_value = state
    bridge.list_safety_requirement_files.return_value = [str(ROOT / path) for path in selected]
    bridge.list_recovery_observations.return_value = []
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    safety = next(element for element in client.elements.values()
                  if isinstance(element, ui.select) and element._props.get("label") == "Safety specifications")
    assert safety.value == selected
    assert set(safety.options) == set(selected)
    assert set(safety.options.values()) == {Path(path).name for path in selected}
    assert safety._props["use-chips"] is True
    bridge.set_failure_example_options.assert_not_called()
    bridge.start_failure_example.assert_not_called()
    client.delete()


@pytest.fixture
def recovery_browser(monkeypatch):
    browser = {}
    class Reply:
        def __init__(self, value):
            self.value = value
        def __await__(self):
            async def result():
                return self.value
            return result().__await__()
    def javascript(script):
        if script.startswith("localStorage.getItem("):
            key = json.loads(script[len("localStorage.getItem("):-1])
            return Reply(browser.get(key))
        key, value = script[len("localStorage.setItem("):].split(", String(", 1)
        browser[json.loads(key)] = value.split(")", 1)[0]
        return Reply(None)
    monkeypatch.setattr(ui, "run_javascript", javascript)
    return browser


def test_recovery_turn_expansions_are_independent_lazy_and_survive_updates(tmp_path, monkeypatch, recovery_browser):
    from cais_spade_llm.ui.components import recovery_turns

    reader = Mock(return_value={"captures": [], "results": [], "prompts": []})
    monkeypatch.setattr(recovery_turns, "read_recovery_turn", reader)
    state = {"run_id": "run-one", "artifact_directory": str(tmp_path / "generation_0001")}
    turn = {"turn": 1, "phase": "outline", "status": "need_context", "files": [], "signature": [1]}
    client = Client(context.client.page)
    with client:
        view = recovery_turns.RecoveryTurns(ui.column())
        view.update(state, "outline", [turn])
        expansions = [row for row in client.elements.values() if isinstance(row, ui.expansion)]
        assert [row.text for row in expansions] == ["Input", "Output"]
        assert all(row.value is False for row in expansions)
        reader.assert_not_called()
        expansions[0].value = True
    _finish_reads(client)
    reader.assert_called_once()
    with client:
        view.update(state, "outline", [turn, {**turn, "turn": 2}])
    assert expansions[0].value is True and expansions[1].value is False
    assert view.rows[(1, "outline")]["label"].text.startswith("Turn 1")
    assert len(view.rows) == 2
    client.delete()

    restored_client = Client(context.client.page)
    with restored_client:
        restored = recovery_turns.RecoveryTurns(ui.column())
        restored.update(state, "outline", [turn])
    async def reconnect():
        for timer in list(restored_client.elements.values()):
            if isinstance(timer, ui.timer) and timer.callback.__qualname__.endswith(".restore"):
                timer.cancel()
                await timer.callback()
    asyncio.run(reconnect())
    restored_expansions = [row for row in restored_client.elements.values() if isinstance(row, ui.expansion)]
    assert restored_expansions[0].value is True
    assert restored_expansions[1].value is False
    restored_client.delete()


def test_output_separates_ra_physical_and_cca_evidence(recovery_browser):
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    evidence = {"captures": [], "prompts": [], "results": [{"path": "/tmp/turn.json", "payload": {
        "validation_stages": [
            {"validator_role": "ResourceAgent", "evidence": {"allowed": True, "feasibility_status": "FEASIBLE",
                "geometry_evidence": {"status": "needs_context", "unresolved": ["Missing interval"]}}},
            {"validator_role": "CentralControllerAgent", "evidence": {"is_safe": False, "validation_status": "NEEDS_CONTEXT"}},
        ]}}]}
    client = Client(context.client.page)
    with client:
        render_turn_content(evidence, "output", state_key="run:attempt:turn")
        validation = next(item for item in client.elements.values()
                          if isinstance(item, ui.expansion) and item.text == "Validation records")
        validation.value = True
    _finish_reads(client)
    sections = [item for item in client.elements.values()
                if isinstance(item, ui.expansion) and item is not validation]
    assert [item.text for item in sections] == ["RA feasibility", "Physical evidence", "CCA safety"]
    assert all(item.value is False for item in sections)
    client.delete()


def _proposal_row(index, event, chosen=""):
    return {"row": index, "chosen": chosen, "event_name": event["event_name"],
            "resource": event.get("resource") or event.get("resource_jid"), "part_name": event.get("part_name"),
            "expected_end_state": "", "neighbor_end_state": "", "params": "", "rationale": ""}


def test_output_lists_exact_proposals_before_responses_and_keeps_attempts(tmp_path, recovery_browser):
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    events = [
        {"event_name": "recover_pick_gear_small", "resource_jid": "recovery-resource-4@localhost", "part_name": "gear_small"},
        {"event_name": "recover_to_idle", "resource_jid": "recovery-resource-4@localhost", "part_name": None},
    ]
    captures = []
    originals = {}
    for attempt, proposed in ((1, events), (2, list(reversed(events)))):
        content = json.dumps({"thought": "Recorded proposal", "candidate_events": proposed}, separators=(",", ":"))
        record = {"kind": "provider_response", "tool_round": 0, "attempt": attempt,
                  "raw_response": {"choices": [{"message": {"content": content}}]}}
        path = _write(tmp_path, f"attempt{attempt}_response.json", record)
        originals[path] = path.read_bytes()
        captures.append({"path": str(path), "status": "captured", "record": record})
    result = {"path": str(tmp_path / "audit.json"), "response": {"event_name": "not_an_llm_proposal"},
              "payload": {"selected_transition": {"event_name": "runtime_selected"},
                          "candidate_evaluation_summary": [{"status": "rejected", "task": events[0]}],
                          "served_context": {"/parts/gear_small/observed_pose": {"value": {"x": 0.3}}}}}
    summary = {"path": str(tmp_path / "turn.json"), "response": {}, "payload": {"decision": "outline_ready"}}
    data = {"captures": captures, "results": [result, summary], "prompts": []}
    before = json.dumps(data)
    client = Client(context.client.page)
    with client:
        render_turn_content(data, "output", state_key="run:generation:turn:output")
    tables = [item for item in client.elements.values() if isinstance(item, ui.table)]
    assert len(tables) == 2
    assert tables[0].rows == [_proposal_row(index, event) for index, event in enumerate(events)]
    assert tables[1].rows == [_proposal_row(index, event) for index, event in enumerate(reversed(events))]
    labels = [item for item in client.elements.values() if isinstance(item, ui.label)]
    response_title = next(item for item in labels if item.text == "LLM response")
    assert all(item.id < response_title.id for item in tables)
    assert any('\n  "thought": "Recorded proposal"' in item.text for item in labels)
    assert not any("runtime_selected" in item.text for item in labels)
    details = [item for item in client.elements.values() if isinstance(item, ui.expansion)]
    assert [item.text for item in details] == ["Validation records", "Returned context"]
    assert all(item.value is False for item in details)
    with client:
        details[0].value = True
    _finish_reads(client)
    assert details[1].value is False
    assert any(isinstance(item, ui.label) and "runtime_selected" in item.text for item in client.elements.values())
    assert len([item for item in client.elements.values() if isinstance(item, ui.button) and item.text == "Download original"]) == 2
    assert json.dumps(data) == before
    assert all(path.read_bytes() == content for path, content in originals.items())
    client.delete()


@pytest.mark.parametrize("response", [
    {"thought": "Saved pose available", "decision": "grounded", "observe_requests": []},
    {"thought": "Read recorded facts", "decision": "need_context", "context_requests": ["/parts"]},
])
def test_output_grounding_and_retrieval_never_show_runtime_events(response):
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    data = {"captures": [{"path": "response.json", "status": "captured", "record": {
        "kind": "provider_response", "raw_response": {"choices": [{"message": {"content": json.dumps(response)}}]},
    }}], "results": [{"path": "audit.json", "response": {}, "payload": {
        "selected_transition": {"event_name": "runtime_only"}, "transition_trace": [{"event_name": "runtime_only"}],
    }}], "prompts": []}
    client = Client(context.client.page)
    with client:
        render_turn_content(data, "output")
    assert not any(isinstance(item, ui.table) for item in client.elements.values())
    labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
    assert "No events proposed in this turn" in labels
    assert ("Requested context: /parts" in labels) is (response.get("decision") == "need_context")
    client.delete()


@pytest.mark.parametrize("complete", [True, False])
def test_closing_turn_shows_the_nominal_handoff_by_resource_name(complete):
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    names = {"robot-3@localhost": "ur5e-3", "robot-4@localhost": "ur5e-4"}
    connection = [{"event_name": "pick_grasp", "resource_jid": "robot-3@localhost", "part_name": "KET4",
                   "expected_end_state": {"part_location": "robot-3@localhost"}}]
    handoff = {"complete": complete, "declared_complete_by_llm": True, "nominal_connection": connection if complete else [],
               "not_connected_reasons": [] if complete else ["robot-4@localhost did the interrupted work and is still reported faulted"],
               "assumptions": [{"task_id": "pending_4", "reason": "no declared event of robot-4@localhost binds the resumed "
                                "advance_conveyor; whether nominal work continues there is not checked"}] if complete else []}
    payload = {"decision": "final_output_ready", "resource_names": names, "nominal_handoff": handoff,
               "transition_trace": [{"event_name": "gear_reseated", "resource_jid": "robot-3@localhost",
                                     "part_name": "gear_small", "expected_end_state": {"part_state": "assembled"}}]}
    data = {"captures": [], "results": [{"path": "final.txt", "payload": payload, "response": {}}], "prompts": []}
    for tab in ("input", "output"):
        client = Client(context.client.page)
        with client:
            render_turn_content(data, tab)
        labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
        tables = [item for item in client.elements.values() if isinstance(item, ui.table)]
        # The closing turn is code's record, so no LLM capture is expected.
        assert "Handoff to nominal work — no LLM call" in labels
        assert not {"Exact request capture missing.", "Raw LLM response capture missing."} & set(labels)
        if tab == "input":
            assert (tables[0].rows[0]["event_name"], tables[0].rows[0]["resource"]) == ("gear_reseated", "ur5e-3")
        elif complete:
            assert "Nominal work takes over." in labels
            assert ("Assumed, not checked: no declared event of ur5e-4 binds the resumed advance_conveyor; whether "
                    "nominal work continues there is not checked") in labels
            assert tables[0].rows[0]["resource"] == "ur5e-3"
            assert tables[0].rows[0]["expected_end_state"] == "part_location: ur5e-3"
        else:
            assert not tables and ("Nominal work cannot take over: ur5e-4 did the interrupted work and is still "
                                   "reported faulted") in labels
        client.delete()


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_raw_output_does_not_present_selected_transition_as_llm_proposal(legacy):
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    payload = {"selected_transition": {"event_name": "runtime_only"}}
    if legacy:
        payload["llm_response"] = {"candidate_events": [{"event_name": "original_event", "resource_jid": "ur5e@localhost"}]}
    data = {"captures": [{"path": "missing.json", "status": "not captured", "record": {}}],
            "results": [{"path": "audit.json", "payload": payload, "response": payload["selected_transition"]}], "prompts": []}
    client = Client(context.client.page)
    with client:
        render_turn_content(data, "output")
    tables = [item for item in client.elements.values() if isinstance(item, ui.table)]
    labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
    assert "Raw LLM response capture missing." in labels
    if legacy:
        assert tables[0].rows[0]["event_name"] == "original_event"
        assert "Legacy parsed LLM response — raw capture unavailable" in labels
    else:
        assert not tables
        assert "Proposed events unavailable: LLM response capture missing." in labels
    client.delete()


@pytest.mark.parametrize("field", ["candidate_traces", "next_transition", "transition_trace"])
def test_output_preserves_legacy_llm_event_formats_and_order(field):
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    events = [{"event_name": name, "resource_jid": "ur5e@localhost", "part_name": "LG"} for name in ("first", "second")]
    response = ({field: [{"events": events}]} if field == "candidate_traces" else
                {field: events[0], "transition_suffix": events[1:]} if field == "next_transition" else {field: events})
    data = {"captures": [{"path": "response.json", "status": "captured", "record": {
        "kind": "provider_response", "raw_response": {"choices": [{"message": {"content": json.dumps(response)}}]},
    }}], "results": [], "prompts": []}
    client = Client(context.client.page)
    with client:
        render_turn_content(data, "output")
    table = next(item for item in client.elements.values() if isinstance(item, ui.table))
    assert table.rows == [_proposal_row(index, event) for index, event in enumerate(events)]
    client.delete()


def test_output_marks_the_chosen_candidate_with_its_resource_outcome_and_rationale():
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    events = [{"event_name": "clear_board", "resource": "ur5e-4", "expected_end_state": {"resource_location": "home"},
               "rationale": "Leave the shared area."},
              {"event_name": "set_down_peg", "resource": "ur5e-3", "part_name": "KET4_Square_4mm",
               "expected_end_state": {"held_part": None, "part_location": "Buffer For Machined parts"},
               "params": {"pose": [0.1, 0.2, 1.0]}, "rationale": "Free the gripper."}]
    content = json.dumps({"thought": "Choose one.", "candidate_events": events, "selected_candidate_index": 1})
    data = {"captures": [{"path": "response.json", "status": "captured", "record": {
        "kind": "provider_response", "raw_response": {"choices": [{"message": {"content": content}}]},
    }}], "results": [], "prompts": []}
    client = Client(context.client.page)
    with client:
        render_turn_content(data, "output")
    table = next(item for item in client.elements.values() if isinstance(item, ui.table))
    assert [column["label"] for column in table.columns] == [
        "Chosen", "Event", "Resource", "Part", "End state", "Params", "Rationale"]
    assert [row["chosen"] for row in table.rows] == ["", "✓"]
    assert table.rows[1]["resource"] == "ur5e-3"
    assert table.rows[1]["expected_end_state"] == "held_part: None; part_location: Buffer For Machined parts"
    assert table.rows[1]["params"] == "pose: [0.1, 0.2, 1.0]" and table.rows[1]["rationale"] == "Free the gripper."
    client.delete()


def test_invalid_llm_json_remains_visible_without_invented_events():
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    content = '{"candidate_events": [unfinished'
    data = {"captures": [{"path": "response.json", "status": "captured", "record": {
        "kind": "provider_response", "raw_response": {"choices": [{"message": {"content": content}}]},
    }}], "results": [], "prompts": []}
    client = Client(context.client.page)
    with client:
        render_turn_content(data, "output")
    assert not any(isinstance(item, ui.table) for item in client.elements.values())
    assert any(isinstance(item, ui.label) and item.text == content for item in client.elements.values())
    client.delete()


def test_legacy_primitive_output_keeps_its_parsed_response():
    from cais_spade_llm.ui.components.recovery_turns import render_turn_content

    response = {"decision": "need_context", "context_requests": ["/parts/gear_small/observed_pose"]}
    data = {"captures": [], "prompts": [], "results": [{
        "path": "multi_turn_RECOVERY_SEQ1_primitive_generation_turn01_response_20261008T100001.txt",
        "payload": {"response": response}, "response": response,
    }]}
    client = Client(context.client.page)
    with client:
        render_turn_content(data, "output")
    labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
    assert "Legacy parsed response — raw capture unavailable" in labels
    assert json.dumps(response, indent=2) in labels
    assert not any(isinstance(item, ui.table) for item in client.elements.values())
    client.delete()


def test_turn_validation_and_context_expansions_survive_refresh_and_reconnection(tmp_path, monkeypatch, recovery_browser):
    from cais_spade_llm.ui.components import recovery_turns

    data = {"captures": [], "prompts": [], "results": [{"path": "audit.json", "response": {}, "payload": {
        "selected_transition": {"event_name": "runtime_selected"},
        "served_context": {"/parts": [{"part_name": "gear_small"}]},
    }}]}
    monkeypatch.setattr(recovery_turns, "read_recovery_turn", lambda *_: data)
    state = {"run_id": "run-one", "artifact_directory": str(tmp_path / "generation_0001")}
    turn = {"turn": 1, "phase": "outline", "status": "recorded", "files": [], "signature": [1]}

    def restore(client):
        async def finish():
            for timer in list(client.elements.values()):
                if isinstance(timer, ui.timer) and not timer._is_canceled and timer.callback.__qualname__.endswith(".restore"):
                    timer.cancel()
                    await timer.callback()
        asyncio.run(finish())
        _finish_reads(client)

    client = Client(context.client.page)
    with client:
        view = recovery_turns.RecoveryTurns(ui.column())
        view.update(state, "outline", [turn])
        output = next(item for item in client.elements.values() if isinstance(item, ui.expansion) and item.text == "Output")
        output.value = True
    _finish_reads(client)
    with client:
        validation = next(item for item in client.elements.values() if isinstance(item, ui.expansion) and item.text == "Validation records")
        validation.value = True
    _finish_reads(client)
    with client:
        view.update(state, "outline", [{**turn, "signature": [2]}, {**turn, "turn": 2}])
    _finish_reads(client)
    restore(client)
    expanded = [item.text for item in client.elements.values() if isinstance(item, ui.expansion) and item.value]
    assert expanded == ["Output", "Validation records"]
    assert any(isinstance(item, ui.label) and "runtime_selected" in item.text for item in client.elements.values())
    client.delete()

    client = Client(context.client.page)
    with client:
        view = recovery_turns.RecoveryTurns(ui.column())
        view.update(state, "outline", [turn])
    restore(client)
    restore(client)
    assert [item.text for item in client.elements.values() if isinstance(item, ui.expansion) and item.value] == ["Output", "Validation records"]
    with client:
        view.update({**state, "artifact_directory": str(tmp_path / "generation_0002")}, "outline", [turn])
    restore(client)
    assert all(not item.value for item in client.elements.values() if isinstance(item, ui.expansion))
    client.delete()


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
            "/recovery-framework?tab=experiments",
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
        "Navigation", "dashboard", "setup", "results", "recovery", "experiments", "dashboard", "setup",
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
    (b"tab=experiments", "experiments"),
    (b"tab=", "run"), (b"tab=unknown", "run"),
])
def test_recovery_direct_links_build_only_the_selected_view(query, expected, monkeypatch):
    """Direct links select one view; recovery reads progress without starting work."""
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample

    bridge = MagicMock(spec=SystemBridge)
    bridge.get_part_slippage_example.return_value = PartSlippageExample(bridge).snapshot()
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
                isinstance(element, ui.select) and element._props.get("label") == "Saved observations"
                for element in client.elements.values()
            )
            assert not any(isinstance(element, ui.badge) and element.text == "Placeholder"
                           for element in client.elements.values())
            assert not any(isinstance(element, ui.expansion) and element.text == "Details"
                           for element in client.elements.values())
        assert all(call[0] in {"get_part_slippage_example", "get_default_recovery_observations"}
                   for call in bridge.mock_calls)
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
            "run", "setup", "results", "recovery", "experiments",
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


def test_experiments_placeholder_keeps_one_view_and_downloads_design(monkeypatch):
    """Opening the placeholder and its download never accesses the runtime."""
    bridge = MagicMock(spec=SystemBridge)
    render_run = Mock()
    monkeypatch.setattr(recovery_run, "render", render_run)
    download = Mock()
    monkeypatch.setattr(ui.download, "file", download)
    callbacks = {}
    original_click = ui.button.on_click

    def capture_click(button, callback):
        callbacks[button.text] = callback
        return original_click(button, callback)

    monkeypatch.setattr(ui.button, "on_click", capture_click)
    request = Request({
        "type": "http", "path": "/recovery-framework", "headers": [],
        "query_string": b"tab=experiments",
    })
    client = Client(context.client.page, request=request)
    with client:
        recovery_framework.render(bridge)
        tabs = next(item for item in client.elements.values() if isinstance(item, ui.tabs))
        badge = next(item for item in client.elements.values() if isinstance(item, ui.badge))
        assert badge.text == "Coming soon"
        labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
        assert "Experiments" in labels
        assert "Experiment configuration and results will be added here." in labels
        assert not any(isinstance(item, ui.timer) for item in client.elements.values())
        callbacks["Download experiment design"]()
        download.assert_called_once_with(recovery_framework._EXPERIMENT_DESIGN)
        assert recovery_framework._EXPERIMENT_DESIGN.is_file()
        tabs.value = "run"
        tabs.value = "experiments"
        assert [item for item in client.elements.values() if isinstance(item, ui.badge)] == [badge]
        assert len([item for item in client.elements.values() if isinstance(item, ui.button)]) == 1
        render_run.assert_called_once()
        assert bridge.mock_calls == []
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
    client = Client(context.client.page, request=Request({
        "type": "http", "path": "/recovery-framework", "headers": [], "query_string": b"",
    }))
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


@pytest.mark.parametrize('recorded', [False, True])
def test_recovery_panel_shows_only_recorded_attempt_provenance(recorded):
    from cais_spade_llm.recovery_framework.part_slippage_example import PartSlippageExample
    from cais_spade_llm.ui.components.part_slippage_recovery import render_part_slippage

    bridge = MagicMock(system_running=False)
    state = PartSlippageExample(bridge).snapshot()
    if recorded:
        state.update(provenance_path='/recorded/attempt_provenance.json', attempt_provenance={
            'capture': {'run_id': 'captured-run', 'captured_at': 'captured-time', 'source_sha256': 'capturehash'},
            'configuration_fingerprint': 'configurationhash', 'model_fingerprint': 'modelhash',
            'implementation_fingerprint': 'implementationhash',
            'safety_specifications': [{'path': 'recorded-safety.txt', 'sha256': 'safetyhash'}], 'unassisted': True})
    bridge.get_part_slippage_example.return_value = state
    bridge.list_safety_requirement_files.return_value = []
    bridge.list_recovery_observations.return_value = []
    client = Client(context.client.page)
    with client:
        render_part_slippage(bridge)
    _finish_reads(client)
    labels = [item.text for item in client.elements.values() if isinstance(item, ui.label)]
    button = next(item for item in client.elements.values() if isinstance(item, ui.button)
                  and item.text == 'Download attempt provenance')
    if recorded:
        assert any('captured-run' in text and 'capturehash' in text and 'recorded-safety.txt safetyhash' in text
                   and 'Implementation implementati' in text and 'unassisted' in text for text in labels)
        assert button.visible
    else:
        assert 'Attempt provenance not recorded.' in labels
        assert not button.visible
    bridge.part_slippage_action.assert_not_called()
    client.delete()
