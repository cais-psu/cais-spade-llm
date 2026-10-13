from __future__ import annotations

"""Operator controls and exact prompt captures for the Part slippage run."""

import asyncio
from pathlib import Path

from nicegui import ui

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.ui.components.recovery_turns import RecoveryTurns
from cais_spade_llm.ui.evidence import BackgroundSection
from cais_spade_llm.ui.recovery_evidence import recovery_turns
from cais_spade_llm.ui.recovery_setup import FAILURE_SCENARIOS
from cais_spade_llm.ui.refresh import PageRefresh


def render_part_slippage(bridge, *, is_active=lambda: True) -> None:  # noqa: C901, PLR0915 - shared page controls
    """Render one bridge-owned workflow; navigation only stops page polling."""
    state = bridge.get_part_slippage_example()
    refresh = PageRefresh(is_active)
    workflow_label = ui.label("Start failure → Recovery outline → Primitive composition → Execute recovery").classes("text-slate-600")
    safety_badge = ui.badge("Gazebo — safety checks deferred").props("color=amber text-color=black")
    progress = ui.label(state["message"]).classes("font-semibold")
    run_id = ui.label().classes("text-xs text-slate-500")
    controls = {}
    stages = {}
    fields = {}
    updating = False
    observed_signature = None
    saved_options = {}
    saved_records = {}
    available_safety = []

    async def act(action: str, feedback: str = "") -> None:
        try:
            if action == "start":
                await bridge.start_failure_example()
            elif action == "load_offline":
                bridge.load_offline_recovery()
            else:
                await bridge.part_slippage_action(action, feedback=feedback)
        except (ValueError, RuntimeError, OSError) as exc:
            ui.notify(str(exc), type="negative")
        update()

    with ui.card().classes("w-full"):
        ui.label("Example").classes("text-lg font-semibold")
        fields["source"] = ui.toggle({"gazebo": "Gazebo", "offline": "Offline"}, value=state.get("source", "gazebo"))
        fields["scenario"] = ui.select(list(FAILURE_SCENARIOS), value=state.get("scenario", "Part slippage"), label="Failure scenario").classes("w-full")
        fields["observations"] = ui.select({}, label="Saved observations").classes("w-full")
        with ui.row().classes("items-center"):
            ui.label("Recovery mode").classes("font-semibold")
            recovery_mode = ui.toggle({"auto": "Auto", "manual": "Manual"},
                value=state["outline_mode"] if state["outline_mode"] == state["primitive_mode"] else None).props("no-caps")
            recovery_mode.tooltip("Applies to Recovery outline and Primitive composition")
        mode_note = ui.label().classes("text-sm text-slate-500")
        with ui.row():
            controls["start"] = ui.button("Start failure", icon="play_arrow", on_click=lambda: act("start"))
            controls["load_offline"] = ui.button("Load saved result", on_click=lambda: act("load_offline")).props("outline")
            controls["stop"] = ui.button("Stop", icon="stop", on_click=lambda: act("stop")).props("outline")
            controls["reset"] = ui.button("Reset example", icon="restart_alt", on_click=lambda: act("reset")).props("flat")
        capture_status = ui.label().classes("text-sm text-slate-500")
        capture_path = ""
        download_observations = ui.button("Download observations", icon="download",
            on_click=lambda: ui.download.file(capture_path)).props("flat dense")
        provenance_status = ui.label().classes("text-sm text-slate-500")
        provenance_path = ""
        download_provenance = ui.button("Download attempt provenance", icon="download",
            on_click=lambda: ui.download.file(provenance_path)).props("flat dense")

    with ui.card().classes("w-full"):
        ui.label("Safety").classes("text-lg font-semibold")
        selected_safety = state.get("selected_safety_files", [])
        fields["safety"] = ui.select({path: Path(path).name for path in selected_safety}, multiple=True,
                                     value=selected_safety, label="Safety specifications").classes("w-full").props("use-chips")
        safety_note = ui.label().classes("text-sm text-amber-700")

    def options_changed() -> None:
        if updating:
            return
        try:
            observation_path = fields["observations"].value or ""
            if (observation_path and saved_records.get(observation_path, {}).get("scenario") != fields["scenario"].value):
                observation_path = ""
            selected = bridge.set_failure_example_options(
                source=fields["source"].value, scenario=fields["scenario"].value,
                safety_files=fields["safety"].value or [], observation_path=observation_path)
            if selected.get("source") == "offline" and selected.get("selected_observation_path"):
                bridge.prepare_offline_recovery()
        except (ValueError, OSError) as exc:
            ui.notify(str(exc), type="warning")
        update()

    for field in fields.values():
        field.on_value_change(options_changed)

    def show_options(values) -> None:
        nonlocal saved_options, saved_records, available_safety
        captures, safety = values
        available_safety = safety
        saved_records = {path: row for row in captures
                         for path in row.get("related_observation_paths", [row["source_path"]])}
        saved_options = {row["source_path"]: (
            f"{row['scenario']} · {row.get('run_id') or row.get('saved_run_id')} · "
            f"{row.get('captured_at') or 'capture time not recorded'} · "
            f"{'Ready to prepare' if row.get('offline_ready') else 'Missing context: ' + ', '.join(row.get('missing') or [])}"
        ) for row in captures}
        update()

    option_reader = BackgroundSection()

    def load_options() -> None:
        option_reader.load(lambda: (bridge.list_recovery_observations(), bridge.list_safety_requirement_files()), show_options)

    def mode_changed() -> None:
        if updating or recovery_mode.value is None:
            return
        try:
            bridge.set_part_slippage_modes(recovery_mode.value, recovery_mode.value)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
        update()

    for key, title, artifact_stage in (
        ("outline", "Recovery outline", "recovery_outline"),
        ("primitive", "Primitive composition", "recovery_primitves"),
    ):
        with ui.card().classes("w-full") as stage_card:
            ui.label(title).classes("text-lg font-semibold")
            status = ui.label().classes("text-sm text-slate-600")
            with ui.row():
                generate_action = "generate_outline" if key == "outline" else "generate_primitives"
                controls[generate_action] = ui.button("Generate", on_click=lambda action=generate_action: act(action)).props("outline")
                if key == "outline":
                    controls["continue_outline"] = ui.button("Continue", on_click=lambda: act("continue_outline"))
            if key == "outline":
                controls["prepare"] = ui.button("Retry input preparation", on_click=lambda: act("prepare")).props("outline")
                controls["prepare_gazebo_evidence"] = ui.button(
                    "Prepare Gazebo evidence", on_click=lambda: act("prepare_gazebo_evidence")).props("outline")
                controls["use_new_observations"] = ui.button(
                    "Use new observations", on_click=lambda: act("use_new_observations")).props("outline")
            turn_view = RecoveryTurns(ui.column().classes("w-full"))
            with ui.expansion("Feedback and Regenerate").classes("w-full"):
                feedback = ui.textarea("Optional feedback").classes("w-full").props("autogrow")
                action = "regenerate_outline" if key == "outline" else "regenerate_primitives"
                controls[action] = ui.button("Regenerate", on_click=lambda action=action, field=feedback: act(action, field.value or "")).props("outline")
            stages[key] = {"status": status, "artifact_stage": artifact_stage,
                           "key": key,
                           "card": stage_card,
                           "turn_view": turn_view}
    recovery_mode.on_value_change(mode_changed)

    with ui.card().classes("w-full") as execution_card:
        ui.label("Execution").classes("text-lg font-semibold")
        controls["execute"] = ui.button("Execute recovery", icon="play_arrow", on_click=lambda: act("execute"))
        controls["test_in_gazebo"] = ui.button("Test in Gazebo", icon="science", on_click=lambda: act("test_in_gazebo"))
        execution = ui.label().classes("text-sm whitespace-pre-wrap")

    def update(current=None, files=None) -> None:
        nonlocal updating, observed_signature, capture_path, provenance_path
        current = current if current is not None else bridge.get_part_slippage_example()
        status = current["status"]
        offline = current.get("source") == "offline"
        validated_outline = current.get("validation_contract_version") == 2
        outline_only = current.get("outline_only", current.get("scenario") != "Part slippage")
        workflow_label.text = ("Start failure → Capture observations → Recovery outline → Review"
                               if outline_only else "Start failure → Recovery outline → Primitive composition → Execute recovery")
        safety_badge.text = (f"{'Offline' if offline else 'Gazebo'} — outline safety: {current.get('outline_safety_status', 'pending')}; execution validation pending"
                             if validated_outline else "Offline — saved observations; safety checks deferred" if offline else "Gazebo — safety checks deferred")
        safety_note.text = ("Selected specifications are checked against RA-derived outline evidence. Runtime CCA integration is pending."
                            if validated_outline else "Selected specifications are included in the prompt. Formal safety checks are skipped.")
        progress.text = current["message"]
        run_id.text = f"Run: {current['run_id']}" if current.get("run_id") else ""
        provenance = current.get("attempt_provenance") or {}
        capture = provenance.get("capture") or {}
        def short(value):
            return str(value)[:12] if value else "not recorded"
        provenance_status.text = (
            f"Capture: {capture.get('run_id') or 'not recorded'} · {capture.get('captured_at') or 'time not recorded'} · "
            f"SHA256 {short(capture.get('source_sha256'))} · Configuration {short(provenance.get('configuration_fingerprint'))} · "
            f"Model {short(provenance.get('model_fingerprint'))} · Implementation {short(provenance.get('implementation_fingerprint'))} · "
            "Safety: " + ", ".join(f"{Path(row['path']).name} {short(row['sha256'])}"
                                    for row in provenance.get("safety_specifications", [])) +
            (" · unassisted" if provenance.get("unassisted") is True else
             " · operator-assisted" if provenance.get("unassisted") is False else " · assistance not recorded")
            if provenance else "Attempt provenance not recorded.")
        provenance_path = current.get("provenance_path") or ""
        download_provenance.set_visibility(bool(provenance_path))
        available = {
            "start": status == "idle", "stop": status not in {"idle", "stopped", "resetting", "completed"},
            "load_offline": status in {"idle", "outline_ready", "stopped"} and bool(saved_records.get(current.get("selected_observation_path"), {}).get("result_available")),
            "prepare": status == "blocked" and bool(current.get("observation_path")),
            "prepare_gazebo_evidence": bool(current.get("can_prepare_gazebo_evidence")),
            "use_new_observations": bool(current.get("new_observation_path")),
            "test_in_gazebo": status == "offline_completed" and not validated_outline,
            "reset": status not in {"starting", "resetting"} and not current.get("busy"),
            "generate_outline": current.get("can_generate_outline", status == "outline_ready"), "continue_outline": status == "outline_review" and not outline_only,
            "generate_primitives": status == "primitive_ready", "execute": status == "ready_to_execute" and not validated_outline,
            "regenerate_outline": status in {"outline_review", "primitive_ready", "ready_to_execute", "offline_completed", "blocked"} and bool(current.get("artifact_directory")),
            "regenerate_primitives": (status in {"primitive_ready", "ready_to_execute", "offline_completed", "blocked"}
                                      and bool(current.get("outline")) and not current.get("imported_recovery")),
        }
        for key, button in controls.items():
            if outline_only and key in {"continue_outline", "generate_primitives", "regenerate_primitives", "execute", "test_in_gazebo"}:
                available[key] = False
            button.set_enabled(available[key] and (key == "stop" or not current.get("busy")))
        controls["generate_outline"].set_visibility(offline or current["outline_mode"] == "manual")
        controls["continue_outline"].set_visibility(not outline_only and (offline or current["outline_mode"] == "manual"))
        controls["generate_primitives"].set_visibility(outline_only or offline or current["primitive_mode"] == "manual")
        controls["start"].set_visibility(not offline)
        controls["load_offline"].set_visibility(offline and available["load_offline"])
        controls["test_in_gazebo"].set_visibility(offline)
        controls["execute"].set_visibility(not offline)
        controls["prepare"].set_visibility(available["prepare"])
        controls["prepare_gazebo_evidence"].set_visibility(available["prepare_gazebo_evidence"])
        controls["use_new_observations"].set_visibility(available["use_new_observations"])
        fields["observations"].set_visibility(offline)
        fields["scenario"].set_visibility(True)
        updating = True
        fields["source"].value = current.get("source", "gazebo")
        fields["scenario"].value = current.get("scenario", "Part slippage")
        selected_observation = current.get("selected_observation_path") or None
        if selected_observation in saved_records:
            selected_observation = saved_records[selected_observation]["source_path"]
        scenario_options = {path: label for path, label in saved_options.items()
                            if saved_records[path].get("scenario") == current.get("scenario")}
        if selected_observation not in scenario_options:
            selected_observation = None
        if fields["observations"].options != scenario_options or fields["observations"].value != selected_observation:
            fields["observations"].set_options(scenario_options, value=selected_observation)
        selected_safety = current.get("selected_safety_files", [])
        # Keep recorded selection strings as option keys even when the bridge
        # lists the same files with absolute paths, including scenario changes.
        selected_paths = {(ROOT / path).resolve(): path for path in selected_safety}
        safety_options = {selected_paths.get((ROOT / path).resolve(), path): Path(path).name for path in available_safety}
        safety_options.update({path: Path(path).name for path in selected_safety})
        if fields["safety"].options != safety_options or fields["safety"].value != selected_safety:
            fields["safety"].set_options(safety_options, value=selected_safety)
        modes_match = current["outline_mode"] == current["primitive_mode"]
        recovery_mode.value = "manual" if offline else current["outline_mode"] if modes_match else None
        recovery_mode.set_enabled(not offline and status == "idle" and not current.get("busy"))
        mode_note.text = "Offline always uses Manual." if offline else (f"Recorded modes: Recovery outline — {current['outline_mode']}; "
                          f"Primitive composition — {current['primitive_mode']}.") if not modes_match else ""
        mode_note.set_visibility(offline or not modes_match)
        for field in fields.values():
            field.set_enabled(not current.get("busy") and status not in {"running_example", "executing", "resuming"})
        if offline and status == "preparing_outline":
            fields["observations"].set_enabled(True)
            fields["source"].set_enabled(True)
        for key in ("scenario", "safety"):
            fields[key].set_enabled(status in {"idle", "stopped"} and not current.get("busy"))
        updating = False
        observed = current.get("observations") or {}
        signature = (current.get("observation_path"), current.get("selected_observation_path"),
                     observed.get("offline_ready"), tuple(observed.get("missing") or []), status == "blocked")
        if observed_signature != signature:
            observed_signature = signature
            capture_path = current.get("observation_path") or ""
            missing = observed.get("missing") or []
            status_text = ("Missing context: " + ", ".join(missing) if missing else
                           "Ready to prepare from saved observations" if observed.get("offline_ready") else
                           "Observations saved" if observed else "No observations captured")
            capture_status.text = (f"{status_text} · Source run: {observed.get('run_id') or 'not recorded'} · "
                                   f"Capture time: {observed.get('captured_at') or 'not recorded'}")
            download_observations.set_visibility(bool(capture_path))
            if current.get("observation_path"):
                load_options()
        for key, stage in stages.items():
            if outline_only and key == "primitive":
                stage["status"].text = f"Primitive composition is not supported for {current['scenario']} yet. This case stops at outline review."
            elif offline and key == "outline":
                stage["status"].text = (current.get("outline_generate_reason") or
                    ("Generate starts a fresh attempt from the selected observations." if status == "stopped" else
                     "Input prepared — not sent. Generate starts the LLM call."))
            else:
                stage["status"].text = current[f"{key}_status"]
            stage["turn_view"].update(current, key, files.get(key, []) if files else [])
        records = current.get("execution") or []
        execution.text = (f"Recovery execution is not supported for {current['scenario']} yet. This case stops at outline review."
                          if outline_only else "Outline validation does not authorize execution. Complete composition validation and runtime CCA integration remain pending."
                          if validated_outline else "\n".join(f"{row['resource_id']} · {row['outline_id']} · {row['status']}" for row in records)
                          or "Ready after both generation stages complete.")

    update()
    load_options()

    async def poll() -> None:
        current = bridge.get_part_slippage_example()
        def captured_files() -> dict:
            directory = current.get("artifact_directory")
            if not directory:
                return {}
            return {key: recovery_turns(Path(current.get(f"{key}_artifact_directory") or Path(directory) / stage["artifact_stage"]),
                                        stage["artifact_stage"]) for key, stage in stages.items()}
        files = await asyncio.to_thread(captured_files)
        if refresh.active():
            latest = bridge.get_part_slippage_example()
            # File reads may finish after a selection or generation changes.
            if any(latest.get(key) != current.get(key) for key in (
                "source", "selected_observation_path", "run_id", "artifact_directory",
                "outline_artifact_directory", "primitive_artifact_directory",
            )):
                files = {}
            update(latest, files)

    refresh.timer(2, poll)
