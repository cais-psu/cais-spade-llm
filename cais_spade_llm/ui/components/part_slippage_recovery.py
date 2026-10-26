from __future__ import annotations

"""Operator controls and exact prompt captures for the Part slippage run."""

import asyncio
import json
from pathlib import Path

from nicegui import ui

from cais_spade_llm.recovery_framework import ROOT
from cais_spade_llm.recovery_framework.capture_files import BELIEFS_NAME
from cais_spade_llm.ui.components.recovery_turns import RecoveryTurns, render_composition_programs
from cais_spade_llm.ui.evidence import BackgroundSection
from cais_spade_llm.ui.recovery_evidence import recovery_turns
from cais_spade_llm.ui.recovery_setup import FAILURE_SCENARIOS
from cais_spade_llm.ui.refresh import PageRefresh


def _value_text(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, dict) and all(isinstance(value.get(axis), (int, float)) for axis in "xy"):
        return "(" + ", ".join(f"{key} {number:.3f}" for key, number in value.items() if isinstance(number, (int, float))) + ")"
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return ", ".join(" ".join(str(part) for part in item.values() if part not in (None, "")) for item in value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def draft_rows(draft: dict) -> list[dict]:
    """Rows of the recovery table: one per LLM recovery event, its end state listed per variable.

    The LLM's values are listed as written; a value code added says so, with its reason. The nominal
    steps that connect to the remaining plan are not recovery events; the handoff record shows them.
    """
    rows = []
    for index, row in enumerate(draft.get("transition_trace", []), 1):
        changes = row.get("condition_changes")
        if changes is None:
            # Drafts recorded before per-variable attribution list the LLM's end state as written.
            changes = [{"entity": row.get("resource") or row.get("resource_jid"), "variable": name, "value": value,
                        "by": "LLM"} for name, value in (row.get("expected_end_state") or {}).items()]
        # Drafts saved before claims left condition_changes list them as "interrupted <task>" rows.
        changes = [change for change in changes if not str(change.get("entity", "")).startswith("interrupted ")]
        lines = [f"{change.get('entity')}  {change.get('variable')} = {_value_text(change.get('value'))}"
                 + ("" if change.get("by") == "LLM" else f"  — {change.get('by')}"
                    + (f" ({change['reason']})" if change.get("reason") else "")) for change in changes]
        rows.append({"step": index, "event_name": row.get("event_name"),
                     "resource": row.get("resource") or row.get("resource_jid"), "part_name": row.get("part_name"),
                     "end_state": "\n".join(lines), "rationale": row.get("rationale") or ""})
    return rows


def render_part_slippage(bridge, *, is_active=lambda: True) -> None:  # noqa: C901, PLR0915 - shared page controls
    """Render one bridge-owned workflow; navigation only stops page polling."""
    state = bridge.get_part_slippage_example()
    default_reader = getattr(bridge, "get_default_recovery_observations", None)
    default_observations = default_reader() if callable(default_reader) else {}
    if not isinstance(default_observations, dict):
        default_observations = {}
    scenario = state.get("scenario", "Part slippage")
    if ((state["status"] == "idle" or state.get("source") == "offline") and not state.get("busy")
            and not state.get("defer_preparation") and scenario in default_observations):
        state = bridge.set_failure_example_options(source="offline", scenario=scenario,
            safety_files=state.get("selected_safety_files", []), observation_path=default_observations[scenario],
            defer_preparation=True)
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
    composition_signature = None
    available_safety = []
    selection_revision = state.get("selection_revision", 0)
    saved_options = {path: f"{scenario} · {Path(path).parent.name} · Verified capture"
                     for scenario, path in default_observations.items()}
    saved_records = {path: {"source_path": path, "scenario": scenario}
                     for scenario, path in default_observations.items()}

    async def act(action: str, feedback: str = "") -> None:
        try:
            if action == "start":
                await bridge.start_failure_example()
            else:
                if action == "generate_outline" and fields["source"].value == "offline":
                    action = "generate_outline_fresh"
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
            controls["stop"] = ui.button("Stop", icon="stop", on_click=lambda: act("stop")).props("outline")
            controls["reset"] = ui.button("Reset example", icon="restart_alt", on_click=lambda: act("reset")).props("flat")
        capture_status = ui.label().classes("text-sm text-slate-500")
        capture_path = ""
        with ui.row().classes("items-center gap-2"):
            download_observations = ui.button("Download observations", icon="download",
                on_click=lambda: ui.download.file(capture_path)).props("flat dense")
            download_beliefs = ui.button("Download beliefs", icon="download",
                on_click=lambda: ui.download.file(Path(capture_path).with_name(BELIEFS_NAME))).props("flat dense")
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
            if fields["source"].value == "offline":
                observation_path = default_observations.get(fields["scenario"].value, observation_path)
            selected = bridge.set_failure_example_options(
                source=fields["source"].value, scenario=fields["scenario"].value,
                safety_files=fields["safety"].value or [], observation_path=observation_path,
                defer_preparation=fields["source"].value == "offline")
        except (ValueError, OSError) as exc:
            ui.notify(str(exc), type="warning")
        update()

    for field in fields.values():
        field.on_value_change(options_changed)

    def show_options(safety) -> None:
        nonlocal available_safety
        available_safety = safety if isinstance(safety, list) else []
        update()

    option_reader = BackgroundSection()

    def load_options() -> None:
        option_reader.load(bridge.list_safety_requirement_files, show_options)

    def mode_changed() -> None:
        if updating or recovery_mode.value is None:
            return
        try:
            bridge.set_part_slippage_modes(recovery_mode.value, recovery_mode.value)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
        update()

    def ra_check_changed() -> None:
        if updating:
            return
        try:
            bridge.set_recovery_ra_feasibility_check(ra_check.value)
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
            if key == "outline":
                ra_check = ui.switch("RA feasibility check", value=bool(state.get("check_ra_feasibility")))
                ra_check.tooltip("Checks each proposed event with Alg. 1 and Eq. 12 before it is applied; "
                                 "a refusal returns to the LLM. Offline only; CCA checks stay off.")
            with ui.row():
                generate_action = "generate_outline" if key == "outline" else "generate_primitives"
                controls[generate_action] = ui.button("Generate", on_click=lambda action=generate_action: act(action)).props("outline")
                if key == "outline":
                    controls["continue_outline"] = ui.button("Continue", on_click=lambda: act("continue_outline"))
                stop_action = "stop_outline" if key == "outline" else "stop_primitives"
                controls[stop_action] = ui.button("Stop", icon="stop", on_click=lambda: act("stop_generation")).props(
                    "outline color=negative")
                controls[stop_action].tooltip("Stops this generation after the current LLM call. Gazebo keeps running.")
            if key == "outline":
                controls["prepare"] = ui.button("Retry input preparation", on_click=lambda: act("prepare")).props("outline")
            turn_view = RecoveryTurns(ui.column().classes("w-full"))
            if key == "outline":
                draft_status = ui.label().classes("text-amber-700")
                draft_table = ui.table(columns=[
                    {"name": name, "label": label, "field": name, "align": "left"}
                    for name, label in (("step", "Step"), ("event_name", "Event"), ("resource", "Resource"),
                        ("part_name", "Part"), ("end_state", "End state"), ("rationale", "Rationale"))
                ], rows=[], row_key="step").classes("w-full").props("wrap-cells").style("white-space: pre-line")
                draft_download = ui.button("Download draft", icon="download",
                    on_click=lambda: ui.download.file(bridge.get_part_slippage_example()["draft_path"])).props("flat dense")
            if key == "primitive":
                composition_status = ui.label().classes("text-sm font-semibold")
                composition_programs = ui.column().classes("w-full")
                composition_download = ui.button("Download composition evidence", icon="download",
                    on_click=lambda: ui.download.file(Path(bridge.get_part_slippage_example()["artifact_directory"]) / "composition.json")).props("flat dense")
            if key == "primitive":
                with ui.expansion("Feedback and Regenerate").classes("w-full"):
                    feedback = ui.textarea("Optional feedback").classes("w-full").props("autogrow")
                    controls["regenerate_primitives"] = ui.button("Regenerate", on_click=lambda field=feedback: act(
                        "regenerate_primitives", field.value or "")).props("outline")
            stages[key] = {"status": status, "artifact_stage": artifact_stage,
                           "key": key,
                           "card": stage_card,
                           "turn_view": turn_view}
    recovery_mode.on_value_change(mode_changed)
    ra_check.on_value_change(ra_check_changed)

    with ui.card().classes("w-full") as execution_card:
        ui.label("Execution").classes("text-lg font-semibold")
        controls["execute"] = ui.button("Execute recovery", icon="play_arrow", on_click=lambda: act("execute"))
        execution = ui.label().classes("text-sm whitespace-pre-wrap")

    def update(current=None, files=None) -> None:
        nonlocal updating, observed_signature, capture_path, provenance_path, composition_signature, selection_revision
        current = current if current is not None else bridge.get_part_slippage_example()
        if current.get("selection_revision", 0) < selection_revision:
            return
        selection_revision = current.get("selection_revision", selection_revision)
        status = current["status"]
        offline = current.get("source") == "offline"
        draft = current.get("outline_draft")
        review_only = bool(draft) or current.get("composition_validation_version") == 1
        readiness = current.get("composition_readiness", {})
        workflow_label.text = ("Saved observations → Generate → Review draft" if offline
                               else "Start failure → Capture observations")
        safety_badge.text = ("Offline — saved observations; safety checks deferred" if offline
                             else "Gazebo — safety checks deferred")
        safety_note.text = "Selected specifications are included in the prompt. Formal safety checks are skipped."
        if offline and current.get("defer_preparation"):
            safety_note.text = "Generate runs these selections in a fresh attempt."
        if draft:
            safety_badge.text = ("RA-checked outline — RA feasibility only, no CCA checks" if draft.get("ra_feasibility_check")
                                 else "Unvalidated outline — no validation checks")
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
            "stop_outline": current.get("generating_stage") == "outline",
            "stop_primitives": current.get("generating_stage") == "primitive",
            "prepare": status == "blocked" and bool(current.get("observation_path")),
            "reset": status not in {"starting", "resetting"} and not current.get("busy"),
            "generate_outline": current.get("can_generate_outline", status == "outline_ready"),
            "continue_outline": not offline and status == "outline_review" and readiness.get("ready", False),
            "generate_primitives": status == "primitive_ready" or status == "outline_review" and readiness.get("ready", False),
            "execute": status == "ready_to_execute" and not review_only,
            "regenerate_primitives": (status in {"primitive_ready", "ready_to_execute", "offline_completed", "blocked"}
                                      and bool(current.get("outline")) and not current.get("imported_recovery")),
        }
        if draft:
            for name in ("continue_outline", "generate_primitives", "regenerate_primitives", "execute", "prepare"):
                available[name] = False
        for key, button in controls.items():
            button.set_enabled(available[key] and (key in {"stop", "stop_outline", "stop_primitives"} or not current.get("busy")))
        controls["generate_outline"].set_visibility(offline)
        controls["continue_outline"].set_visibility(not offline and current["outline_mode"] == "manual")
        controls["generate_primitives"].set_visibility(offline or current["primitive_mode"] == "manual"
                                                       or current.get("primitive_status") == "stopped")
        controls["start"].set_visibility(not offline)
        controls["execute"].set_visibility(not offline)
        controls["prepare"].set_visibility(not offline and available["prepare"])
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
        if not scenario_options and selected_observation:
            scenario_options = {selected_observation: f"{current['scenario']} · {Path(selected_observation).parent.name}"}
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
        ra_check.value = bool(current.get("check_ra_feasibility"))
        ra_check.set_visibility(offline)
        ra_check.set_enabled(not current.get("busy"))
        for field in fields.values():
            field.set_enabled(not current.get("busy") and status not in {"running_example", "executing", "resuming"})
        if offline and status == "preparing_outline":
            fields["observations"].set_enabled(True)
            fields["source"].set_enabled(True)
        for key in ("scenario", "safety"):
            fields[key].set_enabled(not current.get("busy") and
                ((offline and current.get("defer_preparation")) or
                 (status in {"idle", "stopped", "draft_ready", "outline_ready"} and not current.get("outline"))))
        if offline:
            fields["observations"].disable()
        updating = False
        draft_status.set_visibility(bool(draft))
        draft_table.set_visibility(bool(draft))
        draft_download.set_visibility(bool(draft) and not current.get("busy"))
        if draft:
            draft_status.text = f"{draft['message']} Provider calls: {draft['provider_calls']}. Execution authorized: false."
            draft_table.rows = draft_rows(draft)
            draft_table.update()
        observed = current.get("observations") or {}
        signature = (current.get("observation_path"), current.get("selected_observation_path"),
                     observed.get("offline_ready"), tuple(observed.get("missing") or []), status == "blocked")
        if observed_signature != signature:
            observed_signature = signature
            capture_path = current.get("selected_observation_path") or current.get("observation_path") or ""
            missing = observed.get("missing") or []
            status_text = ("Missing context: " + ", ".join(missing) if missing else
                           "Ready to prepare from saved observations" if observed.get("offline_ready") else
                           "Observations saved" if observed else "No observations captured")
            capture_status.text = (f"Verified capture selected · {Path(capture_path).parent.name}. Generate prepares this input."
                                  if capture_path and not observed else
                                  f"{status_text} · Source run: {observed.get('run_id') or 'not recorded'} · "
                                   f"Capture time: {observed.get('captured_at') or 'not recorded'}")
            download_observations.set_visibility(bool(capture_path))
            download_beliefs.set_visibility(
                bool(capture_path) and Path(capture_path).with_name(BELIEFS_NAME).is_file())
        for key, stage in stages.items():
            if draft and key == "outline":
                stage["status"].text = draft["message"]
            elif offline and key == "outline":
                stage["status"].text = (current.get("outline_generate_reason") or
                    ("Generate prepares a fresh outline from the verified capture." if current.get("defer_preparation") else
                     "Generate starts a fresh attempt from the selected observations." if status == "stopped" else
                     "Input prepared — not sent. Generate starts the LLM call."))
            else:
                stage["status"].text = current[f"{key}_status"]
            stage["turn_view"].update(current, key, files.get(key, []) if files else [])
        composition_status.text = ("Composition feasibility: " + current.get("composition_feasibility", "not_checked")
            + ". Execution authorized: false. " + "; ".join(readiness.get("reasons", [])))
        signature = repr((current.get("primitives", []), current.get("composition_findings", [])))
        if signature != composition_signature:
            composition_signature = signature
            composition_programs.clear()
            with composition_programs:
                render_composition_programs(current.get("primitives", []))
                for finding in current.get("composition_findings", []):
                    ui.label(finding.get("reason", str(finding))).classes("text-amber-700 whitespace-pre-wrap")
        composition_download.set_visibility(current.get("composition_validation_version") == 1 and current.get("primitive_status") == "completed")
        records = current.get("execution") or []
        execution.text = ("Composition is for review. CCA validation of the composed trace and physical execution remain pending."
                          if review_only else "\n".join(f"{row['resource_id']} · {row['outline_id']} · {row['status']}" for row in records)
                          or "Ready after both generation stages complete.")
        if draft:
            execution.text = (("RA-checked" if draft.get("ra_feasibility_check") else "Unvalidated")
                              + " draft: composition and execution are disabled.")

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
                "selection_revision",
            )):
                files = {}
            update(latest, files)

    refresh.timer(2, poll)
