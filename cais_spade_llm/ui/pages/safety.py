"""Safety page: safety requirement editing + preview generation + intent approval."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path

from nicegui import ui, events

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat


_SAFETY_DIR = Path("cais_spade_llm/specification/safety")


def render(bridge: SystemBridge) -> None:
    ui.add_head_html(
        """
        <style>
        .compact-upload { min-height: auto !important; }
        .compact-upload .q-uploader__list { display: none !important; }
        .compact-upload .q-uploader__header { min-height: auto !important; padding: 6px 8px !important; }
        </style>
        """
    )
    ui.label("Safety").classes("text-2xl font-bold px-6 pt-6")

    with ui.row().classes("w-full px-6 gap-6 items-start flex-nowrap"):
      with ui.column().classes("flex-grow gap-6 min-w-0"):

        # ── Safety Requirements + Preview/Approval ───────────────────
        _render_safety_requirements_card(bridge)

      # ── Right column: chat panel ──────────────────────────
      with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
        render_chat(bridge, agent_jid="cca", title="Central Controller Agent Chat")


def _render_safety_requirements_card(bridge: SystemBridge) -> None:
    with ui.card().classes("w-full"):
        ui.label("Safety Requirements").classes("text-lg font-semibold mb-2")
        ui.label("Text files that define safety constraints for the system.").classes(
            "text-xs text-slate-500 mb-3"
        )

        _SAFETY_DIR.mkdir(parents=True, exist_ok=True)
        status_label = ui.label("").classes("text-sm")
        preview_status_label = ui.label("Safety rule preview is not generated yet.").classes(
            "text-xs text-slate-600 mt-1"
        )
        intent_status_label = ui.label("").classes("text-xs text-slate-600")
        intent_buttons = {"approve": None, "revoke": None}
        preview_state: dict[str, dict] = {"payload": {"available": False, "rules": []}}
        preview_rules_cache: dict[str, dict] = {}
        selected_preview_rule = {"id": ""}
        preview_generation_state = {"busy": False}
        verification_lock_state = {"locked": False}
        refinement_feedback_drafts: dict[str, str] = {}
        active_feedback_file = {"path": ""}

        with ui.row().classes("w-full gap-2 items-end flex-nowrap"):
            req_select = ui.select(
                {},
                label="Requirement File",
            ).classes("w-56 shrink-0")
            name_input = ui.input(label="New file", placeholder="e.g. safety_case3").classes("w-56 shrink-0")
            create_btn = ui.button("Create", icon="add").props("flat")

        upload_widget = None
        with ui.row().classes("w-full"):
            upload_widget = ui.upload(
                label="Upload .txt",
                auto_upload=True,
                on_upload=lambda e: _handle_upload(e),
                max_file_size=1_000_000,
            ).props("accept=.txt flat dense max-files=1 hide-upload-progress").classes("w-40 compact-upload")
        upload_info_label = ui.label("").classes("text-xs text-slate-600")

        req_editor = ui.textarea(label="Edit requirements").classes(
            "w-full font-mono"
        ).props("outlined autogrow")
        ui.label(
            "Optional refinement feedback: explain what was wrong in the previous preview and how it should change."
        ).classes("text-xs text-slate-500")
        refinement_feedback = ui.textarea(label="Refinement Feedback").classes(
            "w-full font-mono"
        ).props("outlined autogrow")

        with ui.row().classes("w-full mt-2 items-start justify-between"):
            with ui.row().classes("gap-2 items-center"):
                generate_btn = ui.button(
                    "Generate Safety Rule",
                    icon="auto_fix_high",
                ).props("flat color=primary")
                regenerate_btn = ui.button(
                    "Regenerate With Feedback",
                    icon="restart_alt",
                ).props("flat color=secondary")
                generate_loading_row = ui.row().classes("items-center gap-2 text-primary")
                with generate_loading_row:
                    ui.spinner(size="sm")
                    ui.label("Generating safety rule...")
                generate_loading_row.style("display:none;")
            with ui.column().classes("items-end gap-2"):
                with ui.row().classes("gap-2 items-end"):
                    save_btn = ui.button("Save", on_click=lambda: _save_req(), icon="save").props("color=primary")
                    reload_btn = ui.button("Reload", on_click=lambda: _load_req(), icon="refresh").props("flat")
                    delete_btn = ui.button("Delete", on_click=lambda: _delete_req(), icon="delete").props("flat color=red")
                with ui.row().classes("gap-2 items-end"):
                    intent_buttons["approve"] = ui.button(
                        "Verify Safety",
                        on_click=lambda: _approve_intent(),
                        icon="task_alt",
                    ).props("flat color=green")
                    intent_buttons["revoke"] = ui.button(
                        "Unverify Safety",
                        on_click=lambda: _revoke_intent(),
                        icon="remove_done",
                    ).props("flat color=orange")

        with ui.card().classes("w-full bg-slate-50 mt-3"):
            ui.label("Generated Safety Rule Preview").classes("text-base font-semibold mb-2")
            ui.label(
                "Generate first, review LTLf and DFA, then verify safety."
            ).classes("text-xs text-slate-600 mb-2")

            preview_failure_card = ui.card().classes("w-full bg-red-50 border border-red-200 mb-2")
            preview_failure_card.style("display:none;")
            with preview_failure_card:
                ui.label("Latest Preview Failure").classes("text-sm font-semibold text-red-800")
                preview_failure_title = ui.label("").classes("text-sm font-semibold text-red-900")
                preview_failure_meta = ui.label("").classes("text-xs text-red-700 whitespace-pre-wrap")
                preview_failure_summary = ui.label("").classes(
                    "w-full text-sm text-red-900 whitespace-pre-wrap"
                )
                ui.label("Why It Failed").classes("text-xs font-semibold text-red-800 mt-1")
                preview_failure_details = ui.label("").classes(
                    "w-full text-xs text-red-900 whitespace-pre-wrap"
                )
                ui.label("How To Refine It").classes("text-xs font-semibold text-red-800 mt-1")
                preview_failure_suggestions = ui.label("").classes(
                    "w-full text-xs text-red-900 whitespace-pre-wrap"
                )
                ui.label("Raw Error").classes("text-xs font-semibold text-red-800 mt-1")
                preview_failure_raw = ui.code("", language="text").classes("w-full")

            ui.label("Generated Rule Interpretation").classes("text-sm font-semibold")
            preview_interpretation_summary = ui.label(
                "Generated rule interpretation will appear after preview generation."
            ).classes(
                "w-full text-sm text-slate-700 whitespace-pre-wrap bg-white border border-slate-200 rounded px-3 py-2 mb-2"
            )

            preview_rules_table = ui.table(
                columns=[
                    {"name": "id", "label": "Rule ID", "field": "id", "sortable": True},
                    {"name": "constraint_type", "label": "Type", "field": "constraint_type"},
                    {"name": "raw_text", "label": "Rule Text", "field": "raw_text"},
                    {"name": "ltlf", "label": "LTLf Formula", "field": "ltlf"},
                ],
                rows=[],
                row_key="id",
                selection="single",
            ).classes("w-full")

            ui.label("DFA Graphs").classes("text-sm font-semibold mt-2")
            preview_dfa_gallery = ui.row().classes("w-full gap-4 items-start flex-wrap")
            ui.label("Generated LTLf").classes("text-sm font-semibold mt-2")
            preview_ltlf = ui.code("No rule selected.", language="text").classes("w-full")
            ui.label("Latest Refinement Feedback").classes("text-sm font-semibold mt-2")
            preview_refinement_feedback = ui.code(
                "No refinement feedback recorded for this preview.",
                language="text",
            ).classes("w-full")
            ui.label("Previous vs Current Preview").classes("text-sm font-semibold mt-2")
            preview_diff_summary = ui.code(
                "No previous preview comparison available.",
                language="text",
            ).classes("w-full")
            ui.label("AP Mapping").classes("text-sm font-semibold mt-2")
            preview_ap_map = ui.code("{}", language="json").classes("w-full")
            ui.label("DFA Status").classes("text-sm font-semibold mt-2")
            preview_dfa_status = ui.label("No DFA preview generated yet.").classes("text-sm text-slate-700")
            ui.label("DFA Transitions").classes("text-sm font-semibold mt-2")
            preview_dfa_transitions = ui.table(
                columns=[
                    {"name": "from", "label": "From", "field": "from"},
                    {"name": "condition", "label": "Condition", "field": "condition"},
                    {"name": "to", "label": "To", "field": "to"},
                ],
                rows=[],
                row_key="condition",
            ).classes("w-full")
            ui.label("DFA DOT").classes("text-sm font-semibold mt-2")
            preview_dfa_dot = ui.code("No DFA DOT generated yet.", language="text").classes("w-full")
            ui.label("Preview History").classes("text-sm font-semibold mt-2")
            preview_history_table = ui.table(
                columns=[
                    {"name": "preview_id", "label": "Preview ID", "field": "preview_id"},
                    {"name": "generated_at_utc", "label": "Generated", "field": "generated_at_utc"},
                    {"name": "parent_preview_id", "label": "Parent", "field": "parent_preview_id"},
                    {"name": "refinement_feedback", "label": "Feedback", "field": "refinement_feedback"},
                ],
                rows=[],
                row_key="preview_id",
            ).classes("w-full")

        def _intent_reason_text(reason: str) -> str:
            mapping = {
                "approved": "verified",
                "not_approved": "not verified yet",
                "revoked": "verification revoked",
                "content_changed_since_approval": "content changed after verification",
                "safety_file_empty": "file is empty",
                "safety_file_missing": "file missing",
            }
            return mapping.get(str(reason or "").strip(), str(reason or "unknown"))

        def _is_selected_verified() -> bool:
            selected = str(req_select.value or "").strip()
            if not selected:
                return False
            try:
                return bool(bridge.evaluate_safety_intent_approval(selected).get("approved", False))
            except Exception:
                return False

        def _set_preview_generation_busy(is_busy: bool) -> None:
            preview_generation_state["busy"] = bool(is_busy)
            generate_loading_row.style("display:flex;" if is_busy else "display:none;")
            _refresh_intent_status()

        def _preview_reason_text(reason: str) -> str:
            mapping = {
                "ok": "ready",
                "not_generated": "not generated",
                "preview_artifacts_missing": "preview artifacts missing",
                "safety_file_missing": "safety file missing",
            }
            return mapping.get(str(reason or "").strip(), str(reason or "unknown"))

        def _png_data_url(path_value: str) -> str:
            raw = str(path_value or "").strip()
            if not raw:
                return ""
            p = Path(raw)
            if not p.exists():
                return ""
            try:
                encoded = base64.b64encode(p.read_bytes()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            except Exception:
                return ""

        def _clear_rule_detail() -> None:
            preview_ltlf.content = "No rule selected."
            preview_ap_map.content = "{}"
            preview_dfa_status.text = "No DFA preview generated yet."
            preview_dfa_status.classes(replace="text-sm text-slate-700")
            preview_dfa_transitions.rows = []
            preview_dfa_dot.content = "No DFA DOT generated yet."

        def _bulleted_text(items: object, fallback: str) -> str:
            if not isinstance(items, list):
                return fallback
            lines = [f"- {str(item).strip()}" for item in items if str(item).strip()]
            return "\n".join(lines) if lines else fallback

        def _clear_preview_failure() -> None:
            preview_failure_card.style("display:none;")
            preview_failure_title.text = ""
            preview_failure_meta.text = ""
            preview_failure_summary.text = ""
            preview_failure_details.text = ""
            preview_failure_suggestions.text = ""
            preview_failure_raw.content = ""

        def _set_preview_failure(failure: object) -> None:
            if not isinstance(failure, dict) or not failure:
                _clear_preview_failure()
                return
            generated_at = str(failure.get("generated_at_utc", "")).strip()
            stale_failure = not bool(failure.get("hash_matches_current", False))
            meta_lines: list[str] = []
            if generated_at:
                meta_lines.append(f"Failed at: {generated_at}")
            if stale_failure:
                meta_lines.append(
                    "This diagnostic is from an older version of the file. Regenerate after edits to refresh it."
                )
            preview_failure_title.text = str(
                failure.get("title", "") or "Safety preview generation failed"
            )
            preview_failure_meta.text = "\n".join(meta_lines)
            preview_failure_summary.text = str(
                failure.get("summary", "")
                or "The generated safety rule could not be grounded to the current catalog."
            )
            preview_failure_details.text = _bulleted_text(
                failure.get("details"),
                "No structured diagnostic details are available for this failure.",
            )
            preview_failure_suggestions.text = _bulleted_text(
                failure.get("suggestions"),
                "Refine the requirement or feedback, then regenerate the preview.",
            )
            preview_failure_raw.content = str(failure.get("raw_error", "") or "(empty)")
            preview_failure_card.style("display:block;")

        def _render_preview_dfa_gallery(rules: list[dict[str, object]]) -> None:
            preview_dfa_gallery.clear()
            with preview_dfa_gallery:
                if not rules:
                    ui.label("No DFA graphs generated yet.").classes("text-sm text-slate-500 italic")
                    return
                for rule in rules:
                    if not isinstance(rule, dict):
                        continue
                    rid = str(rule.get("id", "") or "rule")
                    dfa_status = str(rule.get("dfa_status", "") or "").strip()
                    dfa_diagnostic = str(rule.get("dfa_diagnostic", "") or "").strip()
                    data_url = _png_data_url(str(rule.get("dfa_png_path", "")))
                    with ui.card().classes("bg-white border border-slate-200 w-[calc(50%-0.5rem)] min-w-[18rem]"):
                        ui.label(rid).classes("text-sm font-semibold")
                        if data_url and dfa_status == "ok":
                            ui.html(
                                f"<img src='{data_url}' style='max-width:100%;height:auto;"
                                "border:1px solid #e2e8f0;border-radius:8px;' />"
                            ).classes("w-full")
                        else:
                            ui.label(dfa_diagnostic or "DFA image unavailable.").classes(
                                "text-sm text-slate-500 italic whitespace-pre-wrap"
                            )

        def _clear_preview_detail() -> None:
            _clear_rule_detail()
            _render_preview_dfa_gallery([])
            preview_interpretation_summary.text = (
                "Generated rule interpretation will appear after preview generation."
            )
            preview_refinement_feedback.content = "No refinement feedback recorded for this preview."
            preview_diff_summary.content = "No previous preview comparison available."
            preview_history_table.rows = []

        def _set_preview_rule_detail(rule_id: str) -> None:
            rid = str(rule_id or "").strip()
            rule = preview_rules_cache.get(rid)
            if not rule:
                _clear_rule_detail()
                return
            preview_ltlf.content = str(rule.get("ltlf", "") or "(empty)")
            preview_ap_map.content = json.dumps(rule.get("aps", []), indent=2)
            dfa_status = str(rule.get("dfa_status", "") or "").strip()
            dfa_diagnostic = str(rule.get("dfa_diagnostic", "") or "").strip()
            if dfa_status == "ok":
                preview_dfa_status.text = "DFA generated successfully."
                preview_dfa_status.classes(replace="text-sm text-green-700")
            else:
                preview_dfa_status.text = dfa_diagnostic or "DFA preview unavailable."
                preview_dfa_status.classes(replace="text-sm text-amber-700")
            transitions = rule.get("dfa_transitions", [])
            preview_dfa_transitions.rows = transitions if isinstance(transitions, list) else []
            preview_dfa_dot.content = str(rule.get("dfa_dot", "") or "No DFA DOT generated.")

        def _refresh_intent_status() -> None:
            selected = str(req_select.value or "").strip()
            approve_btn = intent_buttons.get("approve")
            revoke_btn = intent_buttons.get("revoke")

            if not selected:
                intent_status_label.text = "Safety intent: no file selected."
                intent_status_label.classes(replace="text-xs text-slate-600")
                verification_lock_state["locked"] = False
                save_btn.set_enabled(False)
                reload_btn.set_enabled(False)
                delete_btn.set_enabled(False)
                generate_btn.set_enabled(False)
                regenerate_btn.set_enabled(False)
                create_btn.set_enabled((not bridge.system_running) and (not preview_generation_state["busy"]))
                if upload_widget is not None:
                    upload_widget.set_enabled(False)
                if approve_btn:
                    approve_btn.set_enabled(False)
                if revoke_btn:
                    revoke_btn.set_enabled(False)
                return

            try:
                evaluation = bridge.evaluate_safety_intent_approval(selected)
            except Exception as exc:
                intent_status_label.text = f"Safety intent status unavailable: {exc}"
                intent_status_label.classes(replace="text-xs text-red-700")
                verification_lock_state["locked"] = False
                save_btn.set_enabled(False)
                reload_btn.set_enabled(False)
                delete_btn.set_enabled(False)
                generate_btn.set_enabled(False)
                regenerate_btn.set_enabled(False)
                create_btn.set_enabled((not bridge.system_running) and (not preview_generation_state["busy"]))
                if upload_widget is not None:
                    upload_widget.set_enabled(False)
                if approve_btn:
                    approve_btn.set_enabled(False)
                if revoke_btn:
                    revoke_btn.set_enabled(False)
                return

            preview_payload = preview_state.get("payload", {})
            preview_ready = bool(preview_payload.get("available", False)) and bool(
                preview_payload.get("hash_matches_current", False)
            )
            preview_record = preview_payload.get("record", {}) if isinstance(
                preview_payload.get("record"), dict
            ) else {}

            approved = bool(evaluation.get("approved", False))
            reason_text = _intent_reason_text(str(evaluation.get("reason", "")))
            rec = evaluation.get("record", {}) if isinstance(evaluation.get("record"), dict) else {}

            if approved:
                approved_at = str(rec.get("approved_at_utc", "")).strip()
                suffix = f" at {approved_at}" if approved_at else ""
                intent_status_label.text = (
                    f"Safety intent: VERIFIED{suffix}. "
                    "File editing is locked; click Unverify Safety to modify."
                )
                intent_status_label.classes(replace="text-xs text-green-700")
            else:
                if not preview_ready:
                    preview_reason = _preview_reason_text(str(preview_payload.get("reason", "")))
                    generated_at = str(preview_record.get("generated_at_utc", "")).strip()
                    if generated_at and not bool(preview_payload.get("hash_matches_current", False)):
                        preview_reason = "preview is stale after file edits"
                    intent_status_label.text = (
                        f"Safety intent: NOT VERIFIED ({reason_text}). "
                        f"Preview status: {preview_reason}. Generate current preview before verification."
                    )
                else:
                    intent_status_label.text = f"Safety intent: NOT VERIFIED ({reason_text})."
                intent_status_label.classes(replace="text-xs text-amber-700")

            editable = not bridge.system_running
            mutating_busy = preview_generation_state["busy"]
            verification_lock_state["locked"] = bool(approved and selected)
            can_mutate_file = editable and bool(selected) and not approved and not mutating_busy
            can_create = editable and not mutating_busy

            save_btn.set_enabled(can_mutate_file)
            reload_btn.set_enabled(can_mutate_file)
            delete_btn.set_enabled(can_mutate_file)
            create_btn.set_enabled(can_create)
            generate_btn.set_enabled(can_mutate_file)
            regenerate_btn.set_enabled(can_mutate_file)
            if upload_widget is not None:
                upload_widget.set_enabled(can_mutate_file)

            if approve_btn:
                approve_btn.set_enabled(can_mutate_file and preview_ready)
            if revoke_btn:
                revoke_btn.set_enabled(editable and bool(selected) and approved and not mutating_busy)

        def _refresh_preview() -> None:
            preview_rules_cache.clear()
            selected_preview_rule["id"] = ""
            selected = str(req_select.value or "").strip()

            if not selected:
                preview_status_label.text = "Safety rule preview is not generated yet."
                preview_status_label.classes(replace="text-xs text-slate-600 mt-1")
                preview_rules_table.rows = []
                preview_state["payload"] = {
                    "available": False,
                    "reason": "safety_file_missing",
                    "failure": {},
                    "rules": [],
                }
                _clear_preview_detail()
                _clear_preview_failure()
                _refresh_intent_status()
                return

            try:
                payload = bridge.get_safety_rule_preview(selected)
            except Exception as exc:
                preview_status_label.text = f"Safety preview unavailable: {exc}"
                preview_status_label.classes(replace="text-xs text-red-700 mt-1")
                preview_rules_table.rows = []
                preview_state["payload"] = {
                    "available": False,
                    "reason": "preview_error",
                    "failure": {},
                    "rules": [],
                }
                _clear_preview_detail()
                _clear_preview_failure()
                _refresh_intent_status()
                return

            preview_state["payload"] = payload
            available = bool(payload.get("available", False))
            _set_preview_failure(payload.get("failure"))
            reason_text = _preview_reason_text(str(payload.get("reason", "")))
            if not available:
                if isinstance(payload.get("failure"), dict) and payload.get("failure"):
                    preview_status_label.text = (
                        "Latest safety preview generation failed. "
                        "Review the explanation below, refine the requirement or feedback, and regenerate."
                    )
                    preview_status_label.classes(replace="text-xs text-red-700 mt-1")
                else:
                    preview_status_label.text = (
                        "Safety rule preview not ready. "
                        f"Status: {reason_text}. Click Generate Safety Rule."
                    )
                    preview_status_label.classes(replace="text-xs text-amber-700 mt-1")
                preview_rules_table.rows = []
                _clear_preview_detail()
                _refresh_intent_status()
                return

            record = payload.get("record", {}) if isinstance(payload.get("record"), dict) else {}
            generated_at = str(record.get("generated_at_utc", "")).strip()
            hash_ok = bool(payload.get("hash_matches_current", False))
            if hash_ok:
                preview_status_label.text = (
                    f"Safety rule preview generated at {generated_at or 'unknown time'} (current)."
                )
                preview_status_label.classes(replace="text-xs text-green-700 mt-1")
            else:
                preview_status_label.text = (
                    f"Safety preview generated at {generated_at or 'unknown time'}, "
                    "but file changed after generation. Regenerate preview."
                )
                preview_status_label.classes(replace="text-xs text-amber-700 mt-1")
            preview_interpretation_summary.text = str(
                payload.get("preview_interpretation_summary", "")
                or "No generated rule interpretation available for this preview."
            )
            preview_refinement_feedback.content = str(
                payload.get("refinement_feedback", "") or "No refinement feedback recorded for this preview."
            )
            preview_diff_summary.content = str(
                payload.get("diff_summary", "") or "No previous preview comparison available."
            )
            preview_history_table.rows = payload.get("history", []) if isinstance(
                payload.get("history"), list
            ) else []

            rows: list[dict] = []
            for idx, rule in enumerate(payload.get("rules", []), start=1):
                if not isinstance(rule, dict):
                    continue
                rid = str(rule.get("id", "")).strip() or f"SAFE_{idx}"
                preview_rules_cache[rid] = rule
                rows.append(
                    {
                        "id": rid,
                        "constraint_type": str(rule.get("constraint_type", "")),
                        "raw_text": str(rule.get("raw_text", "")),
                        "ltlf": str(rule.get("ltlf", "")),
                    }
                )
            preview_rules_table.rows = rows
            _render_preview_dfa_gallery(list(preview_rules_cache.values()))

            target_rid = selected_preview_rule["id"]
            if target_rid not in preview_rules_cache and rows:
                target_rid = str(rows[0].get("id", ""))
            selected_preview_rule["id"] = target_rid
            if target_rid:
                _set_preview_rule_detail(target_rid)
            else:
                _clear_rule_detail()

            _refresh_intent_status()

        def _load_req():
            prior_selected = str(active_feedback_file.get("path", "") or "").strip()
            if prior_selected:
                refinement_feedback_drafts[prior_selected] = str(refinement_feedback.value or "")
            if not req_select.value:
                req_editor.value = ""
                refinement_feedback.value = ""
                active_feedback_file["path"] = ""
                status_label.text = ""
                _refresh_preview()
                return
            path = Path(req_select.value)
            if path.exists():
                req_editor.value = path.read_text()
                status_label.text = ""
            selected = str(req_select.value or "").strip()
            refinement_feedback.value = refinement_feedback_drafts.get(selected, "")
            active_feedback_file["path"] = selected
            _refresh_preview()

        def _refresh_file_list(select_path: str | None = None):
            req_files = sorted(_SAFETY_DIR.glob("*.txt"))
            file_map = {str(f): f.stem for f in req_files}
            req_select.options = file_map
            req_select.update()
            if select_path and select_path in file_map:
                req_select.value = select_path
            elif req_files:
                req_select.value = str(req_files[0])
            else:
                req_select.value = None
                req_editor.value = ""
            _load_req()

        def _save_req():
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            if _is_selected_verified():
                ui.notify("Selected safety file is verified. Unverify Safety before editing.", type="warning")
                return
            if not req_select.value:
                status_label.text = "No file selected"
                status_label.classes(replace="text-sm text-amber-600")
                return
            path = Path(req_select.value)
            path.write_text(req_editor.value)
            status_label.text = f"Saved {path.name}"
            status_label.classes(replace="text-sm text-green-600")
            _refresh_preview()

        def _delete_req():
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            if _is_selected_verified():
                ui.notify("Selected safety file is verified. Unverify Safety before editing.", type="warning")
                return
            if not req_select.value:
                return
            path = Path(req_select.value)
            name = path.name
            bridge.delete_safety_intent_state(str(path))
            if path.exists():
                path.unlink()
            status_label.text = f"Deleted {name}"
            status_label.classes(replace="text-sm text-red-600")
            _refresh_file_list()

        _REQ_FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*\.txt$")

        def _normalized_upload_name(raw_name: str) -> str:
            base_name = Path(str(raw_name or "").strip()).name
            if not base_name:
                base_name = "uploaded"
            stem = Path(base_name).stem if Path(base_name).suffix else base_name
            stem = stem.lower()
            stem = re.sub(r"\s+", "_", stem)
            stem = re.sub(r"[^a-z0-9_-]", "_", stem)
            stem = re.sub(r"_+", "_", stem).strip("_-")
            if not stem:
                stem = "uploaded"
            normalized = f"{stem}.txt"
            if not _REQ_FILENAME_RE.fullmatch(normalized):
                normalized = "uploaded.txt"
            return normalized

        def _next_available_upload_path(filename: str) -> Path:
            candidate = _SAFETY_DIR / filename
            if not candidate.exists():
                return candidate
            stem = candidate.stem or "uploaded"
            suffix = candidate.suffix or ".txt"
            idx = 1
            while True:
                alt = _SAFETY_DIR / f"{stem}_{idx}{suffix}"
                if not alt.exists():
                    return alt
                idx += 1

        async def _handle_upload(e: events.UploadEventArguments):
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            if _is_selected_verified():
                ui.notify("Selected safety file is verified. Unverify Safety before editing.", type="warning")
                return
            content = await e.file.read()
            original_name = str(getattr(e.file, "name", "") or "uploaded.txt")
            uploaded_name = _normalized_upload_name(original_name)
            dest = _next_available_upload_path(uploaded_name)
            dest.write_bytes(content)
            status_label.text = f"Uploaded as new file: {dest.name}"
            status_label.classes(replace="text-sm text-green-600")
            _refresh_file_list(str(dest))
            req_select.value = str(dest)
            _load_req()
            req_editor.update()
            if dest.name != original_name:
                upload_info_label.text = f"Uploaded: {original_name} -> {dest.name}"
            else:
                upload_info_label.text = f"Uploaded: {dest.name}"
            if upload_widget is not None:
                try:
                    upload_widget.reset()
                except Exception:
                    pass

        async def _create_new():
            if bridge.system_running:
                ui.notify("Cannot edit while system is running", type="warning")
                return
            name_input.value = name_input.value.strip()
            if not name_input.value:
                status_label.text = "Enter a file name first"
                status_label.classes(replace="text-sm text-amber-600")
                return
            original_name = name_input.value
            name = _normalized_upload_name(original_name)
            dest = _SAFETY_DIR / name
            if dest.exists():
                status_label.text = f"{name} already exists — select it to edit"
                status_label.classes(replace="text-sm text-amber-600")
                return
            dest.write_text("[Safety Requirements]\n- ")
            if name != original_name and name != f"{original_name}.txt":
                status_label.text = f"Created {name} (normalized from \"{original_name}\")"
            else:
                status_label.text = f"Created {name}"
            status_label.classes(replace="text-sm text-green-600")
            name_input.value = ""
            _refresh_file_list(str(dest))

        async def _generate_preview(use_feedback: bool = False) -> None:
            if preview_generation_state["busy"]:
                return
            if bridge.system_running:
                ui.notify("Cannot generate safety preview while system is running", type="warning")
                return
            if _is_selected_verified():
                ui.notify("Selected safety file is verified. Unverify Safety before regenerating.", type="warning")
                return
            selected = str(req_select.value or "").strip()
            if not selected:
                ui.notify("Select a safety requirement file first.", type="warning")
                return
            selected_path = Path(selected)
            try:
                selected_path.write_text(str(req_editor.value or ""), encoding="utf-8")
                status_label.text = f"Saved {selected_path.name}"
                status_label.classes(replace="text-sm text-green-600")
            except Exception as exc:
                ui.notify(f"Failed to save safety requirements before preview generation: {exc}", type="negative")
                return
            feedback_text = ""
            parent_preview_id = ""
            if use_feedback:
                feedback_text = str(refinement_feedback.value or "").strip()
                refinement_feedback_drafts[selected] = feedback_text
                if not feedback_text:
                    ui.notify("Enter refinement feedback before regenerating.", type="warning")
                    return
                current_record = preview_state.get("payload", {}).get("record", {})
                if isinstance(current_record, dict):
                    parent_preview_id = str(current_record.get("preview_id", "")).strip()
            _set_preview_generation_busy(True)
            preview_status_label.text = (
                "Regenerating safety rule preview with refinement feedback..."
                if use_feedback
                else "Generating safety rule preview..."
            )
            preview_status_label.classes(replace="text-xs text-blue-700 mt-1")
            try:
                ui.notify(
                    "Regenerating safety rule preview with feedback..."
                    if use_feedback
                    else "Generating safety rule preview...",
                    type="info",
                )
                await asyncio.to_thread(
                    bridge.generate_safety_rule_preview,
                    selected,
                    refinement_feedback=feedback_text,
                    parent_preview_id=parent_preview_id,
                )
                ui.notify(
                    "Safety rule preview regenerated with feedback."
                    if use_feedback
                    else "Safety rule preview generated.",
                    type="positive",
                )
            except Exception as exc:
                ui.notify(
                    "Safety preview generation failed. Review the explanation box below.",
                    type="negative",
                )
            finally:
                _set_preview_generation_busy(False)
                _refresh_preview()

        async def _generate_initial_preview() -> None:
            await _generate_preview(False)

        async def _regenerate_preview_with_feedback() -> None:
            await _generate_preview(True)

        def _approve_intent() -> None:
            if bridge.system_running:
                ui.notify("Cannot verify while system is running", type="warning")
                return
            selected = str(req_select.value or "").strip()
            if not selected:
                ui.notify("Select a safety requirement file first.", type="warning")
                return
            try:
                out = bridge.approve_safety_intent(selected)
                selected_path = Path(str(out.get("safety_file", selected)))
                ui.notify(f"Safety intent verified: {selected_path.name}", type="positive")
            except Exception as exc:
                ui.notify(f"Safety intent verification failed: {exc}", type="negative")
            finally:
                _refresh_preview()

        def _revoke_intent() -> None:
            if bridge.system_running:
                ui.notify("Cannot unverify while system is running", type="warning")
                return
            selected = str(req_select.value or "").strip()
            if not selected:
                ui.notify("Select a safety requirement file first.", type="warning")
                return
            try:
                out = bridge.revoke_safety_intent_approval(selected)
                selected_path = Path(str(out.get("safety_file", selected)))
                ui.notify(f"Safety intent unverified: {selected_path.name}", type="positive")
            except Exception as exc:
                ui.notify(f"Failed to unverify safety intent: {exc}", type="negative")
            finally:
                _refresh_preview()

        def _on_preview_rule_select(e):
            selected_preview_rule["id"] = ""
            for row in (e.selection or []):
                rid = str(row.get("id", "")).strip()
                if rid:
                    selected_preview_rule["id"] = rid
                    break
            _set_preview_rule_detail(selected_preview_rule["id"])

        req_select.on_value_change(_load_req)
        preview_rules_table.on_select(_on_preview_rule_select)
        create_btn.on_click(_create_new)
        generate_btn.on_click(_generate_initial_preview)
        regenerate_btn.on_click(_regenerate_preview_with_feedback)

        _refresh_file_list()

        def _update_readonly():
            readonly = bridge.system_running or preview_generation_state["busy"] or verification_lock_state["locked"]
            req_editor.props(f"readonly={str(readonly).lower()}")
            refinement_feedback.props(f"readonly={str(readonly).lower()}")
            _refresh_intent_status()

        ui.timer(2.0, _update_readonly)
