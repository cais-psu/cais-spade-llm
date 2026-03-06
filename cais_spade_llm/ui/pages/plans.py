"""Plans page: offline plan-set generation, review, editing, verification, and activation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat
from cais_spade_llm.ui.components.dag_graph import nodes_to_mermaid
from cais_spade_llm.ui.components.fsa_graph import fsa_state_index_text, fsa_to_mermaid


def render(bridge: SystemBridge) -> None:
    ui.label("Plans").classes("text-2xl font-bold px-6 pt-6")

    with ui.row().classes("w-full px-6 gap-6 items-start no-wrap"):
        with ui.column().classes("flex-1 gap-6 min-w-0"):
            with ui.card().classes("w-full"):
                ui.label("Offline Verified Plan Set Builder").classes("text-lg font-semibold mb-2")
                ui.label(
                    "Select requirement files, generate a plan set, inspect DAG/FSA, then verify before startup use."
                ).classes("text-sm text-slate-600")

                SAFETY_MODE_APPROVED = "Approved only"
                SAFETY_MODE_ALL = "All files"

                with ui.row().classes("items-start gap-4 flex-wrap mt-3"):
                    with ui.column().classes("w-96 gap-2"):
                        product_init_files = bridge.list_product_files()
                        product_init_select = ui.select(
                            {f: Path(f).stem for f in product_init_files},
                            value=product_init_files[0] if product_init_files else None,
                            label="Product",
                        ).classes("w-full")

                        requirement_select = ui.select(
                            {},
                            label="Product Requirement (.txt)",
                        ).classes("w-full")

                    with ui.column().classes("w-96 gap-2"):
                        with ui.row().classes("items-end gap-3"):
                            safety_select = ui.select(
                                {},
                                label="Safety Requirement (.txt)",
                            ).classes("w-72")
                            safety_mode_select = ui.radio(
                                [
                                    SAFETY_MODE_APPROVED,
                                    SAFETY_MODE_ALL,
                                ],
                                value=SAFETY_MODE_APPROVED,
                            ).props("inline dense")

                requirement_path_label = ui.label("").classes("text-xs text-slate-700")
                safety_path_label = ui.label("").classes("text-xs text-slate-700")
                safety_intent_gate_label = ui.label("").classes("text-xs text-slate-600")

                scope_label = ui.label("").classes("text-xs text-slate-600 mt-2")
                active_label = ui.label("Startup plan set: none").classes("text-sm text-slate-700 mt-2")
                bundle_select = ui.select({}, label="Generated Plan Sets").classes("w-full")
                status_label = ui.label("No plan set selected.").classes("text-sm text-slate-600")
                meta = ui.code("{}", language="json").classes("w-full")
                ui.label(
                    "Optional refinement feedback: explain what was wrong in the previous plan set and how it should change."
                ).classes("text-xs text-slate-500 mt-2")
                refinement_feedback = ui.textarea(label="Refinement Feedback").classes(
                    "w-full font-mono"
                ).props("outlined autogrow")

                ui.label(
                    "Unverify = move a verified plan set back to draft before editing."
                ).classes("text-xs text-slate-500")

                with ui.row().classes("gap-2 mt-3 flex-wrap"):
                    auto_replan_input = ui.number(
                        "Auto-replan max",
                        value=3,
                        min=0,
                        max=10,
                        step=1,
                    ).classes("w-40")
                    generate_btn = ui.button("Generate Plan Set", icon="bolt").props("color=primary")
                    regenerate_btn = ui.button("Regenerate With Feedback", icon="restart_alt").props(
                        "flat color=secondary"
                    )
                    verify_btn = ui.button("User Verify Plan", icon="verified").props("flat color=green")
                    unverify_btn = ui.button("Unverify Plan Set", icon="remove_done").props("flat color=orange")
                    delete_btn = ui.button("Delete Plan Set", icon="delete").props("flat color=red")
                generation_banner = ui.row().classes(
                    "w-full mt-2 items-center gap-2 rounded p-3 text-sm text-red-700 bg-red-50"
                )
                generation_banner.style("display:none;")
                with generation_banner:
                    generation_banner_icon = ui.icon("error")
                    generation_banner_label = ui.label("")
                with ui.row().classes("items-center gap-2 mt-1 text-primary"):
                    generate_loading_row = ui.row().classes("items-center gap-2")
                    with generate_loading_row:
                        ui.spinner(size="sm")
                        ui.label("Generating plan set...")
                    generate_loading_row.style("display:none;")

                bundle_cache: dict[str, dict] = {}
                bundle_generation_state = {"busy": False}
                refinement_feedback_drafts: dict[str, str] = {}
                active_feedback_scope = {"key": ""}

            with ui.card().classes("w-full"):
                ui.label("Selected Requirement Contents").classes("text-lg font-semibold mb-2")
                with ui.row().classes("w-full gap-4 items-start no-wrap"):
                    product_req_preview = ui.textarea(label="Product Requirement Content").classes(
                        "w-1/2 font-mono"
                    ).props("outlined readonly autogrow")
                    safety_req_preview = ui.textarea(label="Safety Requirement Content").classes(
                        "w-1/2 font-mono"
                    ).props("outlined readonly autogrow")

            with ui.card().classes("w-full"):
                ui.label("Generated Task DAG").classes("text-lg font-semibold mb-2")
                dag_mermaid = ui.mermaid("graph LR\n    empty[No plan set selected]").classes("w-full")

            with ui.card().classes("w-full"):
                ui.label("Compiled Global FSA / DES").classes("text-lg font-semibold mb-2")
                fsa_mermaid = ui.mermaid("graph TB\n    empty[No plan set selected]").classes("w-full")
                fsa_index = ui.code("No global FSA loaded", language="text").classes("w-full")

        with ui.column().classes("w-[30rem] shrink-0 sticky top-20 self-start"):
            async def _chat_send(text: str) -> str:
                bundle_id = str(bundle_select.value or "").strip()
                if not bundle_id:
                    return "Select a plan set first."
                selected_eval = bundle_cache.get(bundle_id, {})
                selected_manifest = selected_eval.get("manifest", {}) if isinstance(selected_eval, dict) else {}
                selected_manifest_status = str(selected_manifest.get("status", "")).strip().lower()
                if selected_manifest_status == "verified":
                    return "Selected plan set is VERIFIED. Click 'Unverify Plan Set' before editing."
                try:
                    result = await asyncio.to_thread(
                        bridge.apply_bundle_chat_edit,
                        bundle_id,
                        text,
                    )
                    _refresh_controls(select_bundle_id=bundle_id)
                    return (
                        "Plan updated. "
                        f"status={result.get('status')} "
                        f"validation_ok={result.get('validation_ok')} "
                        f"witnesses={result.get('witness_count')} "
                        f"auto-replans={result.get('auto_replans_used', 0)}/"
                        f"{result.get('auto_replan_max_attempts', 3)} "
                        f"stop_reason={result.get('stop_reason', '')}"
                    )
                except Exception as exc:
                    return f"Plan update failed: {exc}"

            render_chat(
                bridge,
                agent_jid="product",
                title="Product Agent Chat",
                on_send=_chat_send,
            )

        def _norm(path_value: str) -> str:
            raw = str(path_value or "").strip()
            if not raw:
                return ""
            return str(Path(raw).resolve())

        def _scope_values() -> tuple[str, str]:
            internal = str(bridge.execution_mode or "simulation").strip().lower() or "simulation"
            if internal not in {"dry_run", "simulation", "physical"}:
                internal = "simulation"
            env = str(bridge.robot_env or "").strip().lower()
            if env not in {"gazebo", "real"}:
                env = "real" if internal == "physical" else "gazebo"
            return internal, env

        def _selected_files() -> tuple[str, str]:
            req = str(requirement_select.value or "").strip()
            safe = str(safety_select.value or "").strip()
            return req, safe

        def _feedback_scope_key() -> str:
            req, safe = _selected_files()
            mode, env = _scope_values()
            return "|".join((_norm(req), _norm(safe), mode, env))

        def _sync_refinement_feedback_scope() -> None:
            new_key = _feedback_scope_key()
            old_key = str(active_feedback_scope["key"] or "").strip()
            if old_key and old_key != new_key:
                refinement_feedback_drafts[old_key] = str(refinement_feedback.value or "")
            if old_key != new_key:
                refinement_feedback.value = refinement_feedback_drafts.get(new_key, "")
                active_feedback_scope["key"] = new_key

        def _bundle_scope_matches_selection(manifest: dict, req_file: str, safety_file: str) -> bool:
            return (
                _norm(manifest.get("product_spec_file", "")) == _norm(req_file)
                and _norm(manifest.get("safety_file", "")) == _norm(safety_file)
            )

        def _format_reasons(reasons: list[object]) -> str:
            if not reasons:
                return ""
            return ", ".join(str(r).replace("bundle", "plan_set") for r in reasons)

        def _delete_policy_reason_text(policy: dict[str, object]) -> str:
            missing = policy.get("missing_links", [])
            if isinstance(missing, list) and missing:
                labels = {
                    "manifest_missing": "manifest missing",
                    "product_requirement_file_missing": "product requirement file missing",
                    "safety_requirement_file_missing": "safety requirement file missing",
                }
                return ", ".join(labels.get(str(item), str(item)) for item in missing)
            return str(policy.get("reason", "") or "")

        def _allow_unapproved_safety() -> bool:
            return str(safety_mode_select.value or "").strip() == SAFETY_MODE_ALL

        def _set_generation_banner(kind: str | None, message: str = "") -> None:
            style_map = {
                "warning": ("warning", "text-amber-700 bg-amber-50"),
                "error": ("error", "text-red-700 bg-red-50"),
            }
            if not kind or not message:
                generation_banner.style("display:none;")
                generation_banner_label.text = ""
                return
            icon_name, color_classes = style_map.get(kind, style_map["error"])
            generation_banner_icon.name = icon_name
            generation_banner.classes(remove="text-amber-700 bg-amber-50 text-red-700 bg-red-50")
            generation_banner.classes(add=color_classes)
            generation_banner_label.text = message
            generation_banner.style("display:flex;")

        def _set_bundle_generation_busy(is_busy: bool) -> None:
            bundle_generation_state["busy"] = bool(is_busy)
            generate_loading_row.style("display:flex;" if is_busy else "display:none;")
            auto_replan_input.set_enabled(not is_busy)
            if is_busy:
                generate_btn.set_enabled(False)
                regenerate_btn.set_enabled(False)
                verify_btn.set_enabled(False)
                unverify_btn.set_enabled(False)
                delete_btn.set_enabled(False)

        def _auto_replan_limits(manifest: dict | None) -> tuple[int, int]:
            data = manifest if isinstance(manifest, dict) else {}
            replan_policy = data.get("replan_policy", {})
            validation = data.get("validation_summary", {})
            max_attempts = 3
            if isinstance(replan_policy, dict):
                try:
                    max_attempts = int(replan_policy.get("auto_replan_max_attempts", 3) or 0)
                except Exception:
                    max_attempts = 3
            auto_replans_used = 0
            if isinstance(validation, dict):
                try:
                    auto_replans_used = int(validation.get("auto_replans_used", 0) or 0)
                except Exception:
                    auto_replans_used = 0
            return max(0, min(auto_replans_used, 10)), max(0, min(max_attempts, 10))

        def _refresh_requirement_select_options() -> None:
            files = bridge.list_product_requirement_files()
            options = {f: Path(f).name for f in files}
            current = str(requirement_select.value or "").strip()
            requirement_select.options = options
            requirement_select.update()
            if current in options:
                requirement_select.value = current
            else:
                requirement_select.value = next(iter(options.keys()), None)

        def _safety_intent_reason_text(reason: str) -> str:
            mapping = {
                "approved": "approved",
                "not_approved": "not approved yet",
                "revoked": "approval revoked",
                "content_changed_since_approval": "content changed after approval",
                "safety_file_empty": "file is empty",
                "safety_file_missing": "file missing",
            }
            return mapping.get(str(reason or "").strip(), str(reason or "unknown"))

        def _refresh_safety_select_options() -> None:
            all_files = bridge.list_safety_requirement_files()
            visible_files = (
                all_files
                if _allow_unapproved_safety()
                else bridge.list_safety_requirement_files(approved_only=True)
            )
            options = {f: Path(f).name for f in visible_files}
            current = str(safety_select.value or "").strip()
            safety_select.options = options
            safety_select.update()
            if current in options:
                safety_select.value = current
            else:
                safety_select.value = next(iter(options.keys()), None)

            selected = str(safety_select.value or "").strip()
            approved_count = len(bridge.list_safety_requirement_files(approved_only=True))
            total_count = len(all_files)
            if not selected:
                if _allow_unapproved_safety():
                    safety_intent_gate_label.text = (
                        f"Safety intent approval: {approved_count}/{total_count} approved. "
                        "Advanced mode is ON."
                    )
                    safety_intent_gate_label.classes(replace="text-xs text-amber-700")
                else:
                    safety_intent_gate_label.text = (
                        "No approved safety intent file available for planning. "
                        "Approve one in Safety page first."
                    )
                    safety_intent_gate_label.classes(replace="text-xs text-red-700")
                return

            evaluation = bridge.evaluate_safety_intent_approval(selected)
            approved = bool(evaluation.get("approved", False))
            reason_text = _safety_intent_reason_text(str(evaluation.get("reason", "")))
            if approved:
                rec = evaluation.get("record", {}) if isinstance(evaluation.get("record"), dict) else {}
                approved_at = str(rec.get("approved_at_utc", "")).strip()
                suffix = f" at {approved_at}" if approved_at else ""
                safety_intent_gate_label.text = (
                    f"Safety intent approved ({Path(selected).name}){suffix}."
                )
                safety_intent_gate_label.classes(replace="text-xs text-green-700")
                return

            if _allow_unapproved_safety():
                safety_intent_gate_label.text = (
                    f"Safety intent is {reason_text} for {Path(selected).name}. "
                    "Advanced mode allows generation."
                )
                safety_intent_gate_label.classes(replace="text-xs text-amber-700")
            else:
                safety_intent_gate_label.text = (
                    f"Safety intent is {reason_text} for {Path(selected).name}. "
                    "Approval required to generate."
                )
                safety_intent_gate_label.classes(replace="text-xs text-red-700")

        def _load_requirement_previews(req_file: str, safety_file: str) -> None:
            req_path = str(req_file or "").strip()
            safety_path = str(safety_file or "").strip()

            if req_path:
                try:
                    product_req_preview.value = Path(req_path).read_text(encoding="utf-8")
                except Exception as exc:
                    product_req_preview.value = f"[Failed to load product requirement file: {exc}]"
            else:
                product_req_preview.value = "(No product requirement file selected)"

            if safety_path:
                try:
                    safety_req_preview.value = Path(safety_path).read_text(encoding="utf-8")
                except Exception as exc:
                    safety_req_preview.value = f"[Failed to load safety requirement file: {exc}]"
            else:
                safety_req_preview.value = "(No safety requirement file selected)"

        def _render_graphs(bundle_id: str) -> None:
            if not bundle_id:
                dag_mermaid.content = "graph LR\n    empty[No plan set selected]"
                fsa_mermaid.content = "graph TB\n    empty[No plan set selected]"
                fsa_index.content = "No global FSA loaded"
                return
            try:
                artifacts = bridge.get_bundle_artifacts(bundle_id)
                nodes = artifacts.get("plan_nodes", [])
                fsa = artifacts.get("global_fsa", {})
                dag_mermaid.content = nodes_to_mermaid(nodes if isinstance(nodes, list) else [])
                fsa_mermaid.content = fsa_to_mermaid(fsa if isinstance(fsa, dict) else {})
                fsa_index.content = fsa_state_index_text(fsa if isinstance(fsa, dict) else {})
            except Exception as exc:
                dag_mermaid.content = "graph LR\n    empty[Plan-set artifact load failed]"
                fsa_mermaid.content = "graph TB\n    empty[Plan-set artifact load failed]"
                fsa_index.content = f"Plan-set artifact load failed: {exc}"

        def _render_selected_details(active_id: str, options: dict[str, str]) -> None:
            selected_id = str(bundle_select.value or "")
            selected_eval = bundle_cache.get(selected_id)
            selected_manifest = (selected_eval or {}).get("manifest", {})
            delete_policy = bridge.get_bundle_delete_policy(selected_id) if selected_id else {}
            validation = (
                selected_manifest.get("validation_summary", {})
                if isinstance(selected_manifest, dict)
                else {}
            )
            validation_ok = bool(validation.get("ok", False))
            auto_replans_used, auto_replan_max = _auto_replan_limits(
                selected_manifest if isinstance(selected_manifest, dict) else {}
            )
            stop_reason = (
                str(validation.get("stop_reason", "")).strip()
                if isinstance(validation, dict)
                else ""
            )
            retry_summary = f"auto-replans={auto_replans_used}/{auto_replan_max}"

            if selected_eval:
                status = str(selected_eval.get("status", "")).lower()
                ok = bool(selected_eval.get("ok", False))
                reasons = selected_eval.get("reasons", [])
                if status == "verified" and ok:
                    status_label.text = f"Status: VERIFIED (compatible) | {retry_summary} | id={selected_id}"
                    status_label.classes(replace="text-sm text-green-700")
                elif status == "invalid":
                    suffix = f" | stop={stop_reason}" if stop_reason else ""
                    status_label.text = f"Status: INVALID | {retry_summary}{suffix} | reasons={_format_reasons(reasons)}"
                    status_label.classes(replace="text-sm text-red-700")
                elif status == "draft":
                    suffix = f" | stop={stop_reason}" if stop_reason else ""
                    status_label.text = f"Status: DRAFT (needs user verification) | {retry_summary}{suffix} | id={selected_id}"
                    status_label.classes(replace="text-sm text-amber-700")
                else:
                    status_label.text = (
                        f"Status: {status.upper() or 'UNKNOWN'} | {retry_summary} | reasons={_format_reasons(reasons)}"
                    )
                    status_label.classes(replace="text-sm text-amber-700")

                summary = selected_eval.get("summary") or {}
                manifest = selected_eval.get("manifest") or {}
                meta.content = json.dumps(
                    {
                        "bundle_id": selected_id,
                        "active": bool(active_id and selected_id == active_id),
                        "status": selected_eval.get("status"),
                        "compatible": selected_eval.get("ok"),
                        "reasons": reasons,
                        "created_at_utc": summary.get("created_at_utc"),
                        "manifest_path": summary.get("manifest_path", ""),
                        "requirement_file": manifest.get("product_spec_file", ""),
                        "safety_file": manifest.get("safety_file", ""),
                        "parent_bundle_id": manifest.get("parent_bundle_id", ""),
                        "refinement_feedback": manifest.get("refinement_feedback", ""),
                        "source_hashes": manifest.get("source_hashes", {}),
                        "replan_policy": manifest.get("replan_policy", {}),
                        "validation_summary": manifest.get("validation_summary", {}),
                        "delete_policy": delete_policy,
                    },
                    indent=2,
                )
                _render_graphs(selected_id)
            else:
                if options:
                    status_label.text = "Select a plan set to inspect."
                else:
                    status_label.text = "No plan set generated for selected requirement/safety files."
                status_label.classes(replace="text-sm text-slate-600")
                meta.content = "{}"
                _render_graphs("")

            is_busy = (
                bridge.system_running
                or bridge._starting
                or bridge._stopping
                or bundle_generation_state["busy"]
            )
            selected_safety = str(safety_select.value or "").strip()
            safety_eval = (
                bridge.evaluate_safety_intent_approval(selected_safety)
                if selected_safety
                else {"approved": False}
            )
            safety_ready = bool(selected_safety) and (
                bool(safety_eval.get("approved", False)) or _allow_unapproved_safety()
            )
            generate_btn.set_enabled(
                not is_busy
                and bool(str(requirement_select.value or "").strip())
                and safety_ready
            )
            regenerate_btn.set_enabled(
                bool(selected_id)
                and not is_busy
                and bool(str(requirement_select.value or "").strip())
                and safety_ready
            )
            auto_replan_input.set_enabled(not is_busy)
            verify_btn.set_enabled(
                bool(selected_id)
                and not is_busy
                and str((selected_manifest or {}).get("status", "")).lower() == "draft"
                and validation_ok
            )
            unverify_btn.set_enabled(
                bool(selected_id)
                and not is_busy
                and str((selected_manifest or {}).get("status", "")).lower() == "verified"
            )
            delete_btn.set_enabled(
                bool(selected_id)
                and not is_busy
                and bool(delete_policy.get("can_delete", False))
            )

        refresh_state = {"busy": False}

        def _refresh_controls(select_bundle_id: str | None = None) -> None:
            if refresh_state["busy"]:
                return
            refresh_state["busy"] = True
            try:
                _refresh_requirement_select_options()
                _refresh_safety_select_options()
                req_file, safety_file = _selected_files()
                _sync_refinement_feedback_scope()
                mode, env = _scope_values()
                scope_label.text = f"Generation scope (read-only): execution_mode={mode}, robot_env={env}"
                requirement_path_label.text = f"Selected product requirement file: {req_file or '(none)'}"
                safety_path_label.text = f"Selected safety requirement file: {safety_file or '(none)'}"
                _load_requirement_previews(req_file, safety_file)
                bundle_cache.clear()

                active = bridge.get_active_bundle()
                active_id = str(active.get("bundle_id", "")) if active else ""
                active_label.text = f"Startup plan set: {active_id or 'none'}"

                options: dict[str, str] = {}
                for row in bridge.list_bundles():
                    bid = str(row.get("bundle_id", "")).strip()
                    if not bid:
                        continue
                    evaluation = bridge.evaluate_bundle_for_files(
                        bid,
                        req_file,
                        safety_file,
                        mode,
                        env,
                    )
                    manifest = evaluation.get("manifest") or {}
                    manifest_status = str(
                        manifest.get("status", row.get("status", ""))
                    ).strip().lower()
                    matches_scope = _bundle_scope_matches_selection(manifest, req_file, safety_file)
                    delete_policy = bridge.get_bundle_delete_policy(bid)
                    missing_links = delete_policy.get("missing_links", [])
                    has_missing_links = isinstance(missing_links, list) and bool(missing_links)

                    bundle_cache[bid] = evaluation
                    created = str(row.get("created_at_utc", ""))[:19].replace("T", " ")
                    status = str(manifest_status or evaluation.get("status", row.get("status", ""))).upper()
                    if matches_scope:
                        options[bid] = f"{created} | {status} | SELECTED SCOPE | {bid}"
                    elif has_missing_links:
                        options[bid] = f"{created} | {status} | UNLINKED | {bid}"
                    else:
                        options[bid] = f"{created} | {status} | OTHER SCOPE | {bid}"

                bundle_select.options = options
                bundle_select.update()
                if select_bundle_id and select_bundle_id in options:
                    bundle_select.value = select_bundle_id
                elif bundle_select.value not in options:
                    if active_id and active_id in options:
                        bundle_select.value = active_id
                    else:
                        bundle_select.value = next(iter(options.keys()), None)

                _render_selected_details(active_id, options)
            finally:
                refresh_state["busy"] = False

        async def _generate_bundle(use_feedback: bool = False) -> None:
            if bundle_generation_state["busy"]:
                return
            req_file, safety_file = _selected_files()
            mode, env = _scope_values()
            if not req_file or not safety_file:
                ui.notify("Select both requirement and safety files first.", type="warning")
                return
            safety_eval = bridge.evaluate_safety_intent_approval(safety_file)
            if not bool(safety_eval.get("approved", False)) and not _allow_unapproved_safety():
                reason_text = _safety_intent_reason_text(str(safety_eval.get("reason", "")))
                ui.notify(
                    f"Safety intent approval required before generation ({reason_text}).",
                    type="warning",
                )
                _refresh_controls()
                return
            feedback_text = ""
            parent_bundle_id = ""
            if use_feedback:
                feedback_text = str(refinement_feedback.value or "").strip()
                refinement_feedback_drafts[_feedback_scope_key()] = feedback_text
                if not feedback_text:
                    ui.notify("Enter refinement feedback before regenerating.", type="warning")
                    return
                parent_bundle_id = str(bundle_select.value or "").strip()
                if not parent_bundle_id:
                    ui.notify("Select a plan set first.", type="warning")
                    return
                selected_eval = bundle_cache.get(parent_bundle_id, {})
                selected_manifest = (
                    selected_eval.get("manifest", {}) if isinstance(selected_eval, dict) else {}
                )
                if not _bundle_scope_matches_selection(selected_manifest, req_file, safety_file):
                    ui.notify(
                        "Select a plan set for the same requirement and safety files before regenerating.",
                        type="warning",
                    )
                    return
            _set_bundle_generation_busy(True)
            _set_generation_banner(None)
            status_label.text = (
                "Regenerating plan set with refinement feedback..."
                if use_feedback
                else "Generating plan set..."
            )
            status_label.classes(replace="text-sm text-blue-700")
            target_bundle_id: str | None = None
            try:
                if not bool(safety_eval.get("approved", False)) and _allow_unapproved_safety():
                    ui.notify(
                        "Generating with unapproved safety intent (advanced mode).",
                        type="warning",
                    )
                ui.notify(
                    "Regenerating plan set with feedback..." if use_feedback else "Generating verified plan set...",
                    type="info",
                )
                selected_product = str(product_init_select.value or "").strip() or None
                result = await asyncio.to_thread(
                    bridge.generate_verified_bundle,
                    selected_product,
                    mode,
                    env,
                    product_requirement_file=req_file,
                    safety_requirement_file=safety_file,
                    auto_replan_max_attempts=int(auto_replan_input.value or 0),
                    refinement_feedback=feedback_text,
                    parent_bundle_id=parent_bundle_id,
                )
                summary = result.get("summary", {})
                manifest = result.get("manifest", {}) if isinstance(result.get("manifest"), dict) else {}
                validation = (
                    manifest.get("validation_summary", {})
                    if isinstance(manifest.get("validation_summary"), dict)
                    else {}
                )
                bid = str(summary.get("bundle_id", ""))
                target_bundle_id = bid
                status = str(summary.get("status", "")).upper() or "UNKNOWN"
                retries_used, retries_max = _auto_replan_limits(manifest)
                stop_reason = str(validation.get("stop_reason", "")).strip()
                if status == "INVALID":
                    _set_generation_banner(
                        "error",
                        "Plan set remains INVALID after auto-replan "
                        f"({retries_used}/{retries_max}, stop={stop_reason or 'max_attempts_reached'}).",
                    )
                    ui.notify(
                        f"Plan set {'regenerated' if use_feedback else 'generated'}: {bid} ({status})",
                        type="warning",
                    )
                else:
                    _set_generation_banner(None)
                    ui.notify(
                        f"Plan set {'regenerated' if use_feedback else 'generated'}: "
                        f"{bid} ({status}) auto-replans={retries_used}/{retries_max}",
                        type="positive",
                    )
            except Exception as exc:
                _set_generation_banner("error", f"Plan-set generation failed: {exc}")
                ui.notify(f"Plan-set generation failed: {exc}", type="negative")
            finally:
                _set_bundle_generation_busy(False)
                _refresh_controls(select_bundle_id=target_bundle_id)

        async def _verify_bundle() -> None:
            bundle_id = str(bundle_select.value or "").strip()
            if not bundle_id:
                ui.notify("Select a plan set first.", type="warning")
                return
            try:
                await asyncio.to_thread(bridge.verify_bundle, bundle_id)
                ui.notify(f"Plan set verified: {bundle_id}", type="positive")
            except Exception as exc:
                ui.notify(f"Verification failed: {exc}", type="negative")
            finally:
                _refresh_controls(select_bundle_id=bundle_id)

        async def _unverify_bundle() -> None:
            bundle_id = str(bundle_select.value or "").strip()
            if not bundle_id:
                ui.notify("Select a plan set first.", type="warning")
                return
            try:
                await asyncio.to_thread(bridge.unverify_bundle, bundle_id)
                ui.notify(f"Plan set moved to draft: {bundle_id}", type="positive")
            except Exception as exc:
                ui.notify(f"Failed to unverify plan set: {exc}", type="negative")
            finally:
                _refresh_controls(select_bundle_id=bundle_id)

        async def _delete_bundle() -> None:
            bundle_id = str(bundle_select.value or "").strip()
            if not bundle_id:
                ui.notify("Select a plan set first.", type="warning")
                return
            delete_policy = bridge.get_bundle_delete_policy(bundle_id)
            if not bool(delete_policy.get("can_delete", False)):
                ui.notify("Verified plan set cannot be deleted. Unverify it first.", type="warning")
                return
            if delete_policy.get("missing_links"):
                ui.notify(
                    f"Deleting unlinked plan set: {_delete_policy_reason_text(delete_policy)}",
                    type="warning",
                )
            try:
                await asyncio.to_thread(bridge.delete_bundle, bundle_id)
                ui.notify(f"Plan set deleted: {bundle_id}", type="positive")
            except Exception as exc:
                ui.notify(f"Failed to delete plan set: {exc}", type="negative")
            finally:
                _refresh_controls()

        def _on_selected(_=None) -> None:
            _set_generation_banner(None)
            active = bridge.get_active_bundle()
            active_id = str(active.get("bundle_id", "")) if active else ""
            options = bundle_select.options if isinstance(bundle_select.options, dict) else {}
            _render_selected_details(active_id, options)

        async def _generate_initial_bundle() -> None:
            await _generate_bundle(False)

        async def _regenerate_bundle_with_feedback() -> None:
            await _generate_bundle(True)

        generate_btn.on_click(_generate_initial_bundle)
        regenerate_btn.on_click(_regenerate_bundle_with_feedback)
        verify_btn.on_click(_verify_bundle)
        unverify_btn.on_click(_unverify_bundle)
        delete_btn.on_click(_delete_bundle)
        product_init_select.on_value_change(lambda _: _refresh_controls())
        requirement_select.on_value_change(lambda _: _refresh_controls())
        safety_select.on_value_change(lambda _: _refresh_controls())
        safety_mode_select.on_value_change(lambda _: _refresh_controls())
        bundle_select.on_value_change(_on_selected)

        _refresh_controls()
