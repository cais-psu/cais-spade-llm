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
                    "Unverify = move a verified plan set back to draft before editing."
                ).classes("text-xs text-slate-500")

                with ui.row().classes("gap-2 mt-3 flex-wrap"):
                    generate_btn = ui.button("Generate Plan Set", icon="bolt").props("color=primary")
                    verify_btn = ui.button("User Verify Plan", icon="verified").props("flat color=green")
                    unverify_btn = ui.button("Unverify Plan Set", icon="remove_done").props("flat color=orange")
                    delete_btn = ui.button("Delete Plan Set", icon="delete").props("flat color=red")
                with ui.row().classes("items-center gap-2 mt-1 text-primary"):
                    generate_loading_row = ui.row().classes("items-center gap-2")
                    with generate_loading_row:
                        ui.spinner(size="sm")
                        ui.label("Generating plan set...")
                    generate_loading_row.style("display:none;")

                bundle_cache: dict[str, dict] = {}
                bundle_generation_state = {"busy": False}

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
                        f"witnesses={result.get('witness_count')}"
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

        def _format_reasons(reasons: list[object]) -> str:
            if not reasons:
                return ""
            return ", ".join(str(r).replace("bundle", "plan_set") for r in reasons)

        def _allow_unapproved_safety() -> bool:
            return str(safety_mode_select.value or "").strip() == SAFETY_MODE_ALL

        def _set_bundle_generation_busy(is_busy: bool) -> None:
            bundle_generation_state["busy"] = bool(is_busy)
            generate_loading_row.style("display:flex;" if is_busy else "display:none;")
            if is_busy:
                generate_btn.set_enabled(False)
                verify_btn.set_enabled(False)
                unverify_btn.set_enabled(False)
                delete_btn.set_enabled(False)

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
            validation = (
                selected_manifest.get("validation_summary", {})
                if isinstance(selected_manifest, dict)
                else {}
            )
            validation_ok = bool(validation.get("ok", False))

            if selected_eval:
                status = str(selected_eval.get("status", "")).lower()
                ok = bool(selected_eval.get("ok", False))
                reasons = selected_eval.get("reasons", [])
                if status == "verified" and ok:
                    status_label.text = f"Status: VERIFIED (compatible) | id={selected_id}"
                    status_label.classes(replace="text-sm text-green-700")
                elif status == "invalid":
                    status_label.text = f"Status: INVALID | reasons={_format_reasons(reasons)}"
                    status_label.classes(replace="text-sm text-red-700")
                elif status == "draft":
                    status_label.text = f"Status: DRAFT (needs user verification) | id={selected_id}"
                    status_label.classes(replace="text-sm text-amber-700")
                else:
                    status_label.text = (
                        f"Status: {status.upper() or 'UNKNOWN'} | reasons={_format_reasons(reasons)}"
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
                        "source_hashes": manifest.get("source_hashes", {}),
                        "validation_summary": manifest.get("validation_summary", {}),
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
                and str((selected_manifest or {}).get("status", "")).lower() != "verified"
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
                    matches_scope = (
                        _norm(manifest.get("product_spec_file", "")) == _norm(req_file)
                        and _norm(manifest.get("safety_file", "")) == _norm(safety_file)
                    )
                    if not matches_scope and manifest_status != "verified":
                        continue

                    bundle_cache[bid] = evaluation
                    created = str(row.get("created_at_utc", ""))[:19].replace("T", " ")
                    status = str(manifest_status or evaluation.get("status", row.get("status", ""))).upper()
                    if matches_scope:
                        options[bid] = f"{created} | {status} | {bid}"
                    else:
                        options[bid] = f"{created} | VERIFIED (other req/safety) | {bid}"

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

        async def _generate_bundle() -> None:
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
            _set_bundle_generation_busy(True)
            status_label.text = "Generating plan set..."
            status_label.classes(replace="text-sm text-blue-700")
            target_bundle_id: str | None = None
            try:
                if not bool(safety_eval.get("approved", False)) and _allow_unapproved_safety():
                    ui.notify(
                        "Generating with unapproved safety intent (advanced mode).",
                        type="warning",
                    )
                ui.notify("Generating verified plan set...", type="info")
                result = await asyncio.to_thread(
                    bridge.generate_verified_bundle,
                    None,
                    mode,
                    env,
                    product_requirement_file=req_file,
                    safety_requirement_file=safety_file,
                )
                summary = result.get("summary", {})
                bid = str(summary.get("bundle_id", ""))
                target_bundle_id = bid
                status = str(summary.get("status", "")).upper() or "UNKNOWN"
                ui.notify(f"Plan set generated: {bid} ({status})", type="positive")
            except Exception as exc:
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
            selected_eval = bundle_cache.get(bundle_id, {})
            selected_manifest = selected_eval.get("manifest", {}) if isinstance(selected_eval, dict) else {}
            selected_status = str((selected_manifest or {}).get("status", "")).strip().lower()
            if selected_status == "verified":
                ui.notify("Verified plan set cannot be deleted. Unverify it first.", type="warning")
                return
            try:
                await asyncio.to_thread(bridge.delete_bundle, bundle_id)
                ui.notify(f"Plan set deleted: {bundle_id}", type="positive")
            except Exception as exc:
                ui.notify(f"Failed to delete plan set: {exc}", type="negative")
            finally:
                _refresh_controls()

        def _on_selected(_=None) -> None:
            active = bridge.get_active_bundle()
            active_id = str(active.get("bundle_id", "")) if active else ""
            options = bundle_select.options if isinstance(bundle_select.options, dict) else {}
            _render_selected_details(active_id, options)

        generate_btn.on_click(_generate_bundle)
        verify_btn.on_click(_verify_bundle)
        unverify_btn.on_click(_unverify_bundle)
        delete_btn.on_click(_delete_bundle)
        requirement_select.on_value_change(lambda _: _refresh_controls())
        safety_select.on_value_change(lambda _: _refresh_controls())
        safety_mode_select.on_value_change(lambda _: _refresh_controls())
        bundle_select.on_value_change(_on_selected)

        _refresh_controls()
