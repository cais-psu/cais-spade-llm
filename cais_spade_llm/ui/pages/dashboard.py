"""Dashboard page: system control panel, agent overview, plan DAG, and execution timeline."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nicegui import context, ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.agent_chat import render_chat
from cais_spade_llm.ui.components.dag_graph import nodes_to_mermaid
from cais_spade_llm.ui.components.robot_status_card import render_robot_status_card

# Execution mode labels → internal values.
_MODE_MAP = {
    "Dry Run": "dry_run",
    "Simulation": "simulation",
    "Physical": "physical",
}
_MODE_LABELS = list(_MODE_MAP.keys())
_BRIDGE_MODE_OPTIONS = {
    "auto": "Auto",
    "manual": "Manual",
    "pre_ran": "Pre-ran",
}
_BRIDGE_VALIDATION_POLICY_OPTIONS = {
    "validated": "Recovery Safety Check",
    "no_validation": "No Recovery Safety Check",
}
logger = logging.getLogger(__name__)


def render(bridge: SystemBridge) -> None:
    ui.add_head_html(
        """
        <style>
        .dashboard-select-popup {
            z-index: 10000 !important;
            pointer-events: auto !important;
        }
        </style>
        """
    )
    client = context.client
    timers: list[Any] = []
    select_popup_open = {"value": False}
    select_pause_until = {"value": 0.0}

    def _notify(message: Any, **kwargs: Any) -> None:
        if getattr(client, "_deleted", False):
            return
        client.safe_invoke(lambda: ui.notify(message, **kwargs))

    def _managed_timer(interval: float, callback, **kwargs):
        def _safe_callback():
            if getattr(client, "_deleted", False):
                return None
            if select_popup_open["value"] or time.monotonic() < select_pause_until["value"]:
                return None
            try:
                return callback()
            except Exception as exc:
                logger.exception("Dashboard callback failed: %s", exc)
                if hasattr(bridge, "_diag_emit"):
                    bridge._diag_emit(f"dashboard callback exception: {exc}")
                return None

        timer = ui.timer(interval, _safe_callback, **kwargs)
        timers.append(timer)
        return timer

    def _pause_select_refresh(seconds: float = 8.0) -> None:
        select_pause_until["value"] = max(
            select_pause_until["value"],
            time.monotonic() + seconds,
        )

    def _track_select_popup(element: Any) -> None:
        def _pause(e: Any) -> None:
            _pause_select_refresh()

        def _show(e: Any) -> None:
            select_popup_open["value"] = True
            _pause_select_refresh()

        def _hide(e: Any) -> None:
            select_popup_open["value"] = False

        element.on("click", _pause)
        element.on("focus", _pause)
        element.on("popup-show", _show)
        element.on("popup-hide", _hide)

    class _NativeSelect:
        def __init__(
            self, options: dict[str, str], value: str | None, label: str, width_class: str
        ) -> None:
            self.options = dict(options)
            self.value = value
            self._handlers: list[Callable[[Any], Any]] = []
            with ui.column().classes(f"{width_class} gap-1"):
                ui.label(label).classes("text-xs text-slate-600")
                self.element = (
                    ui.element("select")
                    .classes("w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm")
                    .style("min-height: 40px;")
                )
            self.element.on(
                "change",
                self._handle_change,
                js_handler="e => emit(e.target.value)",
            )
            self.update()

        def _handle_change(self, e: Any) -> None:
            raw = getattr(e, "args", "")
            if isinstance(raw, list):
                value = str(raw[0] if raw else "")
            else:
                value = str(raw or "")
            self.value = value
            event = SimpleNamespace(value=value)
            for handler in list(self._handlers):
                handler(event)

        def on_value_change(self, handler: Callable[[Any], Any]) -> None:
            self._handlers.append(handler)

        def update(self) -> None:
            self.element.clear()
            with self.element:
                for key, text in self.options.items():
                    option = ui.element("option")
                    option._props["value"] = key
                    if key == self.value:
                        option._props["selected"] = True
                    option._text = text
                    option.update()
            self.element.update()

    def _cancel_timers(*_) -> None:
        for timer in timers:
            try:
                timer.cancel(with_current_invocation=True)
            except TypeError:
                timer.cancel()
            except Exception:
                pass
        timers.clear()

    client.on_delete(_cancel_timers)

    ui.label("Dashboard").classes("text-2xl font-bold px-6 pt-6")
    refresh_dag_now = lambda: None

    with ui.row().classes("w-full px-6 gap-6 items-start no-wrap"):
        # ── Left column: existing dashboard content ───────────
        with ui.column().classes("flex-1 gap-6 min-w-0"):
            # ── System Control Card ──────────────────────────────────────
            with ui.card().classes("w-full").style("overflow: visible;"):
                ui.label("System Control").classes("text-lg font-semibold mb-2")

                with ui.row().classes("items-end gap-6 flex-wrap"):
                    startup_source_select = ui.radio(
                        ["Product Order Runtime"],
                        value="Product Order Runtime",
                    ).props("inline")

                product_init_files = bridge.list_product_files()
                product_init_select = (
                    ui.select(
                        {f: Path(f).stem for f in product_init_files},
                        value=product_init_files[0] if product_init_files else None,
                        label="Product",
                    )
                    .props("popup-content-class=dashboard-select-popup")
                    .classes("w-64 mt-2")
                )
                _track_select_popup(product_init_select)

                def _manifest_product_order_file(product_init_file: str | None) -> str:
                    product_file = str(product_init_file or "").strip()
                    if not product_file:
                        return ""
                    try:
                        ctx = bridge._resolve_product_context(product_file, include_hashes=False)
                    except Exception:
                        logger.debug(
                            "Failed to resolve product order file for product init %s",
                            product_file,
                            exc_info=True,
                        )
                        return ""
                    return str(ctx.get("product_order_file") or "").strip()

                product_order_files = bridge.list_product_order_files()
                safety_files = bridge.list_safety_requirement_files()
                product_order_options = {f: Path(f).name for f in product_order_files}
                default_product_order_file = _manifest_product_order_file(
                    str(product_init_select.value or "")
                )
                if default_product_order_file not in product_order_options:
                    default_product_order_file = (
                        product_order_files[0] if product_order_files else None
                    )
                safety_options = {"__NONE__": "None"}
                for f in safety_files:
                    safety_options[f] = Path(f).name
                default_safety_file = next(
                    (f for f in safety_files if Path(f).name == "safety_none.txt"),
                    safety_files[0] if safety_files else "__NONE__",
                )

                with ui.row().classes("items-end gap-4 flex-wrap mt-2"):
                    order_select = _NativeSelect(
                        product_order_options,
                        value=default_product_order_file,
                        label="Product Order JSON",
                        width_class="w-80",
                    )

                    safety_select = _NativeSelect(
                        safety_options,
                        value=default_safety_file,
                        label="Safety Requirement (.txt)",
                        width_class="w-80",
                    )

                source_hint_label = ui.label("").classes("text-xs text-slate-600")
                with ui.row().classes("items-end gap-4 flex-wrap mt-2"):
                    mode_select = _NativeSelect(
                        {label: label for label in _MODE_LABELS},
                        value="Simulation",
                        label="Mode",
                        width_class="w-48",
                    )
                    fast_forward_switch = ui.switch(
                        "Fast Forward Simulation",
                        value=bool(getattr(bridge, "fast_forward_simulation_enabled", False)),
                    ).classes("mb-1")
                runtime_bridge_settings = bridge.get_runtime_bridge_settings()
                bridge_archive_options: dict[str, str] = {}
                bridge_archive_entries: dict[str, dict[str, Any]] = {}
                bridge_control_refreshing = {"value": False}
                selection_refreshing = {"value": False}
                with ui.row().classes("items-end gap-4 flex-wrap mt-2"):
                    bridge_mode_select = (
                        ui.select(
                            _BRIDGE_MODE_OPTIONS,
                            value=str(runtime_bridge_settings.get("mode") or "pre_ran").strip()
                            or "pre_ran",
                            label="Bridge Handoff Mode",
                        )
                        .props("popup-content-class=dashboard-select-popup")
                        .classes("w-48")
                    )
                    _track_select_popup(bridge_mode_select)
                    bridge_validation_select = (
                        ui.select(
                            _BRIDGE_VALIDATION_POLICY_OPTIONS,
                            value=(
                                str(
                                    runtime_bridge_settings.get("validation_policy") or "validated"
                                ).strip()
                                or "validated"
                            ),
                            label="Recovery Safety Mode",
                        )
                        .props("popup-content-class=dashboard-select-popup")
                        .classes("w-52")
                    )
                    _track_select_popup(bridge_validation_select)
                    bridge_archive_select = (
                        ui.select(
                            {},
                            label="Archived Bridge Run",
                        )
                        .props("popup-content-class=dashboard-select-popup")
                        .classes("w-[34rem]")
                    )
                    _track_select_popup(bridge_archive_select)
                bridge_mode_status = ui.label("").classes("text-xs text-slate-600")
                bridge_fixture_status = ui.label("").classes("text-xs text-amber-700")
                bridge_archive_hint = ui.label("").classes("text-xs text-slate-600")
                bridge.execution_mode = _MODE_MAP.get(mode_select.value, "simulation")
                bridge.robot_env = (
                    "gazebo" if bridge.execution_mode in ("dry_run", "simulation") else "real"
                )

                # Prerequisite banner.
                prereq_banner = ui.column().classes("w-full mt-3")
                bundle_gate_banner = ui.column().classes("w-full mt-2")

                with ui.column().classes("w-full gap-2 mt-4"):
                    banner_hide_task: asyncio.Task | None = None
                    start_click_state = {"locked": False}
                    gazebo_launch_state = {"busy": False}
                    start_task: asyncio.Task | None = None

                    def _selected_files() -> tuple[str, str]:
                        order_file = str(order_select.value or "").strip()
                        safe_file = str(safety_select.value or "").strip()
                        return order_file, safe_file

                    def _refresh_product_order_options() -> None:
                        selection_refreshing["value"] = True
                        try:
                            options = {
                                f: Path(f).name
                                for f in bridge.list_product_order_files(
                                    str(product_init_select.value or "")
                                )
                            }
                            current = str(order_select.value or "").strip()
                            manifest_order_file = _manifest_product_order_file(
                                str(product_init_select.value or "")
                            )
                            order_select.options = options
                            if current in options:
                                order_select.value = current
                            elif manifest_order_file in options:
                                order_select.value = manifest_order_file
                            else:
                                order_select.value = next(iter(options.keys()), None)
                            order_select.update()
                            source_hint_label.text = "Startup source: selected product-order JSON and verified safety file."
                        finally:
                            selection_refreshing["value"] = False

                    def _refresh_runtime_bridge_controls() -> None:
                        bridge_control_refreshing["value"] = True
                        try:
                            settings = bridge.get_runtime_bridge_settings()
                            selected_mode = (
                                str(settings.get("mode") or "pre_ran").strip() or "pre_ran"
                            )
                            validation_policy = (
                                str(settings.get("validation_policy") or "validated").strip()
                                or "validated"
                            )
                            selected_path = str(settings.get("selected_archive_path") or "").strip()
                            if str(bridge_mode_select.value or "") != selected_mode:
                                bridge_mode_select.value = selected_mode
                            if str(bridge_validation_select.value or "") != validation_policy:
                                bridge_validation_select.value = validation_policy

                            bridge_archive_options.clear()
                            bridge_archive_entries.clear()
                            for entry in bridge.list_runtime_bridge_archives():
                                if not isinstance(entry, dict):
                                    continue
                                path = str(entry.get("path") or "").strip()
                                label = str(entry.get("label") or path).strip()
                                if not path:
                                    continue
                                bridge_archive_entries[path] = dict(entry)
                                bridge_archive_options[path] = label

                            bridge_archive_select.options = dict(bridge_archive_options)
                            bridge_archive_select.update()
                            resolved_value = (
                                selected_path if selected_path in bridge_archive_options else None
                            )
                            current_value = str(bridge_archive_select.value or "").strip()
                            if current_value != str(resolved_value or ""):
                                bridge_archive_select.value = resolved_value
                            bridge_validation_select.set_enabled(True)
                            bridge_validation_select.style("display:block;")
                            bridge_archive_select.set_enabled(selected_mode == "pre_ran")
                            bridge_mode_status.text = (
                                f"Runtime bridge mode: {_BRIDGE_MODE_OPTIONS.get(selected_mode, selected_mode)}"
                                + f" ({_BRIDGE_VALIDATION_POLICY_OPTIONS.get(validation_policy, validation_policy.replace('_', ' '))})."
                            )
                            if selected_mode == "pre_ran" and selected_path:
                                bridge_fixture_status.text = (
                                    f"Selected archived bridge path: {selected_path}"
                                )
                            else:
                                bridge_fixture_status.text = "No archived bridge run selected."

                            if selected_mode == "pre_ran":
                                if resolved_value:
                                    selected_entry = (
                                        bridge_archive_entries.get(resolved_value) or {}
                                    )
                                    relative_path = str(
                                        selected_entry.get("relative_path") or ""
                                    ).strip()
                                    label = str(
                                        selected_entry.get("label") or resolved_value
                                    ).strip()
                                    extra = f" ({relative_path})" if relative_path else ""
                                    bridge_archive_hint.text = (
                                        f"Selected archived bridge run: {label}{extra}. "
                                        + (
                                            "It will auto-load, run the ordinary CCA runtime plan validation on the merged nominal + recovery plan before execution, and keep Recovery Safety Check enabled for recovery_safety runtime enforcement."
                                            if validation_policy == "validated"
                                            else "It will auto-load, run the ordinary CCA runtime plan validation on the merged nominal + recovery plan before execution, and keep No Recovery Safety Check for recovery_safety runtime enforcement."
                                        )
                                    )
                                elif bridge_archive_options:
                                    bridge_archive_hint.text = (
                                        f"{len(bridge_archive_options)} archived primitive-program bridge outputs are available. "
                                        + (
                                            "Choose one to enable archived auto-start with Recovery Safety Check enabled for recovery_safety runtime enforcement."
                                            if validation_policy == "validated"
                                            else "Choose one to enable archived auto-start with No Recovery Safety Check for recovery_safety runtime enforcement."
                                        )
                                    )
                                else:
                                    bridge_archive_hint.text = "No archived primitive-program bridge outputs were found in llm_bridge/runtime_data."
                            else:
                                bridge_archive_hint.text = "Recovery Safety Mode controls whether recovery_safety runtime enforcement runs for tasks carrying recovery_safety_scope_id after the approved bridge passes ordinary CCA runtime plan validation. Archived Bridge Run is enabled only in Pre-ran mode."
                        finally:
                            bridge_control_refreshing["value"] = False

                    def _handle_bridge_mode_change(e: Any) -> None:
                        if bridge_control_refreshing["value"]:
                            return
                        bridge.set_runtime_bridge_mode(str(e.value or "pre_ran"))
                        _refresh_runtime_bridge_controls()
                        _refresh_runtime_recovery_panel()

                    def _handle_bridge_validation_policy_change(e: Any) -> None:
                        if bridge_control_refreshing["value"]:
                            return
                        bridge.set_runtime_bridge_validation_policy(str(e.value or "validated"))
                        _refresh_runtime_bridge_controls()
                        _refresh_runtime_recovery_panel()

                    def _handle_bridge_archive_change(e: Any) -> None:
                        if bridge_control_refreshing["value"]:
                            return
                        selected_path = str(e.value or "").strip()
                        selected_entry = bridge_archive_entries.get(selected_path) or {}
                        bridge.set_runtime_bridge_archive_selection(
                            selected_path,
                            str(selected_entry.get("label") or "").strip(),
                        )
                        _refresh_runtime_bridge_controls()
                        _refresh_runtime_recovery_panel()

                    def _bundle_gate(*, strict: bool) -> tuple[bool, str]:
                        order_file, safe_file = _selected_files()
                        if not order_file:
                            return False, "Select a product order JSON file."
                        if not safe_file or safe_file.upper() == "__NONE__":
                            return True, ""
                        if Path(safe_file).name == "safety_none.txt":
                            return True, ""
                        safety_eval = bridge.evaluate_safety_intent_approval(safe_file)
                        if not bool(safety_eval.get("approved", False)):
                            reason = str(
                                safety_eval.get("reason", "not_approved") or "not_approved"
                            )
                            return False, f"Selected safety file is not verified: {reason}."
                        return True, ""

                    def _set_action_banner(
                        kind: str, message: str, *, auto_hide_s: float | None = None
                    ) -> None:
                        nonlocal banner_hide_task
                        style_map = {
                            "info": ("info", "text-blue-700 bg-blue-50"),
                            "success": ("check_circle", "text-green-700 bg-green-50"),
                            "warning": ("warning", "text-amber-700 bg-amber-50"),
                            "error": ("error", "text-red-700 bg-red-50"),
                        }
                        icon_name, color_classes = style_map.get(kind, style_map["info"])

                        action_banner_icon.name = icon_name
                        action_banner.classes(
                            remove="text-blue-700 bg-blue-50 text-green-700 bg-green-50 text-amber-700 bg-amber-50 text-red-700 bg-red-50"
                        )
                        action_banner.classes(add=color_classes)
                        action_banner_label.text = message
                        action_banner.style("display:flex;")

                        if banner_hide_task and not banner_hide_task.done():
                            banner_hide_task.cancel()
                            banner_hide_task = None

                        if auto_hide_s is not None and auto_hide_s > 0:

                            async def _hide_later():
                                try:
                                    await asyncio.sleep(auto_hide_s)
                                except asyncio.CancelledError:
                                    return
                                action_banner.style("display:none;")

                            banner_hide_task = asyncio.create_task(_hide_later())

                    async def _start():
                        nonlocal start_task
                        if hasattr(bridge, "_diag_emit"):
                            bridge._diag_emit(
                                f"dashboard start click running={bridge.system_running} "
                                f"starting={bridge._starting} locked={start_click_state['locked']}"
                            )
                        if bridge.system_running:
                            _set_action_banner(
                                "warning", "System is already running.", auto_hide_s=4.0
                            )
                            return
                        if start_click_state["locked"]:
                            _set_action_banner("warning", "Start already in progress...")
                            return
                        if bridge._starting:
                            _set_action_banner("warning", "Start already in progress...")
                            return
                        internal = _MODE_MAP[mode_select.value]
                        bridge.execution_mode = internal
                        bridge.robot_env = (
                            "gazebo" if internal in ("dry_run", "simulation") else "real"
                        )
                        bridge.fast_forward_simulation_enabled = (
                            bool(fast_forward_switch.value) and internal == "simulation"
                        )

                        try:
                            bundle_ok, bundle_msg = _bundle_gate(strict=True)
                        except Exception as exc:
                            _set_action_banner(
                                "error",
                                f"Startup source check failed: {exc}",
                                auto_hide_s=8.0,
                            )
                            return
                        if not bundle_ok:
                            _set_action_banner("warning", bundle_msg, auto_hide_s=8.0)
                            return

                        try:
                            order_file, safe_file = _selected_files()
                            if not order_file:
                                _set_action_banner(
                                    "warning", "Select a product order JSON file.", auto_hide_s=6.0
                                )
                                return
                            selected_product = str(product_init_select.value or "").strip()
                            if selected_product:
                                bridge.selected_product = selected_product
                            else:
                                try:
                                    bridge.selected_product = (
                                        bridge.resolve_product_init_for_product_order(order_file)
                                    )
                                except Exception as exc:
                                    _set_action_banner(
                                        "error",
                                        f"Invalid product order selection: {exc}",
                                        auto_hide_s=8.0,
                                    )
                                    return

                            bridge.selected_product_order_file = order_file
                            bridge.selected_requirement_file = ""
                            bridge.selected_safety_file = safe_file or ""
                            bridge.set_active_bundle(None)
                        except Exception as exc:
                            _set_action_banner(
                                "error",
                                f"Failed to configure startup source: {exc}",
                                auto_hide_s=8.0,
                            )
                            return

                        if not _check_prerequisites(
                            bridge,
                            internal,
                            prereq_banner,
                            launch_simulation=_launch_gazebo_dual,
                            launch_simulation_busy=gazebo_launch_state["busy"],
                        ):
                            detail = f" {bridge.last_error}" if bridge.last_error else ""
                            _set_action_banner(
                                "warning",
                                "Startup is not done yet." + detail,
                                auto_hide_s=6.0,
                            )
                            return
                        start_click_state["locked"] = True
                        _update_controls()
                        _set_action_banner("info", "Start System clicked. Starting agents...")
                        if start_task is None or start_task.done():

                            async def _run_start_in_background() -> None:
                                if hasattr(bridge, "_diag_emit"):
                                    bridge._diag_emit("dashboard start background task begin")
                                try:
                                    await bridge.start_system()
                                    if bridge.system_running:
                                        notice = bridge.consume_notice()
                                        if notice:
                                            _set_action_banner("warning", notice, auto_hide_s=10.0)
                                        else:
                                            _set_action_banner(
                                                "success",
                                                "System started successfully.",
                                                auto_hide_s=5.0,
                                            )
                                    else:
                                        reason = bridge.last_error or "unknown error"
                                        _set_action_banner(
                                            "error", f"Start failed: {reason}", auto_hide_s=8.0
                                        )
                                finally:
                                    if hasattr(bridge, "_diag_emit"):
                                        bridge._diag_emit(
                                            "dashboard start background task end "
                                            f"running={bridge.system_running} error={bridge.last_error or ''}"
                                        )
                                    if not bridge.system_running:
                                        start_click_state["locked"] = False
                                    _update_controls()

                            start_task = asyncio.create_task(_run_start_in_background())

                    async def _stop():
                        if bridge._stopping:
                            _set_action_banner("warning", "Stop already in progress...")
                            return
                        if not bridge.system_running:
                            _set_action_banner(
                                "warning", "System is already stopped.", auto_hide_s=4.0
                            )
                            start_click_state["locked"] = False
                            _update_controls()
                            return
                        _set_action_banner("info", "Stop System clicked. Stopping agents...")
                        try:
                            await bridge.stop_system()
                            if not bridge.system_running:
                                _set_action_banner("success", "System stopped.", auto_hide_s=5.0)
                                start_click_state["locked"] = False
                            else:
                                _set_action_banner(
                                    "error",
                                    "Stop failed. System is still running.",
                                    auto_hide_s=8.0,
                                )
                        finally:
                            _update_controls()

                    async def _run_order_dry_run():
                        order_file, safe_file = _selected_files()
                        if not order_file:
                            _set_action_banner(
                                "warning", "Select a product order JSON file.", auto_hide_s=6.0
                            )
                            return
                        if not safe_file or safe_file.upper() == "__NONE__":
                            _set_action_banner(
                                "warning", "Select a verified safety file.", auto_hide_s=6.0
                            )
                            return
                        dry_run_status.text = "Running product-order dry run..."
                        dry_run_status.classes(replace="text-sm text-blue-700")
                        dry_run_result_code.content = "{}"
                        try:
                            result = await asyncio.to_thread(
                                bridge.run_order_dry_run,
                                order_file,
                                safe_file,
                            )
                        except Exception as exc:
                            dry_run_status.text = f"Order dry run failed: {exc}"
                            dry_run_status.classes(replace="text-sm text-red-700")
                            return

                        summary = {
                            "artifact_path": result.get("artifact_path", ""),
                            "selected_parts": result.get("selected_parts", []),
                            "ready_operations": result.get("ready_operations", []),
                            "blocked_operations": result.get("blocked_operations", []),
                            "completed_count": len(result.get("completed_operations", []) or []),
                            "monitor_state": result.get("monitor_state", {}),
                            "ordering_constraints": result.get("ordering_constraints", []),
                            "monitor_rules": result.get("monitor_rules", []),
                        }
                        dry_run_result_code.content = json.dumps(summary, indent=2)
                        dry_run_status.text = (
                            "Order dry run completed. "
                            f"completed={summary['completed_count']} "
                            f"blocked={len(summary['blocked_operations'])}"
                        )
                        dry_run_status.classes(replace="text-sm text-green-700")

                    async def _launch_gazebo_dual():
                        if gazebo_launch_state["busy"]:
                            _set_action_banner(
                                "warning",
                                "Gazebo + MoveIt launch already in progress...",
                                auto_hide_s=4.0,
                            )
                            return

                        if any(
                            bridge.ros2_proc_status(name) == "running"
                            for name in ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")
                        ):
                            _set_action_banner(
                                "warning",
                                "Gazebo stack is already running.",
                                auto_hide_s=4.0,
                            )
                            _update_controls()
                            return

                        gazebo_launch_state["busy"] = True
                        _update_controls()
                        _set_action_banner("info", "Launching Gazebo + MoveIt...")
                        try:
                            fast_forward = (
                                bool(fast_forward_switch.value)
                                and _MODE_MAP.get(mode_select.value, "dry_run") == "simulation"
                            )
                            bridge.fast_forward_simulation_enabled = fast_forward
                            err = await asyncio.to_thread(
                                bridge.ros2_start,
                                "gazebo_dual",
                                fast_forward_simulation=fast_forward,
                            )
                            if err:
                                _set_action_banner("warning", err, auto_hide_s=8.0)
                            else:
                                _set_action_banner(
                                    "success",
                                    "Gazebo + MoveIt launched. Waiting for ROS services...",
                                    auto_hide_s=6.0,
                                )
                        finally:
                            gazebo_launch_state["busy"] = False
                            _update_controls()

                    reset_scope_options = [
                        "Reset All",
                        "Reset Plan",
                        "Reset Gazebo",
                    ]

                    async def _reset_selected():
                        if bridge._starting:
                            _set_action_banner(
                                "warning",
                                "Start is in progress. Wait before reset.",
                                auto_hide_s=4.0,
                            )
                            return
                        if bridge._stopping:
                            _set_action_banner(
                                "warning",
                                "Stop is in progress. Wait before reset.",
                                auto_hide_s=4.0,
                            )
                            return
                        selected_scope = (
                            str(reset_scope_select.value or "Reset All").strip() or "Reset All"
                        )

                        _set_action_banner("info", f"{selected_scope} clicked. Preparing reset...")
                        try:
                            # Keep agent/world state consistent: stop system first if needed.
                            if bridge.system_running:
                                _set_action_banner(
                                    "info", f"Stopping system before {selected_scope.lower()}..."
                                )
                                await bridge.stop_system()
                                if bridge.system_running:
                                    _set_action_banner(
                                        "error",
                                        "Reset aborted: failed to stop system first.",
                                        auto_hide_s=8.0,
                                    )
                                    return
                                start_click_state["locked"] = False

                            success_messages: list[str] = []
                            warning_messages: list[str] = []

                            if selected_scope in {"Reset Plan", "Reset All"}:
                                ok_plan, msg_plan = await asyncio.to_thread(
                                    bridge.reset_plan_runtime_state
                                )
                                if ok_plan:
                                    success_messages.append(msg_plan)
                                else:
                                    warning_messages.append(msg_plan)

                            if selected_scope in {"Reset Gazebo", "Reset All"}:
                                ok_gz, msg_gz = await asyncio.to_thread(
                                    bridge.ros2_reset_gazebo_environment
                                )
                                if ok_gz:
                                    success_messages.append(msg_gz)
                                else:
                                    warning_messages.append(msg_gz)

                            start_click_state["locked"] = False
                            if success_messages and warning_messages:
                                _set_action_banner(
                                    "warning",
                                    " ; ".join(success_messages + warning_messages),
                                    auto_hide_s=8.0,
                                )
                            elif warning_messages:
                                _set_action_banner(
                                    "warning", " ; ".join(warning_messages), auto_hide_s=8.0
                                )
                            else:
                                _set_action_banner(
                                    "success",
                                    " ; ".join(success_messages)
                                    or "Reset complete. Click Start System.",
                                    auto_hide_s=6.0,
                                )
                        finally:
                            _update_controls()

                    with ui.row().classes("items-end gap-4 flex-wrap"):
                        start_btn = ui.button(
                            "Start System", on_click=_start, icon="play_arrow"
                        ).props("color=green")
                        dry_run_btn = ui.button(
                            "Run Order Dry Run",
                            on_click=_run_order_dry_run,
                            icon="preview",
                        ).props("color=primary")
                        stop_btn = ui.button("Stop System", on_click=_stop, icon="stop").props(
                            "color=red"
                        )
                        reset_scope_select = (
                            ui.select(
                                {opt: opt for opt in reset_scope_options},
                                value="Reset All",
                                label="Reset Scope",
                            )
                            .props("popup-content-class=dashboard-select-popup")
                            .classes("w-44")
                        )
                        _track_select_popup(reset_scope_select)
                        reset_btn = ui.button(
                            "Reset", on_click=_reset_selected, icon="restart_alt"
                        ).props("color=blue")
                    action_banner = ui.row().classes(
                        "w-full mt-2 items-center gap-2 rounded p-3 text-sm text-blue-700 bg-blue-50"
                    )
                    action_banner.style("display:none;")
                    with action_banner:
                        action_banner_icon = ui.icon("info")
                        action_banner_label = ui.label("")
                    with ui.expansion("Order Dry Run Preview", icon="schema", value=False).classes(
                        "w-full mt-2"
                    ):
                        dry_run_status = ui.label("No dry run has been run.").classes(
                            "text-sm text-slate-600"
                        )
                        dry_run_result_code = ui.code("{}", language="json").classes("w-full")
                    plan_safety_banner = ui.row().classes(
                        "w-full mt-2 items-center gap-2 rounded p-3 text-sm text-red-700 bg-red-50"
                    )
                    plan_safety_banner.style("display:none;")
                    with plan_safety_banner:
                        plan_safety_banner_icon = ui.icon("warning")
                        plan_safety_banner_label = ui.label("")

                # Error display.
                error_label = ui.label("").classes("text-red-500 text-sm mt-2")
                hw_probe = {"busy": False}

                def _set_plan_safety_banner(alerts: list[dict]) -> None:
                    if not alerts:
                        plan_safety_banner.style("display:none;")
                        plan_safety_banner_label.text = ""
                        return
                    first = alerts[0] if isinstance(alerts[0], dict) else {}
                    product = str(first.get("product_name", "product")).strip() or "product"
                    stage = str(first.get("stage", "runtime")).strip() or "runtime"
                    message = str(first.get("message", "")).strip() or "Plan safety alert."
                    retries_used = first.get("retries_used")
                    retries_max = first.get("retries_max")
                    retry_text = ""
                    if retries_used is not None and retries_max is not None:
                        retry_text = f" auto-replans={retries_used}/{retries_max}."
                    extra = ""
                    if len(alerts) > 1:
                        extra = f" (+{len(alerts) - 1} more)"
                    plan_safety_banner_icon.name = "warning"
                    plan_safety_banner_label.text = (
                        f"{product} [{stage}] {message}{retry_text}{extra}"
                    )
                    plan_safety_banner.style("display:flex;")

                def _set_bundle_gate_banner(message: str) -> None:
                    text = str(message or "").strip()
                    signature = ("bundle_gate", text)
                    if (
                        getattr(bundle_gate_banner, "_cais_bundle_gate_signature", None)
                        == signature
                    ):
                        return
                    bundle_gate_banner._cais_bundle_gate_signature = signature
                    bundle_gate_banner.clear()
                    if not text:
                        return
                    with (
                        bundle_gate_banner,
                        ui.row().classes(
                            "items-center gap-2 text-amber-700 bg-amber-50 p-3 rounded"
                        ),
                    ):
                        ui.icon("warning").classes("text-lg")
                        ui.label(text).classes("text-sm font-semibold")

                def _update_controls():
                    try:
                        internal = _MODE_MAP.get(mode_select.value, "dry_run")
                        bridge.execution_mode = internal
                        bridge.robot_env = (
                            "gazebo" if internal in ("dry_run", "simulation") else "real"
                        )
                        bridge.fast_forward_simulation_enabled = (
                            bool(fast_forward_switch.value) and internal == "simulation"
                        )
                        fast_forward_switch.set_enabled(
                            internal == "simulation"
                            and not bridge.system_running
                            and not bridge._starting
                        )
                        if internal == "physical" and hasattr(
                            bridge, "hardware_connection_statuses"
                        ):
                            if not hw_probe["busy"]:

                                async def _probe_hw():
                                    hw_probe["busy"] = True
                                    try:
                                        await asyncio.to_thread(bridge.hardware_connection_statuses)
                                    finally:
                                        hw_probe["busy"] = False

                                asyncio.create_task(_probe_hw())

                        if bridge._starting and not bridge.system_running:
                            prereqs_met = False
                        else:
                            prereqs_met = _check_prerequisites(
                                bridge,
                                internal,
                                prereq_banner,
                                launch_simulation=_launch_gazebo_dual,
                                launch_simulation_busy=gazebo_launch_state["busy"],
                            )

                        bundle_ok, bundle_msg = _bundle_gate(strict=False)
                        if not bundle_ok:
                            if not bridge.system_running:
                                _set_bundle_gate_banner(bundle_msg)
                            else:
                                _set_bundle_gate_banner("")
                        else:
                            _set_bundle_gate_banner("")

                        can_start = (
                            prereqs_met
                            and bundle_ok
                            and not bridge.system_running
                            and not bridge._starting
                            and not start_click_state["locked"]
                        )
                        gazebo_running = any(
                            bridge.ros2_proc_status(name) == "running"
                            for name in ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")
                        )
                        reset_scope = str(reset_scope_select.value or "Reset All")
                        if reset_scope == "Reset Gazebo":
                            can_reset = gazebo_running or bridge.system_running
                        else:
                            # Plan reset can run even when Gazebo stack is not up.
                            can_reset = True
                        start_btn.set_enabled(can_start)
                        dry_run_btn.set_enabled(
                            bool(str(order_select.value or "").strip())
                            and bool(str(safety_select.value or "").strip())
                            and str(safety_select.value or "").strip().upper() != "__NONE__"
                            and not bridge.system_running
                            and not bridge._starting
                            and not bridge._stopping
                        )
                        stop_btn.set_enabled(bridge.system_running and not bridge._stopping)
                        reset_btn.set_enabled(
                            can_reset and not bridge._starting and not bridge._stopping
                        )
                        _set_plan_safety_banner(bridge.get_plan_safety_alerts())
                        error_label.text = bridge.last_error or ""
                    except Exception as exc:
                        if hasattr(bridge, "_diag_emit"):
                            bridge._diag_emit(f"dashboard update_controls exception: {exc}")
                        start_btn.set_enabled(False)
                        dry_run_btn.set_enabled(False)
                        stop_btn.set_enabled(False)
                        reset_btn.set_enabled(False)
                        _set_plan_safety_banner([])
                        error_label.text = f"Dashboard control update failed: {exc}"

                _managed_timer(1.0, _update_controls)

                def _refresh_controls_and_dag() -> None:
                    _update_controls()
                    refresh_dag_now()

                def _refresh_selection_and_controls() -> None:
                    _refresh_product_order_options()
                    _refresh_runtime_bridge_controls()
                    _refresh_controls_and_dag()

                def _handle_product_init_change(e: Any) -> None:
                    if selection_refreshing["value"]:
                        return
                    _refresh_product_order_options()
                    _refresh_controls_and_dag()

                def _handle_order_or_safety_change(e: Any) -> None:
                    if selection_refreshing["value"]:
                        return
                    _refresh_controls_and_dag()

                def _handle_mode_or_source_change(e: Any) -> None:
                    _refresh_controls_and_dag()

                def _handle_fast_forward_change(e: Any) -> None:
                    internal = _MODE_MAP.get(mode_select.value, "dry_run")
                    bridge.fast_forward_simulation_enabled = (
                        bool(getattr(e, "value", False)) and internal == "simulation"
                    )
                    _refresh_controls_and_dag()

                def _guard_select_handler(
                    name: str, handler: Callable[[Any], Any]
                ) -> Callable[[Any], Any]:
                    def _wrapped(e: Any) -> Any:
                        select_popup_open["value"] = False
                        select_pause_until["value"] = 0.0
                        try:
                            return handler(e)
                        except Exception as exc:
                            logger.exception("Dashboard select handler failed: %s", exc)
                            if hasattr(bridge, "_diag_emit"):
                                bridge._diag_emit(
                                    f"dashboard select handler {name} exception: {exc}"
                                )
                            error_label.text = f"Dashboard selection update failed: {exc}"
                            return None

                    return _wrapped

                product_init_select.on_value_change(
                    _guard_select_handler("product_init_select", _handle_product_init_change)
                )
                order_select.on_value_change(
                    _guard_select_handler("order_select", _handle_order_or_safety_change)
                )
                safety_select.on_value_change(
                    _guard_select_handler("safety_select", _handle_order_or_safety_change)
                )
                startup_source_select.on_value_change(
                    _guard_select_handler("startup_source_select", _handle_mode_or_source_change)
                )
                mode_select.on_value_change(
                    _guard_select_handler("mode_select", _handle_mode_or_source_change)
                )
                fast_forward_switch.on_value_change(_handle_fast_forward_change)
                bridge_mode_select.on_value_change(
                    _guard_select_handler("bridge_mode_select", _handle_bridge_mode_change)
                )
                bridge_validation_select.on_value_change(
                    _guard_select_handler(
                        "bridge_validation_select", _handle_bridge_validation_policy_change
                    )
                )
                bridge_archive_select.on_value_change(
                    _guard_select_handler("bridge_archive_select", _handle_bridge_archive_change)
                )
                reset_scope_select.on_value_change(
                    _guard_select_handler("reset_scope_select", _handle_mode_or_source_change)
                )
                _refresh_selection_and_controls()
                _managed_timer(5.0, _refresh_runtime_bridge_controls)

            # ── Agent Overview Grid ──────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Agent Status").classes("text-lg font-semibold mb-2")
                agent_container = ui.row().classes("gap-4 flex-wrap")

            def _refresh_agents():
                agent_container.clear()
                if not bridge.system_running:
                    with agent_container:
                        ui.label("System not running").classes("text-slate-400 italic")
                    return

                statuses = bridge.get_agent_statuses()
                with agent_container:
                    for agent in statuses:
                        with ui.card().classes("w-56"):
                            with ui.row().classes("items-center gap-2"):
                                color = "green" if agent["alive"] else "red"
                                ui.icon("circle", color=color).classes("text-xs")
                                ui.label(agent["name"]).classes("font-semibold")
                            ui.label(agent["type"]).classes("text-xs text-slate-500 uppercase")
                            ui.label(agent["jid"]).classes("text-xs text-slate-400 truncate")

            _managed_timer(2.0, _refresh_agents)

            # ── Quick Stats ──────────────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Quick Stats").classes("text-lg font-semibold mb-2")
                stats_row = ui.row().classes("gap-8")

            def _refresh_stats():
                stats_row.clear()
                with stats_row:
                    task_states = bridge.get_task_states()
                    total = len(task_states)
                    completed = (
                        sum(1 for v in task_states.values() if "completed" in v.lower())
                        if task_states
                        else 0
                    )
                    failed = (
                        sum(1 for v in task_states.values() if "failed" in v.lower())
                        if task_states
                        else 0
                    )

                    _stat_card("Tasks Completed", f"{completed}/{total}", "task_alt")
                    _stat_card("Tasks Failed", str(failed), "error_outline")

                    robot_states = bridge.get_robot_states()
                    active = sum(
                        1 for r in robot_states.values() if r.get("current_state", "idle") != "idle"
                    )
                    _stat_card("Active Robots", str(active), "precision_manufacturing")

                    safety = bridge.get_safety_state()
                    blocked = len(safety.get("blocked_tasks", {}))
                    _stat_card("Safety Blocks", str(blocked), "shield")

            _managed_timer(2.0, _refresh_stats)

            def _preview_or_runtime_nodes() -> list[dict]:
                nodes = bridge.get_plan_nodes()
                if nodes or bridge.system_running:
                    return nodes
                return []

            def _preview_or_runtime_task_states(nodes: list[dict]) -> dict[str, str]:
                fallback: dict[str, str] = {}
                for node in nodes:
                    tid = str(node.get("id") or node.get("task_id") or "").strip()
                    if not tid:
                        continue
                    fallback[tid] = str(node.get("status", "pending") or "pending")
                task_states = bridge.get_task_states()
                if task_states:
                    for tid in list(fallback):
                        if tid in task_states:
                            fallback[tid] = str(task_states.get(tid) or fallback[tid])
                return fallback

            def _current_task_dag_nodes() -> list[dict]:
                if hasattr(bridge, "get_current_task_dag_nodes"):
                    return bridge.get_current_task_dag_nodes()
                return _preview_or_runtime_nodes()

            # ── Task DAG ────────────────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Task DAG").classes("text-lg font-semibold mb-2")
                mermaid = ui.mermaid("graph TD\n    empty[No plan loaded]").classes("w-full")

                def _refresh_dag():
                    nodes = _current_task_dag_nodes()
                    if not nodes:
                        mermaid.content = "graph LR\n    empty[No current operations]"
                        mermaid.update()
                        return
                    visible_ids = {
                        str(node.get("id") or node.get("task_id") or "").strip()
                        for node in nodes
                        if str(node.get("id") or node.get("task_id") or "").strip()
                    }
                    task_states = {
                        task_id: status
                        for task_id, status in _preview_or_runtime_task_states(nodes).items()
                        if task_id in visible_ids
                    }
                    mermaid.content = nodes_to_mermaid(nodes, task_states)
                    mermaid.update()

                refresh_dag_now = _refresh_dag
                _refresh_dag()
                _managed_timer(0.5, _refresh_dag)

            # ── Live Robot Status ───────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Live Robot Status").classes("text-lg font-semibold mb-2")
                robot_status_container = ui.column().classes("w-full gap-4")

                def _refresh_robot_status():
                    robot_status_container.clear()
                    states = bridge.get_robot_states()
                    if not states:
                        with robot_status_container:
                            ui.label("No robots available — start the system first").classes(
                                "text-slate-400 italic"
                            )
                        return
                    with robot_status_container:
                        for name, state in states.items():
                            render_robot_status_card(name, state)

                _refresh_robot_status()
                _managed_timer(2.0, _refresh_robot_status)

            # ── Runtime Safety Rules ────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Runtime Safety Rules").classes("text-lg font-semibold mb-2")
                runtime_rules_source = ui.label("").classes("text-xs text-slate-500 mb-2")
                runtime_rules_table = ui.table(
                    columns=[
                        {"name": "id", "label": "Rule ID", "field": "id", "sortable": True},
                        {"name": "raw_text", "label": "Rule Text", "field": "raw_text"},
                        {
                            "name": "generated_interpretation",
                            "label": "Interpretation",
                            "field": "generated_interpretation",
                        },
                        {"name": "constraint_type", "label": "Type", "field": "constraint_type"},
                    ],
                    rows=[],
                ).classes("w-full")

                def _refresh_runtime_safety_rules():
                    selected_safety_file = str(safety_select.value or "").strip()
                    preview = {}
                    if selected_safety_file and selected_safety_file.upper() != "__NONE__":
                        try:
                            preview = bridge.get_safety_rule_preview(selected_safety_file)
                        except Exception:
                            preview = {}
                    rules = (
                        preview.get("rules", []) if isinstance(preview.get("rules"), list) else []
                    )
                    if not rules:
                        rules = bridge.get_safety_rules(None)
                    if (
                        bridge.system_running
                        and bridge.cca
                        and getattr(bridge.cca, "safety_rules", None)
                    ):
                        runtime_rules_source.text = (
                            "Source: live runtime rules from the active controller"
                        )
                    elif rules and selected_safety_file:
                        runtime_rules_source.text = (
                            f"Source: selected safety file {Path(selected_safety_file).name}"
                        )
                    else:
                        runtime_rules_source.text = "Source: no runtime or verified safety loaded"
                    rows = []
                    for i, r in enumerate(rules):
                        rows.append(
                            {
                                "id": r.get("id", f"R{i}"),
                                "raw_text": r.get("raw_text", r.get("text", str(r))),
                                "constraint_type": r.get("constraint_type", ""),
                                "generated_interpretation": r.get(
                                    "generated_interpretation", r.get("ltlf_plain_feedback", "")
                                ),
                                "ltlf": r.get("ltlf", r.get("formula", "")),
                            }
                        )
                    runtime_rules_table.rows = rows

                _managed_timer(5.0, _refresh_runtime_safety_rules)

            # ── Runtime Safety State ────────────────────────────────────
            with ui.card().classes("w-full"):
                with ui.expansion(
                    "Runtime Safety State",
                    icon="shield",
                    value=False,
                ).classes("w-full"):
                    runtime_safety_state = ui.code("{}", language="json").classes("w-full")

                def _refresh_runtime_safety_state():
                    ss = bridge.get_safety_state()
                    runtime_safety_state.content = (
                        json.dumps(ss, indent=2, default=str) if ss else "{}"
                    )

                _managed_timer(5.0, _refresh_runtime_safety_state)

            # ── Replan / Recovery ───────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Replan / Recovery").classes("text-lg font-semibold mb-2")
                ui.label(
                    "Runtime DES recovery stays inside one workflow: DES search, narrow LLM bridge, plan validation, then human intervention if needed."
                ).classes("text-xs text-slate-500 mb-2")
                runtime_recovery_container = ui.column().classes("w-full gap-3")

                guidance_buffers: dict[str, str] = {}
                bridge_feedback_buffers: dict[str, str] = {}
                preprogrammed_bridge_buffers: dict[str, str] = {}
                action_feedback_buffers: dict[str, dict[str, str]] = {}
                pending_action_buffers: dict[str, str] = {}

                def _json_text(value: Any) -> str:
                    if isinstance(value, str):
                        return value
                    try:
                        return json.dumps(value, indent=2, default=str)
                    except Exception:
                        return str(value)

                def _preview_text(value: Any, *, max_chars: int = 1400) -> str:
                    text = str(value or "").strip()
                    if len(text) <= max_chars:
                        return text
                    return f"{text[:max_chars].rstrip()}\n...[truncated]"

                def _set_action_feedback(product_jid: str, kind: str, text: str) -> None:
                    jid = str(product_jid or "").strip()
                    message = str(text or "").strip()
                    if not jid:
                        return
                    if not message:
                        action_feedback_buffers.pop(jid, None)
                        try:
                            client.safe_invoke(_refresh_runtime_recovery_panel)
                        except Exception:
                            pass
                        return
                    action_feedback_buffers[jid] = {
                        "kind": str(kind or "info").strip().lower() or "info",
                        "text": message,
                    }
                    try:
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception:
                        pass

                def _clear_pending_runtime_action(product_jid: str) -> None:
                    jid = str(product_jid or "").strip()
                    if not jid:
                        return
                    pending_action_buffers.pop(jid, None)
                    try:
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception:
                        pass

                def _queue_runtime_action(
                    product_jid: str,
                    *,
                    action_label: str,
                    runner: Callable[[str], asyncio.Future[Any] | asyncio.Task[Any] | Any],
                ) -> None:
                    jid = str(product_jid or "").strip()
                    label = str(action_label or "Action").strip() or "Action"
                    pending_label = str(pending_action_buffers.get(jid, "")).strip()
                    if pending_label:
                        logger.warning(
                            "[Dashboard] Runtime recovery action ignored while pending: requested=%s pending=%s product=%s",
                            label,
                            pending_label,
                            jid,
                        )
                        _set_action_feedback(
                            jid,
                            "warning",
                            f"{pending_label} is already in progress.",
                        )
                        _notify(f"{pending_label} is already in progress.", type="warning")
                        return
                    pending_action_buffers[jid] = label
                    logger.info(
                        "[Dashboard] Runtime recovery action requested: %s product=%s", label, jid
                    )
                    _set_action_feedback(jid, "info", f"{label} requested...")
                    _notify(f"{label} requested.", type="info")
                    asyncio.create_task(runner(jid))

                def _apply_runtime_action_result(
                    product_jid: str,
                    result: Any,
                    *,
                    default_kind: str,
                    default_text: str,
                ) -> None:
                    feedback = (
                        dict(result.get("action_feedback") or {})
                        if isinstance(result, dict)
                        else {}
                    )
                    kind = (
                        str(feedback.get("kind") or default_kind or "info").strip().lower()
                        or "info"
                    )
                    text = str(feedback.get("text") or default_text or "").strip()
                    if not text:
                        return
                    _set_action_feedback(product_jid, kind, text)
                    notify_type = {
                        "warning": "warning",
                        "negative": "negative",
                        "error": "negative",
                        "positive": "positive",
                        "success": "positive",
                    }.get(kind, "info")
                    _notify(text, type=notify_type)

                def _bridge_task_rows_for_display(
                    task_ids: list[str],
                    *,
                    node_lookup: dict[str, dict[str, Any]],
                    task_states: dict[str, str],
                    fallback_rows: list[dict[str, Any]] | None = None,
                ) -> list[dict[str, Any]]:
                    rows: list[dict[str, Any]] = []
                    fallback_lookup = {
                        str(row.get("id", "")).strip(): row
                        for row in (fallback_rows or [])
                        if isinstance(row, dict) and str(row.get("id", "")).strip()
                    }
                    seen: set[str] = set()
                    for task_id in task_ids:
                        task_key = str(task_id or "").strip()
                        if not task_key or task_key in seen:
                            continue
                        seen.add(task_key)
                        row = dict(fallback_lookup.get(task_key) or {})
                        node = node_lookup.get(task_key)
                        if isinstance(node, dict):
                            row.update(
                                {
                                    "id": task_key,
                                    "function_name": str(node.get("function_name", "")).strip(),
                                    "resource_jid": str(node.get("resource_jid", "")).strip(),
                                    "status": str(
                                        task_states.get(
                                            task_key,
                                            node.get("status", ""),
                                        )
                                        or ""
                                    ).strip(),
                                    "predecessors": list(node.get("predecessors") or []),
                                    "successors": list(node.get("successors") or []),
                                    "params": dict(node.get("params") or {}),
                                    "bridge_sequence_id": str(
                                        node.get("bridge_sequence_id", "")
                                    ).strip(),
                                    "bridge_sequence_index": int(
                                        node.get("bridge_sequence_index") or 0
                                    ),
                                    "bridge_sequence_length": int(
                                        node.get("bridge_sequence_length") or 0
                                    ),
                                    "primary_obligation": dict(
                                        node.get("primary_obligation") or {}
                                    ),
                                    "change_reason": str(node.get("change_reason", "")).strip(),
                                }
                            )
                        else:
                            row.setdefault("id", task_key)
                            row.setdefault(
                                "status", str(task_states.get(task_key, "missing") or "missing")
                            )
                        rows.append(row)
                    return rows

                def _status_color(status: str) -> str:
                    key = str(status or "").strip().lower()
                    if key == "resolved":
                        return "green"
                    if key == "bridge_ready":
                        return "orange"
                    if key == "human_required":
                        return "red"
                    if key in {"des_search", "llm_bridge", "validating"}:
                        return "blue"
                    return "grey"

                def _status_label(status: str) -> str:
                    labels = {
                        "idle": "Idle",
                        "des_search": "DES search",
                        "bridge_ready": "Bridge ready",
                        "llm_bridge": "Bridge proposal",
                        "validating": "Plan validation",
                        "human_required": "Human intervention",
                        "resolved": "Resolved",
                    }
                    key = str(status or "").strip().lower()
                    return labels.get(key, key.replace("_", " ").title() or "Unknown")

                def _resolution_label(value: str) -> str:
                    labels = {
                        "none": "In progress",
                        "des_only": "DES only",
                        "des_with_llm_bridge": "DES + LLM bridge",
                        "human_required": "Human required",
                    }
                    key = str(value or "").strip().lower()
                    return labels.get(key, key.replace("_", " ").title() or "Unknown")

                def _resolution_color(value: str) -> str:
                    key = str(value or "").strip().lower()
                    if key in {"des_only", "des_with_llm_bridge"}:
                        return "green"
                    if key == "human_required":
                        return "red"
                    return "grey"

                def _stage_badges(recovery: dict[str, Any]) -> list[tuple[str, str]]:
                    status = str(recovery.get("status", "idle") or "idle").strip().lower()
                    used_bridge = bool(recovery.get("used_llm_bridge", False))
                    badges: list[tuple[str, str]] = []
                    stages = [
                        ("des_search", "DES search"),
                        ("bridge_ready", "Bridge review"),
                        ("llm_bridge", "Bridge proposal"),
                        ("validating", "Plan validation"),
                        ("human_required", "Human intervention"),
                    ]
                    for key, label in stages:
                        color = "grey"
                        if status == "resolved":
                            if (
                                key == "des_search"
                                or key == "bridge_ready"
                                and str(recovery.get("bridge_approval_state", "none") or "none")
                                .strip()
                                .lower()
                                in {"ready", "pending", "approved"}
                                or key == "llm_bridge"
                                and used_bridge
                                or key == "validating"
                            ):
                                color = "green"
                        elif status == key:
                            if key == "human_required":
                                color = "red"
                            elif key == "bridge_ready":
                                color = "orange"
                            else:
                                color = "blue"
                        elif (
                            key == "des_search"
                            and status
                            in {"bridge_ready", "llm_bridge", "validating", "human_required"}
                            or key == "bridge_ready"
                            and status in {"llm_bridge", "validating", "human_required"}
                            or key == "llm_bridge"
                            and used_bridge
                            and status in {"validating", "human_required"}
                            or key == "validating"
                            and status == "human_required"
                        ):
                            color = "green"
                        badges.append((label, color))
                    return badges

                async def _submit_runtime_guidance(product_jid: str) -> None:
                    message = str(guidance_buffers.get(product_jid, "")).strip()
                    if not message:
                        _set_action_feedback(product_jid, "warning", "Operator guidance is empty.")
                        _clear_pending_runtime_action(product_jid)
                        _notify("Operator guidance is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.submit_runtime_recovery_guidance,
                            product_jid,
                            message,
                        )
                        guidance_buffers[product_jid] = ""
                        _set_action_feedback(product_jid, "positive", "Operator guidance recorded.")
                        _notify("Operator guidance recorded.", type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to record guidance: {exc}",
                        )
                        _notify(f"Failed to record guidance: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _retry_runtime_des(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.retry_runtime_recovery_des, product_jid
                        )
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text="Runtime DES recovery retry started.",
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to retry DES recovery: {exc}",
                        )
                        _notify(f"Failed to retry DES recovery: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _run_runtime_bridge(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.generate_runtime_bridge_proposal,
                            product_jid,
                        )
                        approval_state = (
                            str(
                                result.get("bridge_approval_state", "none")
                                if isinstance(result, dict)
                                else "none"
                            )
                            .strip()
                            .lower()
                        )
                        if approval_state == "outline_pending":
                            feedback_text = "Bridge outline is ready for approval."
                            notify_text = "Bridge outline ready."
                        elif approval_state == "primitive_pending":
                            feedback_text = "Bridge primitives are ready for approval."
                            notify_text = "Bridge primitives ready."
                        elif approval_state == "pending":
                            feedback_text = "Bridge proposal is ready for final approval."
                        elif isinstance(result, dict) and isinstance(
                            result.get("action_feedback"), dict
                        ):
                            feedback_text = ""
                        else:
                            feedback_text = "Live bridge reasoning ran from the prepared session."
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text=feedback_text,
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to start bridge reasoning: {exc}",
                        )
                        _notify(f"Failed to start bridge reasoning: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _load_runtime_bridge_archive(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.load_runtime_bridge_archive_proposal,
                            product_jid,
                        )
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text="Archived bridge proposal loaded. Final approval is still required before execution starts.",
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to load archived bridge proposal: {exc}",
                        )
                        _notify(f"Failed to load archived bridge proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _approve_runtime_bridge_outline(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.approve_runtime_bridge_outline,
                            product_jid,
                        )
                        approval_state = (
                            str(
                                result.get("bridge_approval_state", "none")
                                if isinstance(result, dict)
                                else "none"
                            )
                            .strip()
                            .lower()
                        )
                        if approval_state == "primitive_pending":
                            feedback_text = (
                                "Outline approved. Bridge primitives are ready for review."
                            )
                            notify_text = "Bridge primitives ready."
                        elif approval_state == "pending":
                            feedback_text = (
                                "Outline approved. Bridge proposal is ready for final approval."
                            )
                        elif isinstance(result, dict) and isinstance(
                            result.get("action_feedback"), dict
                        ):
                            feedback_text = ""
                        else:
                            feedback_text = "Outline approved. Live bridge reasoning continued."
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text=feedback_text,
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to approve outline checkpoint: {exc}",
                        )
                        _notify(f"Failed to approve outline checkpoint: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _refine_runtime_bridge_outline(product_jid: str) -> None:
                    feedback = str(bridge_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Outline refinement guidance is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Outline refinement guidance is empty.", type="warning")
                        return
                    try:
                        result = await asyncio.to_thread(
                            bridge.refine_runtime_bridge_outline,
                            product_jid,
                            feedback,
                        )
                        bridge_feedback_buffers[product_jid] = ""
                        approval_state = (
                            str(
                                result.get("bridge_approval_state", "none")
                                if isinstance(result, dict)
                                else "none"
                            )
                            .strip()
                            .lower()
                        )
                        if approval_state == "outline_pending":
                            feedback_text = "Outline refinement applied. A regenerated outline is ready for review."
                            notify_text = "Outline regenerated."
                        else:
                            feedback_text = "Outline refinement applied."
                            notify_text = "Outline refinement applied."
                        _set_action_feedback(product_jid, "positive", feedback_text)
                        _notify(notify_text, type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to refine outline checkpoint: {exc}",
                        )
                        _notify(f"Failed to refine outline checkpoint: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _reject_runtime_bridge_outline(product_jid: str) -> None:
                    feedback = str(bridge_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Outline rejection feedback is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Outline rejection feedback is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.reject_runtime_bridge_outline,
                            product_jid,
                            feedback,
                        )
                        bridge_feedback_buffers[product_jid] = ""
                        _set_action_feedback(
                            product_jid,
                            "positive",
                            "Outline rejected. Runtime recovery is paused for manual intervention.",
                        )
                        _notify("Outline rejected.", type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to reject outline checkpoint: {exc}",
                        )
                        _notify(f"Failed to reject outline checkpoint: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _approve_runtime_bridge_primitives(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.approve_runtime_bridge_primitives,
                            product_jid,
                        )
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text="Bridge primitives approved. Final proposal review is ready.",
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to approve primitive checkpoint: {exc}",
                        )
                        _notify(f"Failed to approve primitive checkpoint: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _refine_runtime_bridge_primitives(product_jid: str) -> None:
                    feedback = str(bridge_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Primitive refinement guidance is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Primitive refinement guidance is empty.", type="warning")
                        return
                    try:
                        result = await asyncio.to_thread(
                            bridge.refine_runtime_bridge_primitives,
                            product_jid,
                            feedback,
                        )
                        bridge_feedback_buffers[product_jid] = ""
                        approval_state = (
                            str(
                                result.get("bridge_approval_state", "none")
                                if isinstance(result, dict)
                                else "none"
                            )
                            .strip()
                            .lower()
                        )
                        if approval_state == "primitive_pending":
                            feedback_text = "Primitive refinement applied. A regenerated primitive program is ready for review."
                            notify_text = "Primitive program regenerated."
                        elif approval_state == "pending":
                            feedback_text = (
                                "Primitive refinement applied. Final proposal review is ready."
                            )
                            notify_text = "Bridge proposal ready."
                        else:
                            feedback_text = "Primitive refinement applied."
                            notify_text = "Primitive refinement applied."
                        _set_action_feedback(product_jid, "positive", feedback_text)
                        _notify(notify_text, type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to refine primitive checkpoint: {exc}",
                        )
                        _notify(f"Failed to refine primitive checkpoint: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _reject_runtime_bridge_primitives(product_jid: str) -> None:
                    feedback = str(bridge_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Primitive rejection feedback is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Primitive rejection feedback is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.reject_runtime_bridge_primitives,
                            product_jid,
                            feedback,
                        )
                        bridge_feedback_buffers[product_jid] = ""
                        _set_action_feedback(
                            product_jid,
                            "positive",
                            "Primitive program rejected. Runtime recovery is paused for manual intervention.",
                        )
                        _notify("Primitive program rejected.", type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to reject primitive checkpoint: {exc}",
                        )
                        _notify(f"Failed to reject primitive checkpoint: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _approve_runtime_bridge(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.approve_runtime_bridge_proposal, product_jid
                        )
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text="Bridge proposal approved. Runtime plan validation started.",
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to approve bridge proposal: {exc}",
                        )
                        _notify(f"Failed to approve bridge proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _reject_runtime_bridge(product_jid: str) -> None:
                    feedback = str(bridge_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Bridge rejection feedback is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Bridge rejection feedback is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.reject_runtime_bridge_proposal,
                            product_jid,
                            feedback,
                        )
                        bridge_feedback_buffers[product_jid] = ""
                        _set_action_feedback(
                            product_jid,
                            "positive",
                            "Bridge proposal rejected. Session paused for manual refinement.",
                        )
                        _notify("Bridge proposal rejected.", type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to reject bridge proposal: {exc}",
                        )
                        _notify(f"Failed to reject bridge proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _refine_runtime_bridge(product_jid: str) -> None:
                    feedback = str(bridge_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Refinement guidance is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Refinement guidance is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.submit_runtime_recovery_guidance,
                            product_jid,
                            feedback,
                        )
                        await asyncio.to_thread(
                            bridge.generate_runtime_bridge_proposal, product_jid
                        )
                        bridge_feedback_buffers[product_jid] = ""
                        _set_action_feedback(
                            product_jid,
                            "positive",
                            "Bridge refinement submitted. Re-running bounded bridge reasoning.",
                        )
                        _notify("Bridge refinement submitted.", type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to refine bridge proposal: {exc}",
                        )
                        _notify(f"Failed to refine bridge proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                def _bridge_mode_label(value: str) -> str:
                    key = str(value or "").strip().lower()
                    return _BRIDGE_MODE_OPTIONS.get(key, key.replace("_", " ").title() or "Auto")

                def _bridge_validation_policy_label(value: str) -> str:
                    key = str(value or "").strip().lower()
                    return _BRIDGE_VALIDATION_POLICY_OPTIONS.get(
                        key,
                        key.replace("_", " ").title() or "Validated",
                    )

                def _bridge_outline_rows(
                    bridge_debug: dict[str, Any] | None,
                ) -> list[dict[str, Any]]:
                    debug = bridge_debug if isinstance(bridge_debug, dict) else {}
                    final_output = debug.get("final_output")
                    if isinstance(final_output, dict):
                        rows = final_output.get("transition_trace")
                        if isinstance(rows, list) and rows:
                            return [row for row in rows if isinstance(row, dict)]
                    session = debug.get("multi_turn_session")
                    if isinstance(session, dict):
                        rows = session.get("accepted_outline_prefix")
                        if isinstance(rows, list) and rows:
                            return [row for row in rows if isinstance(row, dict)]
                    rows = debug.get("transition_trace")
                    if isinstance(rows, list) and rows:
                        return [row for row in rows if isinstance(row, dict)]
                    return []

                def _bridge_primitive_rows(
                    bridge_debug: dict[str, Any] | None,
                ) -> list[dict[str, Any]]:
                    debug = bridge_debug if isinstance(bridge_debug, dict) else {}
                    final_output = debug.get("final_output")
                    if isinstance(final_output, dict):
                        rows = final_output.get("accepted_primitive_program")
                        if isinstance(rows, list) and rows:
                            return [row for row in rows if isinstance(row, dict)]
                    session = debug.get("multi_turn_session")
                    if isinstance(session, dict):
                        rows = session.get("accepted_primitive_program")
                        if isinstance(rows, list) and rows:
                            return [row for row in rows if isinstance(row, dict)]
                    return []

                def _bridge_sequence_for_display(recovery: dict[str, Any]) -> dict[str, Any] | None:
                    active = recovery.get("active_bridge_sequence")
                    if isinstance(active, dict):
                        return active
                    last_completed = recovery.get("last_completed_bridge_sequence")
                    if isinstance(last_completed, dict):
                        return last_completed
                    return None

                def _bridge_sequence_progress(sequence: dict[str, Any] | None) -> dict[str, Any]:
                    if not isinstance(sequence, dict):
                        return {
                            "compiled": 0,
                            "dispatched": 0,
                            "completed": 0,
                            "state": "",
                            "mode": "",
                            "archive_path": "",
                            "validation_policy": "",
                        }
                    bridge_task_ids = [
                        str(task_id or "").strip()
                        for task_id in (sequence.get("bridge_task_ids") or [])
                        if str(task_id or "").strip()
                    ]
                    dispatched_task_ids = [
                        str(task_id or "").strip()
                        for task_id in (sequence.get("dispatched_bridge_task_ids") or [])
                        if str(task_id or "").strip()
                    ]
                    completed_task_ids = [
                        str(task_id or "").strip()
                        for task_id in (sequence.get("completed_bridge_task_ids") or [])
                        if str(task_id or "").strip()
                    ]
                    state = str(sequence.get("state") or "").strip().lower()
                    return {
                        "compiled": len(bridge_task_ids),
                        "dispatched": len(dispatched_task_ids),
                        "completed": len(completed_task_ids),
                        "state": state,
                        "mode": str(sequence.get("source_mode") or "").strip().lower(),
                        "archive_path": str(sequence.get("source_archive_path") or "").strip(),
                        "validation_policy": str(sequence.get("validation_policy") or "")
                        .strip()
                        .lower(),
                    }

                def _refresh_runtime_recovery_panel() -> None:
                    runtime_recovery_container.clear()
                    recoveries = bridge.get_runtime_recoveries()
                    if not recoveries:
                        with runtime_recovery_container:
                            ui.label("No runtime recovery sessions.").classes(
                                "text-slate-400 italic"
                            )
                        return

                    nodes = _preview_or_runtime_nodes()
                    task_states = _preview_or_runtime_task_states(nodes)
                    node_lookup = {
                        str(node.get("id") or node.get("task_id") or "").strip(): node
                        for node in nodes
                        if isinstance(node, dict)
                        and str(node.get("id") or node.get("task_id") or "").strip()
                    }

                    with runtime_recovery_container:
                        for recovery in recoveries:
                            if not isinstance(recovery, dict):
                                continue
                            product_jid = str(recovery.get("product_jid", "")).strip()
                            product_name = str(
                                recovery.get("product_name", product_jid)
                                or product_jid
                                or "product"
                            )
                            status = str(recovery.get("status", "idle") or "idle")
                            resolution = str(recovery.get("resolution_class", "none") or "none")
                            message = (
                                str(recovery.get("message", "")).strip() or "No recovery activity."
                            )
                            attempts_used = int(recovery.get("attempts_used", 0) or 0)
                            attempts_max = int(recovery.get("attempts_max", 0) or 0)
                            violated_rules = list(recovery.get("violated_rules") or [])
                            witness_count = int(recovery.get("witness_count", 0) or 0)
                            operator_guidance = str(
                                guidance_buffers.get(
                                    product_jid,
                                    recovery.get("operator_guidance", ""),
                                )
                                or ""
                            )
                            bridge_feedback = str(
                                bridge_feedback_buffers.get(product_jid, "") or ""
                            )
                            selected_preprogrammed = str(
                                preprogrammed_bridge_buffers.get(
                                    product_jid,
                                    "recover_lg_v1",
                                )
                                or "recover_lg_v1"
                            )
                            guidance_buffers[product_jid] = operator_guidance
                            bridge_feedback_buffers[product_jid] = bridge_feedback
                            preprogrammed_bridge_buffers[product_jid] = selected_preprogrammed
                            action_feedback = (
                                action_feedback_buffers.get(product_jid)
                                if isinstance(action_feedback_buffers.get(product_jid), dict)
                                else None
                            )
                            pending_action_label = str(
                                pending_action_buffers.get(product_jid, "") or ""
                            ).strip()
                            bridge_proposal = recovery.get("bridge_proposal")
                            bridge_debug = (
                                recovery.get("bridge_debug")
                                if isinstance(recovery.get("bridge_debug"), dict)
                                else None
                            )
                            bridge_approval_state = str(
                                recovery.get("bridge_approval_state", "none") or "none"
                            ).strip()
                            bridge_mode = (
                                str(recovery.get("bridge_mode", "pre_ran") or "pre_ran")
                                .strip()
                                .lower()
                                or "pre_ran"
                            )
                            validation_policy = (
                                str(recovery.get("validation_policy", "validated") or "validated")
                                .strip()
                                .lower()
                                or "validated"
                            )
                            bridge_stage = (
                                str(recovery.get("bridge_stage", "none") or "none").strip().lower()
                                or "none"
                            )
                            selected_archive_path = str(
                                recovery.get("selected_archive_path", "") or ""
                            ).strip()
                            selected_archive_label = str(
                                recovery.get("selected_archive_label", "") or ""
                            ).strip()
                            artifact_directory = str(
                                recovery.get("artifact_directory")
                                or (bridge_debug or {}).get("artifact_directory")
                                or (bridge_debug or {}).get("per_turn_debug_dir")
                                or ""
                            ).strip()
                            outline_rows = _bridge_outline_rows(bridge_debug)
                            primitive_rows = _bridge_primitive_rows(bridge_debug)
                            active_bridge_sequence = (
                                recovery.get("active_bridge_sequence")
                                if isinstance(recovery.get("active_bridge_sequence"), dict)
                                else None
                            )
                            last_completed_bridge_sequence = (
                                recovery.get("last_completed_bridge_sequence")
                                if isinstance(recovery.get("last_completed_bridge_sequence"), dict)
                                else None
                            )
                            bridge_sequence_display = (
                                active_bridge_sequence
                                if isinstance(active_bridge_sequence, dict)
                                else last_completed_bridge_sequence
                            )
                            bridge_sequence_progress = _bridge_sequence_progress(
                                bridge_sequence_display
                            )
                            active_bridge_state = (
                                str((active_bridge_sequence or {}).get("state") or "")
                                .strip()
                                .lower()
                            )
                            bridge_reentry_locked = active_bridge_state in {"approved", "executing"}
                            controls_locked = bool(pending_action_label)

                            with ui.card().classes("w-full bg-slate-50"):
                                with ui.row().classes(
                                    "w-full items-center justify-between gap-2 flex-wrap"
                                ):
                                    ui.label(product_name).classes("font-semibold")
                                    with ui.row().classes("items-center gap-2 flex-wrap"):
                                        ui.badge(_status_label(status)).props(
                                            f"color={_status_color(status)}"
                                        )
                                        ui.badge(_resolution_label(resolution)).props(
                                            f"color={_resolution_color(resolution)}"
                                        )
                                        ui.badge(f"Mode {_bridge_mode_label(bridge_mode)}").props(
                                            "color=blue-grey"
                                        )
                                        ui.badge(
                                            f"Recovery Safety {_bridge_validation_policy_label(validation_policy)}"
                                        ).props("color=deep-orange")
                                        if bridge_stage not in {"", "none"}:
                                            ui.badge(
                                                f"Stage {bridge_stage.replace('_', ' ')}"
                                            ).props("color=purple")
                                        if bridge_approval_state not in {"", "none"}:
                                            ui.badge(
                                                f"Bridge {bridge_approval_state.replace('_', ' ')}"
                                            ).props("color=teal")

                                with ui.row().classes(
                                    "items-center gap-2 flex-wrap text-xs text-slate-600"
                                ):
                                    ui.label(
                                        f"Trigger: {str(recovery.get('trigger', '') or 'n/a')}"
                                    )
                                    ui.label(
                                        f"Failed task: {str(recovery.get('failed_task_id', '') or 'n/a')}"
                                    )
                                    ui.label(
                                        f"Recovery Safety: {_bridge_validation_policy_label(validation_policy)}"
                                    )
                                    if selected_archive_label:
                                        ui.label(f"Archived run: {selected_archive_label}")
                                    elif bridge_mode == "pre_ran" and selected_archive_path:
                                        ui.label(f"Archived run: {selected_archive_path}")
                                    if artifact_directory:
                                        ui.label(f"Artifacts: {artifact_directory}")

                                with ui.row().classes("items-center gap-2 flex-wrap mt-1"):
                                    for stage_label, stage_color in _stage_badges(recovery):
                                        ui.badge(stage_label).props(f"color={stage_color}")

                                ui.label(message).classes("text-sm mt-2")
                                ui.label(
                                    f"Attempts: {attempts_used}/{attempts_max} | "
                                    f"Witnesses: {witness_count} | "
                                    f"Violated rules: {', '.join(violated_rules) if violated_rules else 'none'}"
                                ).classes("text-xs text-slate-600")
                                if bridge_sequence_display:
                                    sequence_label = (
                                        "Bridge sequence"
                                        if isinstance(active_bridge_sequence, dict)
                                        else "Last bridge sequence"
                                    )
                                    source_mode_label = _bridge_mode_label(
                                        bridge_sequence_progress.get("mode") or bridge_mode
                                    )
                                    status_line = (
                                        f"{sequence_label}: "
                                        f"{str(bridge_sequence_progress.get('state') or 'unknown').replace('_', ' ')} | "
                                        f"Compiled: {int(bridge_sequence_progress.get('compiled') or 0)} | "
                                        f"Dispatched: {int(bridge_sequence_progress.get('dispatched') or 0)} | "
                                        f"Completed: {int(bridge_sequence_progress.get('completed') or 0)} | "
                                        f"Mode: {source_mode_label}"
                                    )
                                    archive_path = str(
                                        bridge_sequence_progress.get("archive_path") or ""
                                    ).strip()
                                    sequence_validation_policy = str(
                                        bridge_sequence_progress.get("validation_policy")
                                        or validation_policy
                                    ).strip()
                                    if archive_path:
                                        status_line += f" | Archive source: {archive_path}"
                                    if sequence_validation_policy:
                                        status_line += (
                                            " | Recovery Safety: "
                                            f"{_bridge_validation_policy_label(sequence_validation_policy)}"
                                        )
                                    ui.label(status_line).classes("text-xs text-slate-600 mt-1")
                                if bridge_reentry_locked:
                                    ui.label(
                                        "Bridge execution is already active. Bridge approval and DES retry controls are locked until it finishes."
                                    ).classes("text-xs text-orange-700 mt-1")
                                if action_feedback and str(action_feedback.get("text", "")).strip():
                                    tone = {
                                        "positive": "bg-green-50 text-green-700",
                                        "warning": "bg-orange-50 text-orange-700",
                                        "negative": "bg-red-50 text-red-700",
                                        "info": "bg-blue-50 text-blue-700",
                                    }.get(
                                        str(action_feedback.get("kind", "info") or "info")
                                        .strip()
                                        .lower(),
                                        "bg-slate-100 text-slate-700",
                                    )
                                    ui.label(str(action_feedback.get("text", "")).strip()).classes(
                                        f"w-full mt-2 px-3 py-2 rounded text-xs {tone}"
                                    )
                                if pending_action_label:
                                    ui.label(f"Pending action: {pending_action_label}").classes(
                                        "w-full mt-2 px-3 py-2 rounded text-xs bg-amber-50 text-amber-700"
                                    )

                                if status == "bridge_ready":
                                    if bridge_mode == "pre_ran":
                                        ui.label(
                                            "Pre-ran mode is selected. If a valid archived run is selected, it will auto-load, run the ordinary CCA runtime plan validation on the merged nominal + recovery plan before execution, and keep Recovery Safety Check enabled for recovery_safety runtime enforcement. Use the button below only to reload it manually while paused."
                                            if validation_policy == "validated"
                                            else "Pre-ran mode is selected with No Recovery Safety Check. If a valid archived run is selected, it will auto-load, run the ordinary CCA runtime plan validation on the merged nominal + recovery plan before execution, and keep recovery_safety runtime enforcement disabled. Use the button below only to reload it manually while paused."
                                        ).classes("text-xs text-orange-700 mt-2")
                                        if selected_archive_label:
                                            ui.label(
                                                f"Selected archived run: {selected_archive_label}"
                                            ).classes("text-xs text-slate-600")
                                        elif selected_archive_path:
                                            ui.label(
                                                f"Selected archived run: {selected_archive_path}"
                                            ).classes("text-xs text-slate-600")
                                        else:
                                            ui.label(
                                                "Choose an Archived Bridge Run from the System Control panel."
                                            ).classes("text-xs text-slate-600")
                                        load_archive_btn = ui.button(
                                            "Load archived bridge",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Load archived bridge",
                                                runner=_load_runtime_bridge_archive,
                                            ),
                                            icon="history",
                                        ).props("color=indigo")
                                        load_archive_btn.set_enabled(
                                            bool(selected_archive_path)
                                            and not bridge_reentry_locked
                                            and not controls_locked
                                        )
                                    else:
                                        run_label = "Run live bridge"
                                        helper_text = (
                                            "Bridge session is prepared. Run live bridge reasoning to reach the next review stage."
                                            if bridge_mode == "manual"
                                            else "Auto mode normally starts live bridge reasoning immediately. Use this to run it again from the prepared session."
                                        )
                                        ui.label(helper_text).classes(
                                            "text-xs text-orange-700 mt-2"
                                        )
                                        with ui.row().classes("gap-2 mt-2 flex-wrap"):
                                            run_live_btn = ui.button(
                                                run_label,
                                                on_click=lambda jid=product_jid: _queue_runtime_action(
                                                    jid,
                                                    action_label=run_label,
                                                    runner=_run_runtime_bridge,
                                                ),
                                                icon="smart_toy",
                                            ).props("color=indigo")
                                            run_live_btn.set_enabled(
                                                not bridge_reentry_locked and not controls_locked
                                            )

                                if (
                                    status == "llm_bridge"
                                    and bridge_approval_state == "outline_pending"
                                ):
                                    with ui.expansion(
                                        "Outline checkpoint",
                                        icon="route",
                                        value=True,
                                    ).classes("w-full mt-2"):
                                        if outline_rows:
                                            for idx, row in enumerate(outline_rows, start=1):
                                                if not isinstance(row, dict):
                                                    continue
                                                summary = str(
                                                    row.get("description")
                                                    or row.get("event_name")
                                                    or row.get("action_name")
                                                    or row.get("outline_id")
                                                    or f"outline_{idx}"
                                                ).strip()
                                                resource_name = (
                                                    str(row.get("resource_jid") or "n/a").strip()
                                                    or "n/a"
                                                )
                                                part_name = str(row.get("part_name") or "").strip()
                                                outline_id = str(
                                                    row.get("outline_id") or ""
                                                ).strip()
                                                ui.label(f"{idx}. {summary}").classes(
                                                    "text-sm font-medium"
                                                )
                                                details = f"Resource: {resource_name}"
                                                if part_name:
                                                    details += f" | Part: {part_name}"
                                                if outline_id:
                                                    details += f" | Outline ID: {outline_id}"
                                                ui.label(details).classes("text-xs text-slate-600")
                                        else:
                                            ui.label(
                                                "No outline events were captured in the current bridge trace."
                                            ).classes("text-xs text-slate-600")
                                    ui.label(
                                        "Approve the outline to continue live primitive generation, refine it with operator guidance, or reject it."
                                    ).classes("text-xs text-indigo-700 mt-2")
                                    review_box = (
                                        ui.textarea(
                                            label="Outline refinement or rejection feedback",
                                            value=bridge_feedback,
                                        )
                                        .props("outlined autogrow")
                                        .classes("w-full mt-2")
                                    )
                                    review_box.on_value_change(
                                        lambda e,
                                        jid=product_jid: bridge_feedback_buffers.__setitem__(
                                            jid,
                                            str(e.value or ""),
                                        )
                                    )
                                    with ui.row().classes("gap-2 mt-2 flex-wrap"):
                                        approve_outline_btn = ui.button(
                                            "Approve outline",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Approve outline",
                                                runner=_approve_runtime_bridge_outline,
                                            ),
                                            icon="check_circle",
                                        ).props("color=green")
                                        refine_outline_btn = ui.button(
                                            "Refine outline",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Refine outline",
                                                runner=_refine_runtime_bridge_outline,
                                            ),
                                            icon="tune",
                                        ).props("color=amber")
                                        reject_outline_btn = ui.button(
                                            "Reject outline",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Reject outline",
                                                runner=_reject_runtime_bridge_outline,
                                            ),
                                            icon="cancel",
                                        ).props("color=red")
                                        for button in (
                                            approve_outline_btn,
                                            refine_outline_btn,
                                            reject_outline_btn,
                                        ):
                                            button.set_enabled(
                                                not bridge_reentry_locked and not controls_locked
                                            )

                                if (
                                    status == "llm_bridge"
                                    and bridge_approval_state == "primitive_pending"
                                ):
                                    with ui.expansion(
                                        "Primitive checkpoint",
                                        icon="precision_manufacturing",
                                        value=True,
                                    ).classes("w-full mt-2"):
                                        if primitive_rows:
                                            for idx, row in enumerate(primitive_rows, start=1):
                                                if not isinstance(row, dict):
                                                    continue
                                                summary = str(
                                                    row.get("description")
                                                    or row.get("event_name")
                                                    or row.get("outline_id")
                                                    or f"primitive_{idx}"
                                                ).strip()
                                                resource_name = (
                                                    str(row.get("resource_jid") or "n/a").strip()
                                                    or "n/a"
                                                )
                                                ui.label(
                                                    f"{idx}. {summary} -> {resource_name}"
                                                ).classes("text-sm font-medium")
                                                primitive_steps = [
                                                    step
                                                    for step in (row.get("primitive_steps") or [])
                                                    if isinstance(step, dict)
                                                ]
                                                if not primitive_steps:
                                                    ui.label(
                                                        "No primitive steps were captured for this outline event."
                                                    ).classes("text-xs text-slate-600")
                                                for step_idx, step in enumerate(
                                                    primitive_steps, start=1
                                                ):
                                                    primitive_name = str(
                                                        step.get("primitive")
                                                        or step.get("function_name")
                                                        or "primitive"
                                                    ).strip()
                                                    params_text = json.dumps(
                                                        step.get("params") or {},
                                                        default=str,
                                                    )
                                                    ui.label(
                                                        f"  {step_idx}. {primitive_name}({params_text})"
                                                    ).classes("text-xs font-mono text-slate-700")
                                        else:
                                            ui.label(
                                                "No primitive program rows were captured in the current bridge trace."
                                            ).classes("text-xs text-slate-600")
                                    ui.label(
                                        "Approve the primitive program to move to final proposal approval, refine it with operator guidance, or reject it."
                                    ).classes("text-xs text-indigo-700 mt-2")
                                    review_box = (
                                        ui.textarea(
                                            label="Primitive refinement or rejection feedback",
                                            value=bridge_feedback,
                                        )
                                        .props("outlined autogrow")
                                        .classes("w-full mt-2")
                                    )
                                    review_box.on_value_change(
                                        lambda e,
                                        jid=product_jid: bridge_feedback_buffers.__setitem__(
                                            jid,
                                            str(e.value or ""),
                                        )
                                    )
                                    with ui.row().classes("gap-2 mt-2 flex-wrap"):
                                        approve_primitives_btn = ui.button(
                                            "Approve primitives",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Approve primitives",
                                                runner=_approve_runtime_bridge_primitives,
                                            ),
                                            icon="check_circle",
                                        ).props("color=green")
                                        refine_primitives_btn = ui.button(
                                            "Refine primitives",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Refine primitives",
                                                runner=_refine_runtime_bridge_primitives,
                                            ),
                                            icon="tune",
                                        ).props("color=amber")
                                        reject_primitives_btn = ui.button(
                                            "Reject primitives",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Reject primitives",
                                                runner=_reject_runtime_bridge_primitives,
                                            ),
                                            icon="cancel",
                                        ).props("color=red")
                                        for button in (
                                            approve_primitives_btn,
                                            refine_primitives_btn,
                                            reject_primitives_btn,
                                        ):
                                            button.set_enabled(
                                                not bridge_reentry_locked and not controls_locked
                                            )

                                if status == "llm_bridge" and isinstance(bridge_proposal, dict):
                                    with ui.expansion(
                                        "Bridge proposal", icon="alt_route", value=True
                                    ).classes("w-full mt-2"):
                                        primary_obligation = bridge_proposal.get(
                                            "primary_obligation"
                                        )
                                        if isinstance(primary_obligation, dict):
                                            ui.label(
                                                "Primary obligation: "
                                                f"{str(primary_obligation.get('rule_id', '') or 'n/a')} on "
                                                f"{str(primary_obligation.get('resource_jid', '') or 'n/a')}"
                                            ).classes("text-xs text-slate-600")
                                        description = str(
                                            bridge_proposal.get("description", "")
                                        ).strip()
                                        rationale = str(
                                            bridge_proposal.get("rationale", "")
                                        ).strip()
                                        if description:
                                            ui.label(description).classes("text-sm")
                                        if rationale:
                                            ui.label(f"Rationale: {rationale}").classes(
                                                "text-xs text-slate-600"
                                            )
                                        macro_tasks = bridge_proposal.get("macro_tasks") or []
                                        if isinstance(macro_tasks, list) and macro_tasks:
                                            for idx, task in enumerate(macro_tasks, start=1):
                                                if not isinstance(task, dict):
                                                    continue
                                                macro_name = str(
                                                    task.get("macro_name", "") or f"macro_{idx}"
                                                )
                                                resource_name = str(
                                                    task.get("resource_jid", "") or "n/a"
                                                )
                                                ui.label(
                                                    f"{idx}. {macro_name} -> {resource_name}"
                                                ).classes("text-sm font-medium mt-2")
                                                ui.label(
                                                    "Expected start: "
                                                    f"{str(task.get('expected_start_state', '') or 'n/a')} | "
                                                    "Projected final: "
                                                    f"{str((task.get('projected_snapshot') or {}).get('current_state', '') or 'n/a')}"
                                                ).classes("text-xs text-slate-600")
                                                for step_idx, step in enumerate(
                                                    task.get("primitive_steps") or [], start=1
                                                ):
                                                    if not isinstance(step, dict):
                                                        continue
                                                    params_text = json.dumps(
                                                        step.get("params") or {}, default=str
                                                    )
                                                    ui.label(
                                                        f"  {step_idx}. {step.get('primitive', '')}({params_text})"
                                                    ).classes("text-xs font-mono text-slate-700")
                                        else:
                                            ui.label(
                                                f"Macro: {str(bridge_proposal.get('function_name', '') or 'unnamed')}"
                                            ).classes("text-sm font-medium")
                                            ui.label(
                                                f"Resource: {str(bridge_proposal.get('resource_jid', '') or 'n/a')}"
                                            ).classes("text-xs text-slate-600")
                                            for idx, step in enumerate(
                                                bridge_proposal.get("macro_steps") or [], start=1
                                            ):
                                                if not isinstance(step, dict):
                                                    continue
                                                params_text = json.dumps(
                                                    step.get("params") or {}, default=str
                                                )
                                                ui.label(
                                                    f"{idx}. {step.get('function_name', '')}({params_text})"
                                                ).classes("text-xs font-mono text-slate-700")
                                    if bridge_approval_state == "pending":
                                        ui.label(
                                            "This final bridge candidate has already passed deterministic validation and is waiting for operator approval."
                                        ).classes("text-xs text-indigo-700 mt-2")
                                        review_box = (
                                            ui.textarea(
                                                label="Refinement or rejection feedback",
                                                value=bridge_feedback,
                                            )
                                            .props("outlined autogrow")
                                            .classes("w-full mt-2")
                                        )
                                        review_box.on_value_change(
                                            lambda e,
                                            jid=product_jid: bridge_feedback_buffers.__setitem__(
                                                jid,
                                                str(e.value or ""),
                                            )
                                        )
                                        with ui.row().classes("gap-2 mt-2 flex-wrap"):
                                            approve_bridge_btn = ui.button(
                                                "Approve",
                                                on_click=lambda jid=product_jid: _queue_runtime_action(
                                                    jid,
                                                    action_label="Approve bridge proposal",
                                                    runner=_approve_runtime_bridge,
                                                ),
                                                icon="check_circle",
                                            ).props("color=green")
                                            refine_bridge_btn = ui.button(
                                                "Refine",
                                                on_click=lambda jid=product_jid: _queue_runtime_action(
                                                    jid,
                                                    action_label="Refine bridge proposal",
                                                    runner=_refine_runtime_bridge,
                                                ),
                                                icon="tune",
                                            ).props("color=amber")
                                            reject_bridge_btn = ui.button(
                                                "Reject",
                                                on_click=lambda jid=product_jid: _queue_runtime_action(
                                                    jid,
                                                    action_label="Reject bridge proposal",
                                                    runner=_reject_runtime_bridge,
                                                ),
                                                icon="cancel",
                                            ).props("color=red")
                                            for button in (
                                                approve_bridge_btn,
                                                refine_bridge_btn,
                                                reject_bridge_btn,
                                            ):
                                                button.set_enabled(
                                                    not bridge_reentry_locked
                                                    and not controls_locked
                                                )

                                if bridge_debug:
                                    debug_status = str(
                                        bridge_debug.get("status", "") or "n/a"
                                    ).strip()
                                    warning_messages = list(
                                        bridge_debug.get("warning_messages") or []
                                    )
                                    request_payload = (
                                        bridge_debug.get("request")
                                        if isinstance(bridge_debug.get("request"), dict)
                                        else {}
                                    )
                                    llm_inputs = (
                                        bridge_debug.get("llm_inputs")
                                        if isinstance(bridge_debug.get("llm_inputs"), dict)
                                        else {}
                                    )
                                    bridge_context = (
                                        bridge_debug.get("bridge_context")
                                        if isinstance(bridge_debug.get("bridge_context"), dict)
                                        else {}
                                    )
                                    modeled_check = (
                                        bridge_debug.get("modeled_continuation_check")
                                        if isinstance(
                                            bridge_debug.get("modeled_continuation_check"), dict
                                        )
                                        else {}
                                    )
                                    approval_info = (
                                        bridge_debug.get("approval")
                                        if isinstance(bridge_debug.get("approval"), dict)
                                        else {}
                                    )
                                    react_turns = list(bridge_debug.get("turns") or [])
                                    debug_prompt = str(bridge_debug.get("prompt", "") or "").strip()
                                    raw_response = str(
                                        bridge_debug.get("raw_response", "") or ""
                                    ).strip()
                                    bridge_proposal_debug = bridge_debug.get("bridge_proposal")
                                    request_ra = (
                                        str(
                                            request_payload.get("ra_jid")
                                            or llm_inputs.get("ra_jid")
                                            or "n/a"
                                        ).strip()
                                        or "n/a"
                                    )
                                    request_goal = (
                                        str(
                                            request_payload.get("goal_state")
                                            or llm_inputs.get("goal_state")
                                            or "n/a"
                                        ).strip()
                                        or "n/a"
                                    )
                                    request_parts = list(
                                        request_payload.get("P_id") or llm_inputs.get("P_id") or []
                                    )
                                    obligations = list(
                                        request_payload.get("obligation_targets")
                                        or llm_inputs.get("obligation_targets")
                                        or []
                                    )
                                    compiled_rows = _bridge_task_rows_for_display(
                                        list(approval_info.get("compiled_bridge_task_ids") or [])
                                        + list(
                                            (active_bridge_sequence or {}).get("bridge_task_ids")
                                            or []
                                        ),
                                        node_lookup=node_lookup,
                                        task_states=task_states,
                                        fallback_rows=list(
                                            approval_info.get("compiled_bridge_tasks") or []
                                        ),
                                    )
                                    with ui.expansion(
                                        "Bridge Debug (temporary)",
                                        icon="bug_report",
                                        value=False,
                                    ).classes("w-full mt-2"):
                                        with ui.row().classes("items-center gap-2 flex-wrap"):
                                            ui.badge(
                                                f"Trace: {(debug_status or 'unknown').replace('_', ' ')}"
                                            ).props("color=teal")
                                            ui.badge(
                                                "Primitive bridge"
                                                if bool(bridge_debug.get("primitive_mode"))
                                                else "Legacy bridge"
                                            ).props("color=indigo")
                                            if warning_messages:
                                                ui.badge(
                                                    f"Warnings: {len(warning_messages)}"
                                                ).props("color=orange")
                                        if modeled_check:
                                            ui.label(
                                                "Modeled continuation check: "
                                                f"{'accepted' if modeled_check.get('accepted') else 'rejected'}"
                                                + (
                                                    f" ({str(modeled_check.get('reason', '')).strip()})"
                                                    if str(modeled_check.get("reason", "")).strip()
                                                    else ""
                                                )
                                            ).classes("text-xs text-slate-600")
                                        ui.label(
                                            f"Focused resource: {request_ra} | Goal state: {request_goal} | "
                                            f"Open obligations: {len(obligations)} | Remaining parts: "
                                            f"{', '.join(str(item) for item in request_parts) if request_parts else 'none'}"
                                        ).classes("text-xs text-slate-600 mt-2")
                                        if status == "bridge_ready" and not raw_response:
                                            ui.label(
                                                "The LLM has not run yet. The prepared request, grounding context, and prompt "
                                                "below are exactly what will be used when you click the run button."
                                            ).classes("text-xs text-orange-700")
                                        if debug_prompt:
                                            ui.label("Prompt preview").classes(
                                                "text-xs font-medium mt-2"
                                            )
                                            ui.code(
                                                _preview_text(debug_prompt),
                                                language="text",
                                            ).classes("w-full text-xs")
                                        if raw_response:
                                            ui.label("Raw LLM output preview").classes(
                                                "text-xs font-medium mt-2"
                                            )
                                            ui.code(
                                                _preview_text(raw_response),
                                                language="text",
                                            ).classes("w-full text-xs")
                                        elif debug_status not in {"ready", "running"}:
                                            ui.label("No raw LLM output was captured.").classes(
                                                "text-xs text-slate-600 mt-2"
                                            )
                                        if bridge_proposal_debug:
                                            ui.label("Bridge proposal preview").classes(
                                                "text-xs font-medium mt-2"
                                            )
                                            ui.code(
                                                _preview_text(_json_text(bridge_proposal_debug)),
                                                language="json",
                                            ).classes("w-full text-xs")
                                        if react_turns:
                                            with ui.expansion(
                                                "ReAct trace",
                                                icon="route",
                                                value=False,
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(react_turns),
                                                    language="json",
                                                ).classes("w-full text-xs")
                                        if request_payload:
                                            with ui.expansion(
                                                "Bridge request inputs",
                                                icon="input",
                                                value=False,
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(request_payload), language="json"
                                                ).classes("w-full text-xs")
                                        if llm_inputs:
                                            with ui.expansion(
                                                "Prompt grounding context",
                                                icon="hub",
                                                value=False,
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(llm_inputs), language="json"
                                                ).classes("w-full text-xs")
                                        if bridge_context:
                                            with ui.expansion(
                                                "Primitive bridge context",
                                                icon="precision_manufacturing",
                                                value=False,
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(bridge_context), language="json"
                                                ).classes("w-full text-xs")
                                        if debug_prompt:
                                            with ui.expansion(
                                                "Prompt sent to LLM",
                                                icon="description",
                                                value=False,
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    debug_prompt,
                                                    language="text",
                                                ).classes("w-full text-xs")
                                        if raw_response:
                                            with ui.expansion(
                                                "Raw LLM output",
                                                icon="smart_toy",
                                                value=status in {"llm_bridge", "human_required"},
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    raw_response,
                                                    language="text",
                                                ).classes("w-full text-xs")
                                        with ui.expansion(
                                            "Bridge proposal",
                                            icon="rule",
                                            value=status in {"llm_bridge", "human_required"},
                                        ).classes("w-full mt-2"):
                                            if warning_messages:
                                                ui.code(
                                                    "\n".join(
                                                        f"- {msg}" for msg in warning_messages
                                                    ),
                                                    language="text",
                                                ).classes("w-full text-xs")
                                            if bridge_proposal_debug:
                                                ui.code(
                                                    _json_text(bridge_proposal_debug),
                                                    language="json",
                                                ).classes("w-full text-xs")
                                            else:
                                                ui.label(
                                                    "No compilable bridge proposal was produced."
                                                ).classes("text-xs text-slate-600")
                                        if compiled_rows:
                                            with ui.expansion(
                                                "Compiled bridge tasks",
                                                icon="account_tree",
                                                value=status in {"llm_bridge", "human_required"},
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(compiled_rows),
                                                    language="json",
                                                ).classes("w-full text-xs")
                                        if active_bridge_sequence:
                                            with ui.expansion(
                                                "Bridge execution state",
                                                icon="play_circle",
                                                value=status
                                                in {"llm_bridge", "human_required", "resolved"},
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(active_bridge_sequence),
                                                    language="json",
                                                ).classes("w-full text-xs")

                                history = list(recovery.get("history") or [])
                                if history:
                                    with ui.expansion(
                                        "Recent history", icon="history", value=False
                                    ).classes("w-full mt-2"):
                                        for item in history[-6:]:
                                            if not isinstance(item, dict):
                                                continue
                                            stamp = str(item.get("timestamp", "")).strip()
                                            item_status = _status_label(
                                                str(item.get("status", "")).strip()
                                            )
                                            item_message = str(item.get("message", "")).strip()
                                            ui.label(
                                                f"{stamp} | {item_status} | {item_message}"
                                            ).classes("text-xs text-slate-600")

                                if status == "human_required":
                                    guidance_box = (
                                        ui.textarea(
                                            label="Operator guidance",
                                            value=operator_guidance,
                                        )
                                        .props("outlined autogrow")
                                        .classes("w-full mt-2")
                                    )
                                    guidance_box.on_value_change(
                                        lambda e, jid=product_jid: guidance_buffers.__setitem__(
                                            jid,
                                            str(e.value or ""),
                                        )
                                    )
                                    with ui.row().classes("gap-2 mt-2 flex-wrap"):
                                        retry_des_btn = ui.button(
                                            "Retry DES",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Retry DES",
                                                runner=_retry_runtime_des,
                                            ),
                                            icon="restart_alt",
                                        ).props("color=blue")
                                        submit_guidance_btn = ui.button(
                                            "Submit operator guidance",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Submit operator guidance",
                                                runner=_submit_runtime_guidance,
                                            ),
                                            icon="person",
                                        ).props("color=amber")
                                        retry_des_btn.set_enabled(
                                            not bridge_reentry_locked and not controls_locked
                                        )
                                        submit_guidance_btn.set_enabled(not controls_locked)
                                        run_bridge_btn = ui.button(
                                            "Run bridge reasoning",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Run bridge reasoning",
                                                runner=_run_runtime_bridge,
                                            ),
                                            icon="smart_toy",
                                        ).props("color=indigo")
                                        run_bridge_btn.set_enabled(
                                            not bridge_reentry_locked and not controls_locked
                                        )

                _refresh_runtime_recovery_panel()
                _managed_timer(5.0, _refresh_runtime_recovery_panel)

            # ── Runtime Blocked Tasks ───────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Runtime Blocked Tasks").classes("text-lg font-semibold mb-2")
                runtime_blocked_container = ui.column().classes("w-full gap-2")

                def _refresh_runtime_blocked_tasks():
                    runtime_blocked_container.clear()
                    ss = bridge.get_safety_state()
                    blocked = ss.get("blocked_tasks", {})
                    if not blocked:
                        with runtime_blocked_container:
                            ui.label("No blocked tasks").classes("text-slate-400 italic")
                        return
                    with runtime_blocked_container:
                        for tid, info in blocked.items():
                            with ui.card().classes("w-full bg-red-50"):
                                ui.label(f"Task: {tid}").classes("font-semibold")
                                ui.label(
                                    f"Violated rule: {info.get('violated_rule', 'unknown')}"
                                ).classes("text-sm text-red-600")

                _managed_timer(3.0, _refresh_runtime_blocked_tasks)

            # ── Task States Table ───────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Task States").classes("text-lg font-semibold mb-2")
                task_table = ui.table(
                    columns=[
                        {
                            "name": "task_id",
                            "label": "Task ID",
                            "field": "task_id",
                            "sortable": True,
                        },
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
                    ],
                    rows=[],
                    row_key="task_id",
                    selection="multiple",
                ).classes("w-full")

                selected_task_ids: list[str] = []

                def _set_node_details(task_ids: list[str]) -> None:
                    """Populate the detail table with one or more selected nodes."""
                    if not task_ids:
                        detail_label.text = "Node Detail"
                        detail_table.rows = []
                        return
                    nodes = _preview_or_runtime_nodes()
                    all_rows: list[dict] = []
                    row_idx = 0
                    for tid in task_ids:
                        node = next(
                            (n for n in nodes if (n.get("id") or n.get("task_id")) == tid), None
                        )
                        if not node:
                            continue
                        if all_rows:
                            all_rows.append(
                                {"row_id": f"sep_{row_idx}", "key": "———", "value": "———"}
                            )
                            row_idx += 1
                        for k, v in node.items():
                            display = (
                                json.dumps(v, default=str)
                                if isinstance(v, (dict, list))
                                else str(v)
                            )
                            all_rows.append({"row_id": f"{tid}_{k}", "key": k, "value": display})
                            row_idx += 1
                    detail_label.text = "Node Detail — " + ", ".join(task_ids)
                    detail_table.rows = all_rows

                def _refresh_tasks():
                    nodes = _preview_or_runtime_nodes()
                    ts = _preview_or_runtime_task_states(nodes)
                    task_table.rows = [{"task_id": k, "status": v} for k, v in ts.items()]
                    if selected_task_ids:
                        _set_node_details(selected_task_ids)
                    elif ts:
                        active = next((k for k, v in ts.items() if "running" in v.lower()), None)
                        if not active:
                            active = next(
                                (
                                    k
                                    for k, v in reversed(ts.items())
                                    if any(
                                        s in v.lower()
                                        for s in ("dispatched", "accepted", "completed", "failed")
                                    )
                                ),
                                None,
                            )
                        if active:
                            _set_node_details([active])

                _managed_timer(2.0, _refresh_tasks)

            # ── Execution Timeline ──────────────────────────────────────
            with ui.card().classes("w-full"):
                with ui.expansion("Execution Timeline", icon="timeline", value=False).classes(
                    "w-full"
                ):
                    ui.label("Expand this panel to inspect the latest execution events.").classes(
                        "text-xs text-slate-500 mb-2"
                    )
                    timeline_table = ui.table(
                        columns=[
                            {
                                "name": "timestamp",
                                "label": "Time",
                                "field": "timestamp",
                                "sortable": True,
                            },
                            {
                                "name": "task_id",
                                "label": "Task",
                                "field": "task_id",
                                "sortable": True,
                            },
                            {
                                "name": "status",
                                "label": "Status",
                                "field": "status",
                                "sortable": True,
                            },
                            {"name": "resource_jid", "label": "Resource", "field": "resource_jid"},
                        ],
                        rows=[],
                    ).classes("w-full")

                def _refresh_timeline():
                    tl = bridge.get_execution_timeline()
                    timeline_table.rows = tl[-50:]

                _managed_timer(3.0, _refresh_timeline)

            # ── Node Detail ─────────────────────────────────────────────
            with ui.card().classes("w-full"):
                detail_label = ui.label("Node Detail").classes("text-lg font-semibold mb-2")
                detail_table = ui.table(
                    columns=[
                        {"name": "key", "label": "Field", "field": "key"},
                        {"name": "value", "label": "Value", "field": "value"},
                    ],
                    rows=[],
                    row_key="row_id",
                ).classes("w-full")

                def _on_task_select(e):
                    selected_task_ids.clear()
                    for row in e.selection or []:
                        tid = row.get("task_id")
                        if tid:
                            selected_task_ids.append(tid)
                    _set_node_details(selected_task_ids)

                task_table.on_select(_on_task_select)

        # ── Right column: chat panel ──────────────────────────
        with ui.column().classes("w-96 shrink-0 sticky top-20 self-start"):
            render_chat(
                bridge,
                agent_options={
                    "auto": "Auto-route",
                    "product": "Product Agent",
                    "cca": "Central Controller Agent",
                    "xarm6": "xArm6",
                    "ur5e": "UR5e",
                },
                title="System Chat",
            )


def _check_prerequisites(
    bridge: SystemBridge,
    mode: str,
    banner: ui.column,
    *,
    launch_simulation: Callable[[], Any] | None = None,
    launch_simulation_busy: bool = False,
) -> bool:
    """Check if prerequisites are met for the selected mode. Updates the banner. Returns True if OK."""

    def _replace_banner(signature: tuple[Any, ...], render_body: Callable[[], None] | None) -> None:
        if getattr(banner, "_cais_prereq_signature", None) == signature:
            return
        banner._cais_prereq_signature = signature
        banner.clear()
        if render_body is None:
            return
        with banner:
            render_body()

    def _suppress_sim_ready_note(message: str) -> bool:
        text = str(message or "").strip().lower()
        return text.startswith("perception is still warming up:")

    if bridge.system_running:
        _replace_banner(("system_running",), None)
        return True  # Already running, don't block.

    if bridge._starting:

        def _render_starting() -> None:
            with ui.row().classes("items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded"):
                ui.icon("hourglass_top").classes("text-lg")
                ui.label("System startup in progress...").classes("text-sm font-semibold")

        _replace_banner(("starting",), _render_starting)
        return False

    if mode == "dry_run":
        # Dry Run has no prerequisites.
        def _render_dry_run() -> None:
            with ui.row().classes("items-center gap-2 text-green-600"):
                ui.icon("check_circle").classes("text-sm")
                ui.label("Dry Run mode — no prerequisites required.").classes("text-sm")

        _replace_banner(("dry_run",), _render_dry_run)
        return True

    if mode == "simulation":
        # Simulation needs Gazebo stack plus MoveIt/services ready.
        statuses = bridge.ros2_all_statuses()
        gazebo_running = any(
            statuses.get(k) == "running" for k in ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")
        )
        sim_ready = False
        sim_reason = ""
        if gazebo_running and hasattr(bridge, "simulation_start_ready"):
            sim_ready, sim_reason = bridge.simulation_start_ready()
        elif gazebo_running:
            sim_ready = True

        def _render_simulation() -> None:
            if (
                gazebo_running
                and sim_ready
                and sim_reason
                and not _suppress_sim_ready_note(sim_reason)
            ):
                with ui.row().classes("items-center gap-2 text-amber-700 bg-amber-50 p-3 rounded"):
                    ui.icon("info").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Gazebo + MoveIt are ready.").classes("text-sm font-semibold")
                        ui.label(sim_reason).classes("text-xs")
            elif gazebo_running and sim_ready:
                with ui.row().classes("items-center gap-2 text-green-600"):
                    ui.icon("check_circle").classes("text-sm")
                    ui.label("Gazebo + MoveIt are ready — safe to start.").classes("text-sm")
            elif gazebo_running:
                with ui.row().classes("items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded"):
                    ui.icon("warning").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Startup is not done yet.").classes("text-sm font-semibold")
                        ui.label(
                            sim_reason or "Waiting for ROS services/prewarm to complete..."
                        ).classes("text-xs")
            else:
                with ui.row().classes("items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded"):
                    ui.icon("warning").classes("text-lg")
                    with ui.column().classes("gap-2"):
                        ui.label("Gazebo is not running.").classes("text-sm font-semibold")
                        ui.label("Launch Gazebo + MoveIt first.").classes("text-sm")
                        launch_btn = ui.button(
                            "Start Gazebo + MoveIt",
                            on_click=launch_simulation,
                            icon="play_arrow",
                        ).props("dense color=amber")
                        launch_btn.set_enabled(
                            launch_simulation is not None and not launch_simulation_busy
                        )

        _replace_banner(
            (
                "simulation",
                gazebo_running,
                sim_ready,
                sim_reason,
                bool(launch_simulation),
                launch_simulation_busy,
            ),
            _render_simulation,
        )
        return gazebo_running and sim_ready

    if mode == "physical":
        # Physical mode requires real perception backend availability.
        if hasattr(bridge, "hardware_connection_statuses_cached"):
            hw = bridge.hardware_connection_statuses_cached()
        elif hasattr(bridge, "hardware_connection_statuses"):
            hw = bridge.hardware_connection_statuses()
        else:
            hw = {}
        xarm = hw.get("xarm6", {})
        ur5e = hw.get("ur5e", {})
        ready, reason = bridge.physical_perception_ready()

        def _line(name: str, entry: dict) -> str:
            ip = entry.get("ip", "?")
            if entry.get("reachable"):
                latency = entry.get("latency_ms")
                if latency is None:
                    return f"{name}: {ip} reachable"
                return f"{name}: {ip} reachable ({latency:.1f} ms)"
            return f"{name}: {ip} unreachable ({entry.get('message', 'no reply')})"

        xarm_line = _line("xArm6", xarm)
        ur5e_line = _line("UR5e", ur5e)

        def _render_physical() -> None:
            if not ready:
                with ui.row().classes("items-center gap-2 text-red-700 bg-red-50 p-3 rounded"):
                    ui.icon("error").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Physical mode is blocked.").classes("text-sm font-semibold")
                        ui.label(reason).classes("text-xs")
                        ui.label(xarm_line).classes("text-xs")
                        ui.label(ur5e_line).classes("text-xs")
            else:
                with ui.row().classes("items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded"):
                    ui.icon("info").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label(
                            "Physical mode — ensure robots are powered on and controllers are running."
                        ).classes("text-sm")
                        ui.label(xarm_line).classes("text-xs")
                        ui.label(ur5e_line).classes("text-xs")

        _replace_banner(("physical", ready, reason, xarm_line, ur5e_line), _render_physical)
        return ready

    _replace_banner(("mode", mode), None)
    return True


def _stat_card(label: str, value: str, icon: str) -> None:
    with ui.column().classes("items-center"):
        ui.icon(icon).classes("text-2xl text-slate-500")
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-slate-500")
