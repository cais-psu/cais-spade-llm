"""Existing system control and recovery workflow for recovery-framework."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import context, ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.refresh import PageRefresh
from cais_spade_llm.ui import recovery_setup as settings
from cais_spade_llm.ui.components.agent_chat import render_chat
from cais_spade_llm.ui.components.dag_graph import nodes_to_mermaid
from cais_spade_llm.ui.components.robot_status_card import render_robot_status_card
from cais_spade_llm.recovery_framework.delivery import prepare_start, request_stop, reset_stop
from cais_spade_llm.recovery_framework.startup import prepare_delivery_start
from cais_spade_llm.recovery_framework.environment_runtime import prepare_environment_start

# Execution mode labels → internal values.
_MODE_MAP = {
    "Simulation": "simulation",
    "Physical": "physical",
}
_MODE_LABELS = list(_MODE_MAP.keys())
_MODE_LABEL_BY_VALUE = {value: label for label, value in _MODE_MAP.items()}
_RECOVERY_MODE_OPTIONS = {
    "auto": "Auto",
    "manual": "Manual",
    "pre_ran": "Pre-ran",
}
_RECOVERY_VALIDATION_POLICY_OPTIONS = {
    "validated": "Recovery Safety Check",
    "no_validation": "No Recovery Safety Check",
}
logger = logging.getLogger(__name__)


def render(bridge: SystemBridge, *, is_active: Callable[[], bool] | None = None) -> None:
    """Render the recovery workflow, polling only while its run tab is active."""
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
    select_popup_open = {"value": False}
    select_pause_until = {"value": 0.0}
    polling = PageRefresh(is_active)
    editing_recovery = {"value": False}

    def _track_recovery_input(element) -> None:
        element.on("focus", lambda: editing_recovery.update(value=True))
        element.on("blur", lambda: editing_recovery.update(value=False))

    display_values: dict[str, Any] = {}

    def _changed(name: str, value: Any) -> bool:
        if name in display_values and display_values[name] == value:
            return False
        display_values[name] = deepcopy(value)
        return True

    def _notify(message: Any, **kwargs: Any) -> None:
        if not getattr(client, "_deleted", False):
            client.safe_invoke(lambda: ui.notify(message, **kwargs))

    def _managed_timer(interval: float, callback, **kwargs):
        async def refresh():
            if select_popup_open["value"] or time.monotonic() < select_pause_until["value"]:
                return
            result = callback()
            if asyncio.iscoroutine(result):
                await result
        return polling.timer(interval, refresh, **kwargs)

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

    refresh_dag_now = lambda: None

    toolbar = ui.column().classes("w-full px-6 py-3 bg-white border-b shadow-sm").style(
        "position: sticky; top: var(--q-header-height, 64px); z-index: 100;"
    )
    toolbar.mark("recovery-controls")
    with ui.row().classes("w-full px-6 gap-6 items-start no-wrap"):
        # ── Left column: existing dashboard content ───────────
        with ui.column().classes("flex-1 gap-6 min-w-0"):
            # ── System Control Card ──────────────────────────────────────
            with ui.card().classes("w-full").style("overflow: visible;"):
                ui.label("System Control").classes("text-lg font-semibold mb-2")

                ui.label("Saved experiment setup").classes("font-semibold")
                ui.link("Edit setup", "/recovery-framework?tab=setup")
                setup_message = ui.label().classes("text-sm text-slate-600")
                setup_table = ui.table(
                    columns=[
                        {"name": "setting", "field": "setting", "label": "Setting"},
                        {"name": "value", "field": "value", "label": "Saved value"},
                    ], rows=[], row_key="setting",
                ).classes("w-full").props("wrap-cells")
                with ui.expansion("Saved setup details", icon="data_object").classes("w-full") as setup_details:
                    setup_json = ui.code(language="json").classes("w-full")
                selected_setup: dict[str, Any] = {}
                setup_error = {"message": "Checking saved setup..."}
                validation = settings.StartupValidation()
                controls_lock = asyncio.Lock()
                setup_is_delivery = {"value": False}

                def _apply_setup(result: tuple[dict, str, bool]) -> None:
                    loaded, reason, delivery = result
                    if loaded != selected_setup:
                        selected_setup.clear()
                        selected_setup.update(loaded)
                        setup_table.rows = settings.setup_summary(loaded) if loaded else []
                    setup_error["message"] = reason
                    setup_is_delivery["value"] = delivery
                    setup_message.text = (
                        reason
                        or "Saved settings. Configuration does not establish execution support or observed state."
                    )

                def _show_setup_json(e=None) -> None:
                    if setup_details.value:
                        setup_json.content = json.dumps(selected_setup, indent=2, ensure_ascii=False)

                setup_details.on_value_change(_show_setup_json)
                # Prerequisite banner.
                prereq_banner = ui.column().classes("w-full mt-3")
                bundle_gate_banner = ui.column().classes("w-full mt-2")

                with ui.column().classes("w-full gap-2 mt-4"):
                    banner_hide_task: asyncio.Task | None = None
                    start_click_state = {"locked": False}
                    start_cancelled = {"value": False}
                    gazebo_launch_state = {"busy": False}
                    start_task: asyncio.Task | None = None

                    def _selected_files() -> tuple[str, str]:
                        return (
                            selected_setup.get("selected_product_order_file", ""),
                            selected_setup.get("selected_safety_file", ""),
                        )

                    def _bundle_gate(setup: dict, reason: str) -> tuple[bool, str]:
                        if reason:
                            return False, reason
                        order_file = setup.get("selected_product_order_file", "")
                        safe_file = setup.get("selected_safety_file", "")
                        if not order_file:
                            return False, "Select a product order JSON file."
                        if (
                            not safe_file
                            or safe_file == "__NONE__"
                            or Path(safe_file).name == "safety_none.txt"
                        ):
                            return True, ""
                        safety_eval = bridge.evaluate_safety_intent_approval(safe_file)
                        if not bool(safety_eval.get("approved", False)):
                            return (
                                False,
                                f"Selected safety file is not verified: {safety_eval.get('reason', 'not_approved')}.",
                            )
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
                        try:
                            start_click_state["locked"] = True
                            start_cancelled["value"] = False
                            start_btn.set_enabled(False)
                            reset_btn.set_enabled(False)
                            _set_action_banner(
                                "info", "Checking saved setup before Start System..."
                            )
                            try:
                                async with controls_lock:
                                    checked = await asyncio.to_thread(validation.read, force=True)
                                    _apply_setup(checked)
                                    bundle_ok, bundle_msg = await asyncio.to_thread(
                                        _bundle_gate, checked[0], checked[1]
                                    )
                            except Exception as exc:
                                start_click_state["locked"] = False
                                _set_action_banner(
                                    "error",
                                    f"Startup source check failed: {exc}",
                                    auto_hide_s=8.0,
                                )
                                return
                            if start_cancelled["value"]:
                                return
                            if not bundle_ok:
                                _set_action_banner("warning", bundle_msg, auto_hide_s=8.0)
                                return
                            internal = selected_setup["execution_mode"]
                            bridge.execution_mode = internal
                            bridge.robot_env = "gazebo" if internal == "simulation" else "real"

                            try:
                                order_file, safe_file = _selected_files()
                                if not order_file:
                                    _set_action_banner(
                                        "warning",
                                        "Select a product order JSON file.",
                                        auto_hide_s=6.0,
                                    )
                                    return
                                selected_product = selected_setup["selected_product"]
                                if selected_product:
                                    bridge.selected_product = selected_product
                                else:
                                    try:
                                        bridge.selected_product = (
                                            bridge.resolve_product_init_for_product_order(
                                                order_file
                                            )
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
                                bridge.set_runtime_recovery_mode(
                                    selected_setup["runtime_recovery_mode"]
                                )
                                bridge.set_runtime_recovery_validation_policy(
                                    selected_setup["runtime_recovery_validation_policy"]
                                )
                                bridge.set_runtime_recovery_archive_selection(
                                    selected_setup["runtime_recovery_archive_path"],
                                    selected_setup["runtime_recovery_archive_label"],
                                )
                            except Exception as exc:
                                _set_action_banner(
                                    "error",
                                    f"Failed to configure startup source: {exc}",
                                    auto_hide_s=8.0,
                                )
                                return

                            status = await asyncio.to_thread(_read_prerequisites, bridge, internal)
                            if start_cancelled["value"]:
                                return
                            if not setup_is_delivery["value"] and not _check_prerequisites(
                                bridge,
                                internal,
                                prereq_banner,
                                status=status,
                            ):
                                detail = f" {bridge.last_error}" if bridge.last_error else ""
                                _set_action_banner(
                                    "warning",
                                    "Startup is not done yet." + detail,
                                    auto_hide_s=6.0,
                                )
                                return
                            start_click_state["locked"] = True
                            reset_stop()
                            await _update_controls()
                            _set_action_banner("info", "Start System clicked. Starting agents...")
                            if start_task is None or start_task.done():

                                async def _run_start_in_background() -> None:
                                    if hasattr(bridge, "_diag_emit"):
                                        bridge._diag_emit("dashboard start background task begin")
                                    try:
                                        if setup_is_delivery["value"]:
                                            prepare_environment_start(None)
                                            _set_action_banner(
                                                "info",
                                                "Preparing Gazebo and checking Storage, KMR, and M1...",
                                            )
                                            await prepare_delivery_start(bridge, selected_setup)
                                        else:
                                            prepare_start(None)
                                            await asyncio.to_thread(
                                                prepare_environment_start, deepcopy(selected_setup)
                                            )
                                        if start_cancelled["value"]:
                                            return
                                        await bridge.start_system()
                                        if bridge.system_running:
                                            notice = bridge.consume_notice()
                                            if notice:
                                                _set_action_banner(
                                                    "warning", notice, auto_hide_s=10.0
                                                )
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
                                    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                                        bridge.last_error = str(exc)
                                        _set_action_banner(
                                            "error", f"Start failed: {exc}", auto_hide_s=10.0
                                        )
                                    finally:
                                        prepare_start(None)
                                        prepare_environment_start(None)
                                        if hasattr(bridge, "_diag_emit"):
                                            bridge._diag_emit(
                                                "dashboard start background task end "
                                                f"running={bridge.system_running} error={bridge.last_error or ''}"
                                            )
                                        start_click_state["locked"] = False
                                        await _update_controls()

                                start_task = asyncio.create_task(_run_start_in_background())
                        finally:
                            if (
                                start_task is None or start_task.done()
                            ) and not bridge.system_running:
                                start_click_state["locked"] = False
                                start_btn.set_enabled(False)

                    async def _stop():
                        if bridge._stopping:
                            _set_action_banner("warning", "Stop already in progress...")
                            return
                        start_cancelled["value"] = True
                        request_stop()
                        if (
                            not bridge.system_running
                            and start_click_state["locked"]
                            and (start_task is None or start_task.done())
                        ):
                            _set_action_banner("info", "Stopping startup checks...")
                            return
                        if start_task is not None and not start_task.done():
                            _set_action_banner("info", "Stopping preparation and pending delivery tasks...")
                            await start_task
                        if not bridge.system_running:
                            _set_action_banner(
                                "warning", "System is already stopped.", auto_hide_s=4.0
                            )
                            start_click_state["locked"] = False
                            await _update_controls()
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
                            await _update_controls()

                    async def _launch_gazebo_dual():
                        if gazebo_launch_state["busy"]:
                            _set_action_banner(
                                "warning",
                                "Gazebo + MoveIt launch already in progress...",
                                auto_hide_s=4.0,
                            )
                            return
                        gazebo_launch_state["busy"] = True
                        simulation_btn.set_enabled(False)
                        _set_action_banner(
                            "info", "Launching no-hardware dual Gazebo + MoveIt/RViz..."
                        )
                        try:
                            if await asyncio.to_thread(bridge.simulation_environment_running):
                                _set_action_banner(
                                    "warning", "Gazebo stack is already running.", auto_hide_s=4.0
                                )
                                return
                            err = await asyncio.to_thread(bridge.ros2_start, "gazebo_dual")
                            if err:
                                _set_action_banner("warning", err, auto_hide_s=8.0)
                            else:
                                _set_action_banner(
                                    "success",
                                    "No-hardware dual Gazebo + MoveIt/RViz launched. Waiting for ROS services...",
                                    auto_hide_s=6.0,
                                )
                        finally:
                            gazebo_launch_state["busy"] = False
                            await _update_controls()

                    reset_scope_options = [
                        "Reset All",
                        "Reset Plan",
                        "Reset Gazebo",
                    ]

                    async def _reset_selected():
                        if bridge._starting or start_click_state["locked"]:
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
                            await _update_controls()

                    with toolbar, ui.row().classes("items-end gap-4 flex-wrap"):
                        simulation_btn = ui.button(
                            "Start Simulation", on_click=_launch_gazebo_dual, icon="rocket_launch"
                        ).props("color=blue")
                        start_btn = ui.button(
                            "Start System", on_click=_start, icon="play_arrow"
                        ).props("color=green")
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
                    with toolbar:
                        control_status = ui.label("Checking saved setup...").classes("text-sm")
                        action_banner = ui.row().classes(
                            "w-full mt-2 items-center gap-2 rounded p-3 text-sm text-blue-700 bg-blue-50"
                        )
                    action_banner.style("display:none;")
                    with action_banner:
                        action_banner_icon = ui.icon("info")
                        action_banner_label = ui.label("")
                    with ui.row().classes(
                        "w-full mt-2 items-center gap-3 rounded border border-slate-200 p-3"
                    ):
                        ui.icon("photo_camera").classes("text-slate-600")
                        dashboard_perception_status = ui.label(
                            "Perception status: checking..."
                        ).classes("text-xs text-slate-600")
                        ui.space()
                        ui.link("Open Camera & Perception", target="/perception").classes(
                            "text-sm font-medium"
                        )

                    async def _refresh_dashboard_perception_status() -> None:
                        status = await asyncio.to_thread(bridge.physical_perception_status)
                        if not polling.active():
                            return
                        frame_age = status.get("frame_age_sec")
                        age_text = "n/a" if frame_age is None else f"{float(frame_age):.1f} s"
                        dashboard_perception_status.text = (
                            "UR5e camera: "
                            f"{'connected' if status.get('realsense_connected') else 'not connected'} | "
                            f"frame age={age_text} | Roboflow="
                            f"{'ready' if status.get('roboflow_ready') else 'not ready'} | "
                            f"mirror={(status.get('twin') or {}).get('state', 'not running')}"
                        )

                    _managed_timer(2.0, _refresh_dashboard_perception_status)
                    plan_safety_banner = ui.row().classes(
                        "w-full mt-2 items-center gap-2 rounded p-3 text-sm text-red-700 bg-red-50"
                    )
                    plan_safety_banner.style("display:none;")
                    with plan_safety_banner:
                        plan_safety_banner_icon = ui.icon("warning")
                        plan_safety_banner_label = ui.label("")

                # Error display.
                error_label = ui.label("").classes("text-red-500 text-sm mt-2")
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

                async def _update_controls(*, force: bool = False):
                    if controls_lock.locked() and not force:
                        return
                    try:
                        async with controls_lock:
                            checked = (
                                (
                                    deepcopy(selected_setup),
                                    setup_error["message"],
                                    setup_is_delivery["value"],
                                )
                                if start_click_state["locked"]
                                else await asyncio.to_thread(validation.read, force=force)
                            )
                            bundle_ok, bundle_msg = await asyncio.to_thread(
                                _bundle_gate, checked[0], checked[1]
                            )
                            mode = checked[0].get("execution_mode", "")
                            status = await asyncio.to_thread(_read_prerequisites, bridge, mode)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        start_btn.set_enabled(False)
                        control_status.text = f"Startup check failed: {exc}"
                        return
                    if getattr(client, "_deleted", False):
                        return
                    _apply_setup(checked)
                    prereqs_met = _check_prerequisites(bridge, mode, prereq_banner, status=status)
                    _set_bundle_gate_banner(bundle_msg if not bridge.system_running else "")
                    can_start = (
                        (prereqs_met or setup_is_delivery["value"])
                        and bundle_ok
                        and not bridge.system_running
                        and not bridge._starting
                        and not bridge._stopping
                        and not start_click_state["locked"]
                        and not gazebo_launch_state["busy"]
                    )
                    gazebo_running = status.get("gazebo_running", False)
                    can_reset = (
                        str(reset_scope_select.value) != "Reset Gazebo"
                        or gazebo_running
                        or bridge.system_running
                    )
                    start_btn.set_enabled(can_start)
                    simulation_btn.set_enabled(
                        mode == "simulation"
                        and not gazebo_running
                        and not gazebo_launch_state["busy"]
                    )
                    stop_btn.set_enabled(
                        (bridge.system_running or start_click_state["locked"])
                        and not bridge._stopping
                    )
                    reset_btn.set_enabled(
                        can_reset
                        and not bridge._starting
                        and not bridge._stopping
                        and not start_click_state["locked"]
                    )
                    if bridge.system_running:
                        reason = "System is running."
                    elif bridge._starting or start_click_state["locked"]:
                        reason = "Start System is in progress..."
                    elif gazebo_launch_state["busy"]:
                        reason = "Start Simulation is in progress..."
                    else:
                        reason = bundle_msg or status.get("reason", "") or "Ready to start."
                    control_status.text = reason
                    alerts = bridge.get_plan_safety_alerts()
                    if _changed("alerts", alerts):
                        _set_plan_safety_banner(alerts)
                    error_label.text = bridge.last_error or ""

                start_btn.set_enabled(False)
                stop_btn.set_enabled(False)
                _managed_timer(1.0, _update_controls)

                async def _refresh_controls_and_dag() -> None:
                    await _update_controls(force=True)
                    refresh_dag_now()

                reset_scope_select.on_value_change(lambda: _update_controls())
                ui.button("Refresh saved setup", icon="refresh", on_click=_refresh_controls_and_dag).props("flat")

            # ── Agent Overview Grid ──────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Agent Status").classes("text-lg font-semibold mb-2")
                agent_container = ui.row().classes("gap-4 flex-wrap")

            def _refresh_agents():
                statuses = bridge.get_agent_statuses() if bridge.system_running else []
                if not _changed("agents", (bridge.system_running, statuses)):
                    return
                agent_container.clear()
                if not bridge.system_running:
                    with agent_container:
                        ui.label("System not running").classes("text-slate-400 italic")
                    return

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
                task_states = bridge.get_task_states()
                robot_states = bridge.get_robot_states()
                safety = bridge.get_safety_state()
                if not _changed("stats", (task_states, robot_states, safety.get("blocked_tasks", {}))):
                    return
                stats_row.clear()
                with stats_row:
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

                    active = sum(
                        1 for r in robot_states.values() if r.get("current_state", "idle") != "idle"
                    )
                    _stat_card("Active Robots", str(active), "precision_manufacturing")

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
                        if mermaid.content != "graph LR\n    empty[No current operations]":
                            mermaid.set_content("graph LR\n    empty[No current operations]")
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
                    if _changed("dag", (nodes, task_states)):
                        content = nodes_to_mermaid(nodes, task_states)
                        if mermaid.content != content:
                            mermaid.set_content(content)

                refresh_dag_now = _refresh_dag
                _refresh_dag()
                _managed_timer(0.5, _refresh_dag)

            # ── Live Robot Status ───────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Live Robot Status").classes("text-lg font-semibold mb-2")
                robot_status_container = ui.column().classes("w-full gap-4")

                def _refresh_robot_status():
                    states = bridge.get_robot_states()
                    if not _changed("robot_status", states):
                        return
                    robot_status_container.clear()
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

                async def _refresh_runtime_safety_rules():
                    selected_safety_file = selected_setup.get("selected_safety_file", "")
                    preview = {}
                    if selected_safety_file and selected_safety_file.upper() != "__NONE__":
                        try:
                            preview = await asyncio.to_thread(bridge.get_safety_rule_preview, selected_safety_file)
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
                    if runtime_rules_table.rows != rows:
                        runtime_rules_table.rows = rows

                _managed_timer(5.0, _refresh_runtime_safety_rules)

            # ── Runtime Safety State ────────────────────────────────────
            with ui.card().classes("w-full"):
                with ui.expansion(
                    "Runtime Safety State",
                    icon="shield",
                    value=False,
                ).classes("w-full") as safety_expansion:
                    runtime_safety_state = ui.code("{}", language="json").classes("w-full")

                async def _refresh_runtime_safety_state():
                    if not safety_expansion.value:
                        return
                    ss = await asyncio.to_thread(bridge.get_safety_state)
                    runtime_safety_state.content = (
                        json.dumps(ss, indent=2, default=str) if ss else "{}"
                    )

                safety_expansion.on_value_change(_refresh_runtime_safety_state)
                _managed_timer(5.0, _refresh_runtime_safety_state)

            # ── Replan / Recovery ───────────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Replan / Recovery").classes("text-lg font-semibold mb-2")
                ui.label(
                    "Runtime DES recovery stays inside one workflow: DES search, narrow LLM recovery, plan validation, then human intervention if needed."
                ).classes("text-xs text-slate-500 mb-2")
                runtime_recovery_container = ui.column().classes("w-full gap-3")

                guidance_buffers: dict[str, str] = {}
                recovery_feedback_buffers: dict[str, str] = {}
                preprogrammed_recovery_buffers: dict[str, str] = {}
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

                def _recovery_task_rows_for_display(
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
                                    "recovery_sequence_id": str(
                                        node.get("recovery_sequence_id", "")
                                    ).strip(),
                                    "recovery_sequence_index": int(
                                        node.get("recovery_sequence_index") or 0
                                    ),
                                    "recovery_sequence_length": int(
                                        node.get("recovery_sequence_length") or 0
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
                    if key == "recovery_ready":
                        return "orange"
                    if key == "human_required":
                        return "red"
                    if key in {"des_search", "llm_recovery", "validating"}:
                        return "blue"
                    return "grey"

                def _status_label(status: str) -> str:
                    labels = {
                        "idle": "Idle",
                        "des_search": "DES search",
                        "recovery_ready": "Recovery ready",
                        "llm_recovery": "Recovery proposal",
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
                        "des_with_llm_recovery": "DES + LLM recovery",
                        "human_required": "Human required",
                    }
                    key = str(value or "").strip().lower()
                    return labels.get(key, key.replace("_", " ").title() or "Unknown")

                def _resolution_color(value: str) -> str:
                    key = str(value or "").strip().lower()
                    if key in {"des_only", "des_with_llm_recovery"}:
                        return "green"
                    if key == "human_required":
                        return "red"
                    return "grey"

                def _stage_badges(recovery: dict[str, Any]) -> list[tuple[str, str]]:
                    status = str(recovery.get("status", "idle") or "idle").strip().lower()
                    used_recovery = bool(recovery.get("used_llm_recovery", False))
                    badges: list[tuple[str, str]] = []
                    stages = [
                        ("des_search", "DES search"),
                        ("recovery_ready", "Recovery review"),
                        ("llm_recovery", "Recovery proposal"),
                        ("validating", "Plan validation"),
                        ("human_required", "Human intervention"),
                    ]
                    for key, label in stages:
                        color = "grey"
                        if status == "resolved":
                            if (
                                key == "des_search"
                                or key == "recovery_ready"
                                and str(recovery.get("recovery_approval_state", "none") or "none")
                                .strip()
                                .lower()
                                in {"ready", "pending", "approved"}
                                or key == "llm_recovery"
                                and used_recovery
                                or key == "validating"
                            ):
                                color = "green"
                        elif status == key:
                            if key == "human_required":
                                color = "red"
                            elif key == "recovery_ready":
                                color = "orange"
                            else:
                                color = "blue"
                        elif (
                            key == "des_search"
                            and status
                            in {"recovery_ready", "llm_recovery", "validating", "human_required"}
                            or key == "recovery_ready"
                            and status in {"llm_recovery", "validating", "human_required"}
                            or key == "llm_recovery"
                            and used_recovery
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

                async def _run_runtime_recovery(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.generate_runtime_recovery_proposal,
                            product_jid,
                        )
                        approval_state = (
                            str(
                                result.get("recovery_approval_state", "none")
                                if isinstance(result, dict)
                                else "none"
                            )
                            .strip()
                            .lower()
                        )
                        if approval_state == "outline_pending":
                            feedback_text = "Recovery outline is ready for approval."
                            notify_text = "Recovery outline ready."
                        elif approval_state == "primitive_pending":
                            feedback_text = "Recovery primitives are ready for approval."
                            notify_text = "Recovery primitives ready."
                        elif approval_state == "pending":
                            feedback_text = "Recovery proposal is ready for final approval."
                        elif isinstance(result, dict) and isinstance(
                            result.get("action_feedback"), dict
                        ):
                            feedback_text = ""
                        else:
                            feedback_text = "Live recovery reasoning ran from the prepared session."
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
                            f"Failed to start recovery reasoning: {exc}",
                        )
                        _notify(f"Failed to start recovery reasoning: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _load_runtime_recovery_archive(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.load_runtime_recovery_archive_proposal,
                            product_jid,
                        )
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text="Archived recovery proposal loaded. Final approval is still required before execution starts.",
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to load archived recovery proposal: {exc}",
                        )
                        _notify(f"Failed to load archived recovery proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _approve_runtime_recovery_outline(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.approve_runtime_recovery_outline,
                            product_jid,
                        )
                        approval_state = (
                            str(
                                result.get("recovery_approval_state", "none")
                                if isinstance(result, dict)
                                else "none"
                            )
                            .strip()
                            .lower()
                        )
                        if approval_state == "primitive_pending":
                            feedback_text = (
                                "Outline approved. Recovery primitives are ready for review."
                            )
                            notify_text = "Recovery primitives ready."
                        elif approval_state == "pending":
                            feedback_text = (
                                "Outline approved. Recovery proposal is ready for final approval."
                            )
                        elif isinstance(result, dict) and isinstance(
                            result.get("action_feedback"), dict
                        ):
                            feedback_text = ""
                        else:
                            feedback_text = "Outline approved. Live recovery reasoning continued."
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

                async def _refine_runtime_recovery_outline(product_jid: str) -> None:
                    feedback = str(recovery_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Outline refinement guidance is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Outline refinement guidance is empty.", type="warning")
                        return
                    try:
                        result = await asyncio.to_thread(
                            bridge.refine_runtime_recovery_outline,
                            product_jid,
                            feedback,
                        )
                        recovery_feedback_buffers[product_jid] = ""
                        approval_state = (
                            str(
                                result.get("recovery_approval_state", "none")
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

                async def _reject_runtime_recovery_outline(product_jid: str) -> None:
                    feedback = str(recovery_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Outline rejection feedback is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Outline rejection feedback is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.reject_runtime_recovery_outline,
                            product_jid,
                            feedback,
                        )
                        recovery_feedback_buffers[product_jid] = ""
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

                async def _approve_runtime_recovery_primitives(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.approve_runtime_recovery_primitives,
                            product_jid,
                        )
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text="Recovery primitives approved. Final proposal review is ready.",
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

                async def _refine_runtime_recovery_primitives(product_jid: str) -> None:
                    feedback = str(recovery_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Primitive refinement guidance is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Primitive refinement guidance is empty.", type="warning")
                        return
                    try:
                        result = await asyncio.to_thread(
                            bridge.refine_runtime_recovery_primitives,
                            product_jid,
                            feedback,
                        )
                        recovery_feedback_buffers[product_jid] = ""
                        approval_state = (
                            str(
                                result.get("recovery_approval_state", "none")
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
                            notify_text = "Recovery proposal ready."
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

                async def _reject_runtime_recovery_primitives(product_jid: str) -> None:
                    feedback = str(recovery_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Primitive rejection feedback is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Primitive rejection feedback is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.reject_runtime_recovery_primitives,
                            product_jid,
                            feedback,
                        )
                        recovery_feedback_buffers[product_jid] = ""
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

                async def _approve_runtime_recovery(product_jid: str) -> None:
                    try:
                        result = await asyncio.to_thread(
                            bridge.approve_runtime_recovery_proposal, product_jid
                        )
                        _apply_runtime_action_result(
                            product_jid,
                            result,
                            default_kind="positive",
                            default_text="Recovery proposal approved. Runtime plan validation started.",
                        )
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to approve recovery proposal: {exc}",
                        )
                        _notify(f"Failed to approve recovery proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _reject_runtime_recovery(product_jid: str) -> None:
                    feedback = str(recovery_feedback_buffers.get(product_jid, "") or "").strip()
                    if not feedback:
                        _set_action_feedback(
                            product_jid, "warning", "Recovery rejection feedback is empty."
                        )
                        _clear_pending_runtime_action(product_jid)
                        _notify("Recovery rejection feedback is empty.", type="warning")
                        return
                    try:
                        await asyncio.to_thread(
                            bridge.reject_runtime_recovery_proposal,
                            product_jid,
                            feedback,
                        )
                        recovery_feedback_buffers[product_jid] = ""
                        _set_action_feedback(
                            product_jid,
                            "positive",
                            "Recovery proposal rejected. Session paused for manual refinement.",
                        )
                        _notify("Recovery proposal rejected.", type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to reject recovery proposal: {exc}",
                        )
                        _notify(f"Failed to reject recovery proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                async def _refine_runtime_recovery(product_jid: str) -> None:
                    feedback = str(recovery_feedback_buffers.get(product_jid, "") or "").strip()
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
                            bridge.generate_runtime_recovery_proposal, product_jid
                        )
                        recovery_feedback_buffers[product_jid] = ""
                        _set_action_feedback(
                            product_jid,
                            "positive",
                            "Recovery refinement submitted. Re-running bounded recovery reasoning.",
                        )
                        _notify("Recovery refinement submitted.", type="positive")
                        client.safe_invoke(_refresh_runtime_recovery_panel)
                    except Exception as exc:
                        _set_action_feedback(
                            product_jid,
                            "negative",
                            f"Failed to refine recovery proposal: {exc}",
                        )
                        _notify(f"Failed to refine recovery proposal: {exc}", type="negative")
                    finally:
                        _clear_pending_runtime_action(product_jid)

                def _recovery_mode_label(value: str) -> str:
                    key = str(value or "").strip().lower()
                    return _RECOVERY_MODE_OPTIONS.get(key, key.replace("_", " ").title() or "Auto")

                def _recovery_validation_policy_label(value: str) -> str:
                    key = str(value or "").strip().lower()
                    return _RECOVERY_VALIDATION_POLICY_OPTIONS.get(
                        key,
                        key.replace("_", " ").title() or "Validated",
                    )

                def _recovery_outline_rows(
                    recovery_debug: dict[str, Any] | None,
                ) -> list[dict[str, Any]]:
                    debug = recovery_debug if isinstance(recovery_debug, dict) else {}
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

                def _recovery_primitive_rows(
                    recovery_debug: dict[str, Any] | None,
                ) -> list[dict[str, Any]]:
                    debug = recovery_debug if isinstance(recovery_debug, dict) else {}
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

                def _recovery_sequence_for_display(recovery: dict[str, Any]) -> dict[str, Any] | None:
                    active = recovery.get("active_recovery_sequence")
                    if isinstance(active, dict):
                        return active
                    last_completed = recovery.get("last_completed_recovery_sequence")
                    if isinstance(last_completed, dict):
                        return last_completed
                    return None

                def _recovery_sequence_progress(sequence: dict[str, Any] | None) -> dict[str, Any]:
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
                    recovery_task_ids = [
                        str(task_id or "").strip()
                        for task_id in (sequence.get("recovery_task_ids") or [])
                        if str(task_id or "").strip()
                    ]
                    dispatched_task_ids = [
                        str(task_id or "").strip()
                        for task_id in (sequence.get("dispatched_recovery_task_ids") or [])
                        if str(task_id or "").strip()
                    ]
                    completed_task_ids = [
                        str(task_id or "").strip()
                        for task_id in (sequence.get("completed_recovery_task_ids") or [])
                        if str(task_id or "").strip()
                    ]
                    state = str(sequence.get("state") or "").strip().lower()
                    return {
                        "compiled": len(recovery_task_ids),
                        "dispatched": len(dispatched_task_ids),
                        "completed": len(completed_task_ids),
                        "state": state,
                        "mode": str(sequence.get("source_mode") or "").strip().lower(),
                        "archive_path": str(sequence.get("source_archive_path") or "").strip(),
                        "validation_policy": str(sequence.get("validation_policy") or "")
                        .strip()
                        .lower(),
                    }

                async def _refresh_runtime_recovery_panel() -> None:
                    if editing_recovery["value"]:
                        return
                    recoveries = await asyncio.to_thread(bridge.get_runtime_recoveries)
                    if not polling.active() or editing_recovery["value"]:
                        return
                    nodes = _preview_or_runtime_nodes() if recoveries else []
                    task_states = _preview_or_runtime_task_states(nodes) if recoveries else {}
                    if not _changed("recovery", (recoveries, nodes, task_states, action_feedback_buffers, pending_action_buffers)):
                        return
                    expansion_values: dict[str, list[bool]] = {}
                    for element in runtime_recovery_container.descendants():
                        if isinstance(element, ui.expansion):
                            expansion_values.setdefault(element.text, []).append(element.value)
                    runtime_recovery_container.clear()
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
                            recovery_feedback = str(
                                recovery_feedback_buffers.get(product_jid, "") or ""
                            )
                            selected_preprogrammed = str(
                                preprogrammed_recovery_buffers.get(
                                    product_jid,
                                    "recover_lg_v1",
                                )
                                or "recover_lg_v1"
                            )
                            guidance_buffers[product_jid] = operator_guidance
                            recovery_feedback_buffers[product_jid] = recovery_feedback
                            preprogrammed_recovery_buffers[product_jid] = selected_preprogrammed
                            action_feedback = (
                                action_feedback_buffers.get(product_jid)
                                if isinstance(action_feedback_buffers.get(product_jid), dict)
                                else None
                            )
                            pending_action_label = str(
                                pending_action_buffers.get(product_jid, "") or ""
                            ).strip()
                            recovery_proposal = recovery.get("recovery_proposal")
                            recovery_debug = (
                                recovery.get("recovery_debug")
                                if isinstance(recovery.get("recovery_debug"), dict)
                                else None
                            )
                            recovery_approval_state = str(
                                recovery.get("recovery_approval_state", "none") or "none"
                            ).strip()
                            recovery_mode = (
                                str(recovery.get("recovery_mode", "pre_ran") or "pre_ran")
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
                            recovery_stage = (
                                str(recovery.get("recovery_stage", "none") or "none").strip().lower()
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
                                or (recovery_debug or {}).get("artifact_directory")
                                or (recovery_debug or {}).get("per_turn_debug_dir")
                                or ""
                            ).strip()
                            outline_rows = _recovery_outline_rows(recovery_debug)
                            primitive_rows = _recovery_primitive_rows(recovery_debug)
                            active_recovery_sequence = (
                                recovery.get("active_recovery_sequence")
                                if isinstance(recovery.get("active_recovery_sequence"), dict)
                                else None
                            )
                            last_completed_recovery_sequence = (
                                recovery.get("last_completed_recovery_sequence")
                                if isinstance(recovery.get("last_completed_recovery_sequence"), dict)
                                else None
                            )
                            recovery_sequence_display = (
                                active_recovery_sequence
                                if isinstance(active_recovery_sequence, dict)
                                else last_completed_recovery_sequence
                            )
                            recovery_sequence_progress = _recovery_sequence_progress(
                                recovery_sequence_display
                            )
                            active_recovery_state = (
                                str((active_recovery_sequence or {}).get("state") or "")
                                .strip()
                                .lower()
                            )
                            recovery_reentry_locked = active_recovery_state in {"approved", "executing"}
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
                                        ui.badge(f"Mode {_recovery_mode_label(recovery_mode)}").props(
                                            "color=blue-grey"
                                        )
                                        ui.badge(
                                            f"Recovery Safety {_recovery_validation_policy_label(validation_policy)}"
                                        ).props("color=deep-orange")
                                        if recovery_stage not in {"", "none"}:
                                            ui.badge(
                                                f"Stage {recovery_stage.replace('_', ' ')}"
                                            ).props("color=purple")
                                        if recovery_approval_state not in {"", "none"}:
                                            ui.badge(
                                                f"Recovery {recovery_approval_state.replace('_', ' ')}"
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
                                        f"Recovery Safety: {_recovery_validation_policy_label(validation_policy)}"
                                    )
                                    if selected_archive_label:
                                        ui.label(f"Archived run: {selected_archive_label}")
                                    elif recovery_mode == "pre_ran" and selected_archive_path:
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
                                if recovery_sequence_display:
                                    sequence_label = (
                                        "Recovery sequence"
                                        if isinstance(active_recovery_sequence, dict)
                                        else "Last recovery sequence"
                                    )
                                    source_mode_label = _recovery_mode_label(
                                        recovery_sequence_progress.get("mode") or recovery_mode
                                    )
                                    status_line = (
                                        f"{sequence_label}: "
                                        f"{str(recovery_sequence_progress.get('state') or 'unknown').replace('_', ' ')} | "
                                        f"Compiled: {int(recovery_sequence_progress.get('compiled') or 0)} | "
                                        f"Dispatched: {int(recovery_sequence_progress.get('dispatched') or 0)} | "
                                        f"Completed: {int(recovery_sequence_progress.get('completed') or 0)} | "
                                        f"Mode: {source_mode_label}"
                                    )
                                    archive_path = str(
                                        recovery_sequence_progress.get("archive_path") or ""
                                    ).strip()
                                    sequence_validation_policy = str(
                                        recovery_sequence_progress.get("validation_policy")
                                        or validation_policy
                                    ).strip()
                                    if archive_path:
                                        status_line += f" | Archive source: {archive_path}"
                                    if sequence_validation_policy:
                                        status_line += (
                                            " | Recovery Safety: "
                                            f"{_recovery_validation_policy_label(sequence_validation_policy)}"
                                        )
                                    ui.label(status_line).classes("text-xs text-slate-600 mt-1")
                                if recovery_reentry_locked:
                                    ui.label(
                                        "Recovery execution is already active. Recovery approval and DES retry controls are locked until it finishes."
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

                                if status == "recovery_ready":
                                    if recovery_mode == "pre_ran":
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
                                                "Choose an Archived Recovery Run from the System Control panel."
                                            ).classes("text-xs text-slate-600")
                                        load_archive_btn = ui.button(
                                            "Load archived recovery",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Load archived recovery",
                                                runner=_load_runtime_recovery_archive,
                                            ),
                                            icon="history",
                                        ).props("color=indigo")
                                        load_archive_btn.set_enabled(
                                            bool(selected_archive_path)
                                            and not recovery_reentry_locked
                                            and not controls_locked
                                        )
                                    else:
                                        run_label = "Run live recovery"
                                        helper_text = (
                                            "Recovery session is prepared. Run live recovery reasoning to reach the next review stage."
                                            if recovery_mode == "manual"
                                            else "Auto mode normally starts live recovery reasoning immediately. Use this to run it again from the prepared session."
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
                                                    runner=_run_runtime_recovery,
                                                ),
                                                icon="smart_toy",
                                            ).props("color=indigo")
                                            run_live_btn.set_enabled(
                                                not recovery_reentry_locked and not controls_locked
                                            )

                                if (
                                    status == "llm_recovery"
                                    and recovery_approval_state == "outline_pending"
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
                                                "No outline events were captured in the current recovery trace."
                                            ).classes("text-xs text-slate-600")
                                    ui.label(
                                        "Approve the outline to continue live primitive generation, refine it with operator guidance, or reject it."
                                    ).classes("text-xs text-indigo-700 mt-2")
                                    review_box = (
                                        ui.textarea(
                                            label="Outline refinement or rejection feedback",
                                            value=recovery_feedback,
                                        )
                                        .props("outlined autogrow")
                                        .classes("w-full mt-2")
                                    )
                                    _track_recovery_input(review_box)
                                    review_box.on_value_change(
                                        lambda e,
                                        jid=product_jid: recovery_feedback_buffers.__setitem__(
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
                                                runner=_approve_runtime_recovery_outline,
                                            ),
                                            icon="check_circle",
                                        ).props("color=green")
                                        refine_outline_btn = ui.button(
                                            "Refine outline",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Refine outline",
                                                runner=_refine_runtime_recovery_outline,
                                            ),
                                            icon="tune",
                                        ).props("color=amber")
                                        reject_outline_btn = ui.button(
                                            "Reject outline",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Reject outline",
                                                runner=_reject_runtime_recovery_outline,
                                            ),
                                            icon="cancel",
                                        ).props("color=red")
                                        for button in (
                                            approve_outline_btn,
                                            refine_outline_btn,
                                            reject_outline_btn,
                                        ):
                                            button.set_enabled(
                                                not recovery_reentry_locked and not controls_locked
                                            )

                                if (
                                    status == "llm_recovery"
                                    and recovery_approval_state == "primitive_pending"
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
                                                "No primitive program rows were captured in the current recovery trace."
                                            ).classes("text-xs text-slate-600")
                                    ui.label(
                                        "Approve the primitive program to move to final proposal approval, refine it with operator guidance, or reject it."
                                    ).classes("text-xs text-indigo-700 mt-2")
                                    review_box = (
                                        ui.textarea(
                                            label="Primitive refinement or rejection feedback",
                                            value=recovery_feedback,
                                        )
                                        .props("outlined autogrow")
                                        .classes("w-full mt-2")
                                    )
                                    _track_recovery_input(review_box)
                                    review_box.on_value_change(
                                        lambda e,
                                        jid=product_jid: recovery_feedback_buffers.__setitem__(
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
                                                runner=_approve_runtime_recovery_primitives,
                                            ),
                                            icon="check_circle",
                                        ).props("color=green")
                                        refine_primitives_btn = ui.button(
                                            "Refine primitives",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Refine primitives",
                                                runner=_refine_runtime_recovery_primitives,
                                            ),
                                            icon="tune",
                                        ).props("color=amber")
                                        reject_primitives_btn = ui.button(
                                            "Reject primitives",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Reject primitives",
                                                runner=_reject_runtime_recovery_primitives,
                                            ),
                                            icon="cancel",
                                        ).props("color=red")
                                        for button in (
                                            approve_primitives_btn,
                                            refine_primitives_btn,
                                            reject_primitives_btn,
                                        ):
                                            button.set_enabled(
                                                not recovery_reentry_locked and not controls_locked
                                            )

                                if status == "llm_recovery" and isinstance(recovery_proposal, dict):
                                    with ui.expansion(
                                        "Recovery proposal", icon="alt_route", value=True
                                    ).classes("w-full mt-2"):
                                        primary_obligation = recovery_proposal.get(
                                            "primary_obligation"
                                        )
                                        if isinstance(primary_obligation, dict):
                                            ui.label(
                                                "Primary obligation: "
                                                f"{str(primary_obligation.get('rule_id', '') or 'n/a')} on "
                                                f"{str(primary_obligation.get('resource_jid', '') or 'n/a')}"
                                            ).classes("text-xs text-slate-600")
                                        description = str(
                                            recovery_proposal.get("description", "")
                                        ).strip()
                                        rationale = str(
                                            recovery_proposal.get("rationale", "")
                                        ).strip()
                                        if description:
                                            ui.label(description).classes("text-sm")
                                        if rationale:
                                            ui.label(f"Rationale: {rationale}").classes(
                                                "text-xs text-slate-600"
                                            )
                                        macro_tasks = recovery_proposal.get("macro_tasks") or []
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
                                                f"Macro: {str(recovery_proposal.get('function_name', '') or 'unnamed')}"
                                            ).classes("text-sm font-medium")
                                            ui.label(
                                                f"Resource: {str(recovery_proposal.get('resource_jid', '') or 'n/a')}"
                                            ).classes("text-xs text-slate-600")
                                            for idx, step in enumerate(
                                                recovery_proposal.get("macro_steps") or [], start=1
                                            ):
                                                if not isinstance(step, dict):
                                                    continue
                                                params_text = json.dumps(
                                                    step.get("params") or {}, default=str
                                                )
                                                ui.label(
                                                    f"{idx}. {step.get('function_name', '')}({params_text})"
                                                ).classes("text-xs font-mono text-slate-700")
                                    if recovery_approval_state == "pending":
                                        ui.label(
                                            "This final recovery candidate has already passed deterministic validation and is waiting for operator approval."
                                        ).classes("text-xs text-indigo-700 mt-2")
                                        review_box = (
                                            ui.textarea(
                                                label="Refinement or rejection feedback",
                                                value=recovery_feedback,
                                            )
                                            .props("outlined autogrow")
                                            .classes("w-full mt-2")
                                        )
                                        _track_recovery_input(review_box)
                                        review_box.on_value_change(
                                            lambda e,
                                            jid=product_jid: recovery_feedback_buffers.__setitem__(
                                                jid,
                                                str(e.value or ""),
                                            )
                                        )
                                        with ui.row().classes("gap-2 mt-2 flex-wrap"):
                                            approve_recovery_btn = ui.button(
                                                "Approve",
                                                on_click=lambda jid=product_jid: _queue_runtime_action(
                                                    jid,
                                                    action_label="Approve recovery proposal",
                                                    runner=_approve_runtime_recovery,
                                                ),
                                                icon="check_circle",
                                            ).props("color=green")
                                            refine_recovery_btn = ui.button(
                                                "Refine",
                                                on_click=lambda jid=product_jid: _queue_runtime_action(
                                                    jid,
                                                    action_label="Refine recovery proposal",
                                                    runner=_refine_runtime_recovery,
                                                ),
                                                icon="tune",
                                            ).props("color=amber")
                                            reject_recovery_btn = ui.button(
                                                "Reject",
                                                on_click=lambda jid=product_jid: _queue_runtime_action(
                                                    jid,
                                                    action_label="Reject recovery proposal",
                                                    runner=_reject_runtime_recovery,
                                                ),
                                                icon="cancel",
                                            ).props("color=red")
                                            for button in (
                                                approve_recovery_btn,
                                                refine_recovery_btn,
                                                reject_recovery_btn,
                                            ):
                                                button.set_enabled(
                                                    not recovery_reentry_locked
                                                    and not controls_locked
                                                )

                                if recovery_debug:
                                    debug_status = str(
                                        recovery_debug.get("status", "") or "n/a"
                                    ).strip()
                                    warning_messages = list(
                                        recovery_debug.get("warning_messages") or []
                                    )
                                    request_payload = (
                                        recovery_debug.get("request")
                                        if isinstance(recovery_debug.get("request"), dict)
                                        else {}
                                    )
                                    llm_inputs = (
                                        recovery_debug.get("llm_inputs")
                                        if isinstance(recovery_debug.get("llm_inputs"), dict)
                                        else {}
                                    )
                                    recovery_context = (
                                        recovery_debug.get("recovery_context")
                                        if isinstance(recovery_debug.get("recovery_context"), dict)
                                        else {}
                                    )
                                    modeled_check = (
                                        recovery_debug.get("modeled_continuation_check")
                                        if isinstance(
                                            recovery_debug.get("modeled_continuation_check"), dict
                                        )
                                        else {}
                                    )
                                    approval_info = (
                                        recovery_debug.get("approval")
                                        if isinstance(recovery_debug.get("approval"), dict)
                                        else {}
                                    )
                                    react_turns = list(recovery_debug.get("turns") or [])
                                    debug_prompt = str(recovery_debug.get("prompt", "") or "").strip()
                                    raw_response = str(
                                        recovery_debug.get("raw_response", "") or ""
                                    ).strip()
                                    recovery_proposal_debug = recovery_debug.get("recovery_proposal")
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
                                    compiled_rows = _recovery_task_rows_for_display(
                                        list(approval_info.get("compiled_recovery_task_ids") or [])
                                        + list(
                                            (active_recovery_sequence or {}).get("recovery_task_ids")
                                            or []
                                        ),
                                        node_lookup=node_lookup,
                                        task_states=task_states,
                                        fallback_rows=list(
                                            approval_info.get("compiled_recovery_tasks") or []
                                        ),
                                    )
                                    with ui.expansion(
                                        "Recovery Debug (temporary)",
                                        icon="bug_report",
                                        value=False,
                                    ).classes("w-full mt-2"):
                                        with ui.row().classes("items-center gap-2 flex-wrap"):
                                            ui.badge(
                                                f"Trace: {(debug_status or 'unknown').replace('_', ' ')}"
                                            ).props("color=teal")
                                            ui.badge(
                                                "Primitive recovery"
                                                if bool(recovery_debug.get("primitive_mode"))
                                                else "Legacy recovery"
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
                                        if status == "recovery_ready" and not raw_response:
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
                                        if recovery_proposal_debug:
                                            ui.label("Recovery proposal preview").classes(
                                                "text-xs font-medium mt-2"
                                            )
                                            ui.code(
                                                _preview_text(_json_text(recovery_proposal_debug)),
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
                                                "Recovery request inputs",
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
                                        if recovery_context:
                                            with ui.expansion(
                                                "Primitive recovery context",
                                                icon="precision_manufacturing",
                                                value=False,
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(recovery_context), language="json"
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
                                                value=status in {"llm_recovery", "human_required"},
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    raw_response,
                                                    language="text",
                                                ).classes("w-full text-xs")
                                        with ui.expansion(
                                            "Recovery proposal",
                                            icon="rule",
                                            value=status in {"llm_recovery", "human_required"},
                                        ).classes("w-full mt-2"):
                                            if warning_messages:
                                                ui.code(
                                                    "\n".join(
                                                        f"- {msg}" for msg in warning_messages
                                                    ),
                                                    language="text",
                                                ).classes("w-full text-xs")
                                            if recovery_proposal_debug:
                                                ui.code(
                                                    _json_text(recovery_proposal_debug),
                                                    language="json",
                                                ).classes("w-full text-xs")
                                            else:
                                                ui.label(
                                                    "No compilable recovery proposal was produced."
                                                ).classes("text-xs text-slate-600")
                                        if compiled_rows:
                                            with ui.expansion(
                                                "Compiled recovery tasks",
                                                icon="account_tree",
                                                value=status in {"llm_recovery", "human_required"},
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(compiled_rows),
                                                    language="json",
                                                ).classes("w-full text-xs")
                                        if active_recovery_sequence:
                                            with ui.expansion(
                                                "Recovery execution state",
                                                icon="play_circle",
                                                value=status
                                                in {"llm_recovery", "human_required", "resolved"},
                                            ).classes("w-full mt-2"):
                                                ui.code(
                                                    _json_text(active_recovery_sequence),
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
                                    _track_recovery_input(guidance_box)
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
                                            not recovery_reentry_locked and not controls_locked
                                        )
                                        submit_guidance_btn.set_enabled(not controls_locked)
                                        run_recovery_btn = ui.button(
                                            "Run recovery reasoning",
                                            on_click=lambda jid=product_jid: _queue_runtime_action(
                                                jid,
                                                action_label="Run recovery reasoning",
                                                runner=_run_runtime_recovery,
                                            ),
                                            icon="smart_toy",
                                        ).props("color=indigo")
                                        run_recovery_btn.set_enabled(
                                            not recovery_reentry_locked and not controls_locked
                                        )

                    for element in runtime_recovery_container.descendants():
                        if isinstance(element, ui.expansion):
                            values = expansion_values.get(element.text, [])
                            if values:
                                element.set_value(values.pop(0))

                _managed_timer(5.0, _refresh_runtime_recovery_panel)

            # ── Runtime Blocked Tasks ───────────────────────────────────
            with ui.card().classes("w-full"):
                ui.label("Runtime Blocked Tasks").classes("text-lg font-semibold mb-2")
                runtime_blocked_container = ui.column().classes("w-full gap-2")

                def _refresh_runtime_blocked_tasks():
                    ss = bridge.get_safety_state()
                    blocked = ss.get("blocked_tasks", {})
                    if not _changed("blocked", blocked):
                        return
                    runtime_blocked_container.clear()
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
                    if detail_table.rows != all_rows:
                        detail_table.rows = all_rows

                def _refresh_tasks():
                    nodes = _preview_or_runtime_nodes()
                    ts = _preview_or_runtime_task_states(nodes)
                    rows = [{"task_id": k, "status": v} for k, v in ts.items()]
                    if task_table.rows != rows:
                        task_table.rows = rows
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
                ) as timeline_expansion:
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

                async def _refresh_timeline():
                    if not timeline_expansion.value:
                        return
                    tl = await asyncio.to_thread(bridge.get_execution_timeline)
                    if timeline_table.rows != tl[-50:]:
                        timeline_table.rows = tl[-50:]

                timeline_expansion.on_value_change(_refresh_timeline)

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


def _read_prerequisites(bridge: SystemBridge, mode: str) -> dict:
    """Collect potentially blocking startup observations away from the UI loop."""
    if mode == "physical":
        hardware = bridge.hardware_connection_statuses()
        ready, reason = bridge.physical_perception_ready()
        return {"hardware": hardware, "ready": ready, "reason": reason}
    if mode != "simulation":
        return {"reason": "Select Simulation or Physical mode."}
    running = bridge.simulation_environment_running()
    passive = bridge.passive_digital_twin_environment_running()
    ready, reason = (
        bridge.simulation_start_ready()
        if running and not passive
        else (False, "Gazebo is not running.")
    )
    if passive:
        reason = "A passive hardware-authoritative Digital Twin is running. Select Physical mode."
    return {
        "gazebo_running": running,
        "passive_digital_twin_running": passive,
        "ready": ready,
        "reason": reason,
    }


def _check_prerequisites(
    bridge: SystemBridge,
    mode: str,
    banner: ui.column,
    *,
    launch_simulation: Callable[[], Any] | None = None,
    launch_simulation_busy: bool = False,
    status: dict | None = None,
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

    if mode not in _MODE_LABEL_BY_VALUE:

        def _render_unsupported_mode() -> None:
            ui.label("Select Simulation or Physical mode.").classes("text-sm text-amber-700")

        _replace_banner(("mode", mode), _render_unsupported_mode)
        return False

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

    if mode == "simulation":
        status = status if status is not None else _read_prerequisites(bridge, mode)
        gazebo_running = status["gazebo_running"]
        passive_digital_twin_running = status["passive_digital_twin_running"]
        sim_ready, sim_reason = status["ready"], status["reason"]

        def _render_simulation() -> None:
            if passive_digital_twin_running:
                with ui.row().classes("items-center gap-2 text-amber-700 bg-amber-50 p-3 rounded"):
                    ui.icon("warning").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label(
                            "A passive hardware-authoritative Digital Twin is running."
                        ).classes("text-sm font-semibold")
                        ui.label(
                            "Select Physical mode to start CAIS with this Digital Twin."
                        ).classes("text-sm")
            elif (
                gazebo_running
                and sim_ready
                and sim_reason
                and not _suppress_sim_ready_note(sim_reason)
            ):
                with ui.row().classes("items-center gap-2 text-amber-700 bg-amber-50 p-3 rounded"):
                    ui.icon("info").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("No-hardware dual Gazebo + MoveIt/RViz are ready.").classes(
                            "text-sm font-semibold"
                        )
                        ui.label(sim_reason).classes("text-xs")
            elif gazebo_running and sim_ready:
                with ui.row().classes("items-center gap-2 text-green-600"):
                    ui.icon("check_circle").classes("text-sm")
                    ui.label(
                        "No-hardware dual Gazebo + MoveIt/RViz are ready - safe to start."
                    ).classes("text-sm")
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
                        ui.label("Launch no-hardware dual Gazebo + MoveIt/RViz first.").classes(
                            "text-sm"
                        )
                        if launch_simulation is not None:
                            launch_btn = ui.button(
                                "start simulation",
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
                passive_digital_twin_running,
                sim_ready,
                sim_reason,
                bool(launch_simulation),
                launch_simulation_busy,
            ),
            _render_simulation,
        )
        return not passive_digital_twin_running and gazebo_running and sim_ready

    if mode == "physical":
        status = status if status is not None else _read_prerequisites(bridge, mode)
        hw = status["hardware"]
        xarm = hw.get("xarm6", {})
        ur5e = hw.get("ur5e", {})
        ready, reason = status["ready"], status["reason"]

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
    return False


def _stat_card(label: str, value: str, icon: str) -> None:
    with ui.column().classes("items-center"):
        ui.icon(icon).classes("text-2xl text-slate-500")
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-slate-500")
