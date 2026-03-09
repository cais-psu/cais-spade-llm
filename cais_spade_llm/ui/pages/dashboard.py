"""Dashboard page: system control panel, agent overview, plan DAG, and execution timeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

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


def render(bridge: SystemBridge) -> None:
    client = context.client
    timers: list[Any] = []

    def _managed_timer(interval: float, callback, **kwargs):
        def _safe_callback():
            if getattr(client, "_deleted", False):
                return None
            return callback()

        timer = ui.timer(interval, _safe_callback, **kwargs)
        timers.append(timer)
        return timer

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
        with ui.card().classes("w-full"):
            ui.label("System Control").classes("text-lg font-semibold mb-2")

            with ui.row().classes("items-end gap-6 flex-wrap"):
                startup_source_select = ui.radio(
                    ["Generate Plan At Startup", "Use Verified Plan Set"],
                    value="Use Verified Plan Set",
                ).props("inline")

            product_init_files = bridge.list_product_files()
            product_init_select = ui.select(
                {f: Path(f).stem for f in product_init_files},
                value=product_init_files[0] if product_init_files else None,
                label="Product",
            ).classes("w-64 mt-2")

            requirement_files = bridge.list_product_requirement_files()
            safety_files = bridge.list_safety_requirement_files()
            requirement_options = {f: Path(f).name for f in requirement_files}
            safety_options = {"__NONE__": "None"}
            for f in safety_files:
                safety_options[f] = Path(f).name

            with ui.row().classes("items-end gap-4 flex-wrap mt-2") as generate_source_row:
                requirement_select = ui.select(
                    requirement_options,
                    value=requirement_files[0] if requirement_files else None,
                    label="Product Requirement (.txt)",
                ).classes("w-80")

                safety_select = ui.select(
                    safety_options,
                    value=safety_files[0] if safety_files else "__NONE__",
                    label="Safety Requirement (.txt)",
                ).classes("w-80")

            with ui.row().classes("items-end gap-4 flex-wrap mt-2") as verified_source_row:
                plan_set_select = ui.select({}, label="Verified Plan Set").classes("w-96")

            source_hint_label = ui.label("").classes("text-xs text-slate-600")
            with ui.row().classes("items-end gap-4 flex-wrap mt-2"):
                mode_select = ui.radio(
                    _MODE_LABELS,
                    value="Simulation",
                ).props("inline")
            bridge.execution_mode = _MODE_MAP.get(mode_select.value, "simulation")
            bridge.robot_env = "gazebo" if bridge.execution_mode in ("dry_run", "simulation") else "real"

            # Prerequisite banner.
            prereq_banner = ui.column().classes("w-full mt-3")

            with ui.column().classes("w-full gap-2 mt-4"):
                banner_hide_task: asyncio.Task | None = None
                start_click_state = {"locked": False}
                gazebo_launch_state = {"busy": False}
                start_task: asyncio.Task | None = None

                def _selected_files() -> tuple[str, str]:
                    req_file = str(requirement_select.value or "").strip()
                    safe_file = str(safety_select.value or "").strip()
                    return req_file, safe_file

                def _update_source_picker_visibility() -> None:
                    use_verified = str(startup_source_select.value or "") == "Use Verified Plan Set"
                    generate_source_row.style("display:none;" if use_verified else "display:flex;")
                    verified_source_row.style("display:flex;" if use_verified else "display:none;")

                def _refresh_plan_set_options() -> None:
                    options: dict[str, str] = {}
                    selected_product_stem = Path(
                        str(product_init_select.value or "")
                    ).stem or ""

                    for row in bridge.list_bundles():
                        bid = str(row.get("bundle_id", "")).strip()
                        if not bid:
                            continue
                        status = str(row.get("status", "")).lower()
                        if status != "verified":
                            continue
                        bundle_product = str(row.get("product_name", "")).strip()
                        if selected_product_stem and bundle_product and bundle_product != selected_product_stem:
                            continue
                        created = str(row.get("created_at_utc", ""))[:19].replace("T", " ")
                        options[bid] = f"{created} | {bid}"

                    current = str(plan_set_select.value or "").strip()
                    plan_set_select.options = options
                    plan_set_select.update()
                    if current in options:
                        plan_set_select.value = current
                    else:
                        plan_set_select.value = next(iter(options.keys()), None)

                    if str(startup_source_select.value) == "Use Verified Plan Set":
                        source_hint_label.text = (
                            "Startup source: verified plan set (runtime plan/safety generation skipped)."
                        )
                    else:
                        source_hint_label.text = (
                            "Startup source: runtime generation from selected requirement/safety files."
                        )

                    plan_set_select.set_enabled(str(startup_source_select.value) == "Use Verified Plan Set")

                def _bundle_gate(*, strict: bool) -> tuple[bool, str]:
                    source = str(startup_source_select.value or "")
                    if source == "Use Verified Plan Set":
                        bid = str(plan_set_select.value or "").strip()
                        if not bid:
                            return False, "Select a verified plan set."
                        return True, ""

                    req_file, _ = _selected_files()
                    if not req_file:
                        return False, "Select a product requirement file."
                    return True, ""

                def _set_action_banner(kind: str, message: str, *, auto_hide_s: float | None = None) -> None:
                    nonlocal banner_hide_task
                    style_map = {
                        "info": ("info", "text-blue-700 bg-blue-50"),
                        "success": ("check_circle", "text-green-700 bg-green-50"),
                        "warning": ("warning", "text-amber-700 bg-amber-50"),
                        "error": ("error", "text-red-700 bg-red-50"),
                    }
                    icon_name, color_classes = style_map.get(kind, style_map["info"])

                    action_banner_icon.name = icon_name
                    action_banner.classes(remove="text-blue-700 bg-blue-50 text-green-700 bg-green-50 text-amber-700 bg-amber-50 text-red-700 bg-red-50")
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
                        _set_action_banner("warning", "System is already running.", auto_hide_s=4.0)
                        return
                    if start_click_state["locked"]:
                        _set_action_banner("warning", "Start already in progress...")
                        return
                    if bridge._starting:
                        _set_action_banner("warning", "Start already in progress...")
                        return
                    internal = _MODE_MAP[mode_select.value]
                    bridge.execution_mode = internal
                    bridge.robot_env = "gazebo" if internal in ("dry_run", "simulation") else "real"

                    bundle_ok, bundle_msg = _bundle_gate(strict=True)
                    if not bundle_ok:
                        _set_action_banner("warning", bundle_msg, auto_hide_s=8.0)
                        return

                    source = str(startup_source_select.value or "")
                    try:
                        if source == "Use Verified Plan Set":
                            selected_plan_set_id = str(plan_set_select.value or "").strip()
                            if not selected_plan_set_id:
                                _set_action_banner("warning", "Select a verified plan set.", auto_hide_s=6.0)
                                return
                            bridge.set_active_bundle(selected_plan_set_id)

                            active = bridge.get_active_bundle() or {}
                            manifest = active.get("manifest", {}) if isinstance(active, dict) else {}
                            if not isinstance(manifest, dict):
                                manifest = {}
                            bundle_req_file = str(manifest.get("product_spec_file", "")).strip()
                            bundle_safe_file = str(manifest.get("safety_file", "")).strip()
                            if not bundle_req_file:
                                _set_action_banner(
                                    "error",
                                    "Selected verified plan set is missing product requirement file in manifest.",
                                    auto_hide_s=8.0,
                                )
                                return
                            try:
                                bridge.selected_product = bridge.resolve_product_init_for_requirement(bundle_req_file)
                            except Exception as exc:
                                _set_action_banner(
                                    "error",
                                    f"Verified plan set product mapping failed: {exc}",
                                    auto_hide_s=8.0,
                                )
                                return

                            # In verified-plan mode, startup should use the plan set's own files.
                            bridge.selected_requirement_file = bundle_req_file
                            bridge.selected_safety_file = bundle_safe_file

                            # Keep visible selectors in sync when values exist in current options.
                            req_opts = requirement_select.options if isinstance(requirement_select.options, dict) else {}
                            safe_opts = safety_select.options if isinstance(safety_select.options, dict) else {}
                            if bundle_req_file in req_opts:
                                requirement_select.value = bundle_req_file
                            if bundle_safe_file in safe_opts:
                                safety_select.value = bundle_safe_file
                        else:
                            req_file, safe_file = _selected_files()
                            if not req_file:
                                _set_action_banner("warning", "Select a product requirement file.", auto_hide_s=6.0)
                                return
                            selected_product = str(product_init_select.value or "").strip()
                            if selected_product:
                                bridge.selected_product = selected_product
                            else:
                                try:
                                    bridge.selected_product = bridge.resolve_product_init_for_requirement(req_file)
                                except Exception as exc:
                                    _set_action_banner("error", f"Invalid requirement selection: {exc}", auto_hide_s=8.0)
                                    return

                            bridge.selected_requirement_file = req_file
                            bridge.selected_safety_file = safe_file or ""
                            bridge.set_active_bundle(None)
                    except Exception as exc:
                        _set_action_banner("error", f"Failed to configure startup source: {exc}", auto_hide_s=8.0)
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
                                        _set_action_banner("success", "System started successfully.", auto_hide_s=5.0)
                                else:
                                    reason = bridge.last_error or "unknown error"
                                    _set_action_banner("error", f"Start failed: {reason}", auto_hide_s=8.0)
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
                        _set_action_banner("warning", "System is already stopped.", auto_hide_s=4.0)
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
                            _set_action_banner("error", "Stop failed. System is still running.", auto_hide_s=8.0)
                    finally:
                        _update_controls()

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
                        err = await asyncio.to_thread(bridge.ros2_start, "gazebo_dual")
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
                        _set_action_banner("warning", "Start is in progress. Wait before reset.", auto_hide_s=4.0)
                        return
                    if bridge._stopping:
                        _set_action_banner("warning", "Stop is in progress. Wait before reset.", auto_hide_s=4.0)
                        return
                    selected_scope = str(reset_scope_select.value or "Reset All").strip() or "Reset All"

                    _set_action_banner("info", f"{selected_scope} clicked. Preparing reset...")
                    try:
                        # Keep agent/world state consistent: stop system first if needed.
                        if bridge.system_running:
                            _set_action_banner("info", f"Stopping system before {selected_scope.lower()}...")
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
                            ok_plan, msg_plan = await asyncio.to_thread(bridge.reset_plan_runtime_state)
                            if ok_plan:
                                success_messages.append(msg_plan)
                            else:
                                warning_messages.append(msg_plan)

                        if selected_scope in {"Reset Gazebo", "Reset All"}:
                            ok_gz, msg_gz = await asyncio.to_thread(bridge.ros2_reset_gazebo_environment)
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
                            _set_action_banner("warning", " ; ".join(warning_messages), auto_hide_s=8.0)
                        else:
                            _set_action_banner(
                                "success",
                                " ; ".join(success_messages) or "Reset complete. Click Start System.",
                                auto_hide_s=6.0,
                            )
                    finally:
                        _update_controls()

                with ui.row().classes("items-end gap-4 flex-wrap"):
                    start_btn = ui.button("Start System", on_click=_start, icon="play_arrow").props("color=green")
                    stop_btn = ui.button("Stop System", on_click=_stop, icon="stop").props("color=red")
                    reset_scope_select = ui.select(
                        {opt: opt for opt in reset_scope_options},
                        value="Reset All",
                        label="Reset Scope",
                    ).classes("w-44")
                    reset_btn = ui.button("Reset", on_click=_reset_selected, icon="restart_alt").props("color=blue")
                action_banner = ui.row().classes(
                    "w-full mt-2 items-center gap-2 rounded p-3 text-sm text-blue-700 bg-blue-50"
                )
                action_banner.style("display:none;")
                with action_banner:
                    action_banner_icon = ui.icon("info")
                    action_banner_label = ui.label("")
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
                plan_safety_banner_label.text = f"{product} [{stage}] {message}{retry_text}{extra}"
                plan_safety_banner.style("display:flex;")

            def _update_controls():
                try:
                    internal = _MODE_MAP.get(mode_select.value, "dry_run")
                    bridge.execution_mode = internal
                    bridge.robot_env = "gazebo" if internal in ("dry_run", "simulation") else "real"
                    if internal == "physical" and hasattr(bridge, "hardware_connection_statuses"):
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
                        prereqs_met = False
                        if not bridge.system_running:
                            with prereq_banner:
                                with ui.row().classes("items-center gap-2 text-amber-700 bg-amber-50 p-3 rounded"):
                                    ui.icon("warning").classes("text-lg")
                                    ui.label(bundle_msg).classes("text-sm font-semibold")

                    can_start = (
                        prereqs_met
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
                    stop_btn.set_enabled(bridge.system_running and not bridge._stopping)
                    reset_btn.set_enabled(
                        can_reset
                        and not bridge._starting
                        and not bridge._stopping
                    )
                    _set_plan_safety_banner(bridge.get_plan_safety_alerts())
                    error_label.text = bridge.last_error or ""
                except Exception as exc:
                    if hasattr(bridge, "_diag_emit"):
                        bridge._diag_emit(f"dashboard update_controls exception: {exc}")
                    start_btn.set_enabled(False)
                    stop_btn.set_enabled(False)
                    reset_btn.set_enabled(False)
                    _set_plan_safety_banner([])
                    error_label.text = f"Dashboard control update failed: {exc}"

            _managed_timer(1.0, _update_controls)

            def _refresh_selection_and_controls() -> None:
                _update_source_picker_visibility()
                _refresh_plan_set_options()
                _update_controls()
                refresh_dag_now()

            product_init_select.on_value_change(lambda e: _refresh_selection_and_controls())
            requirement_select.on_value_change(lambda e: _refresh_selection_and_controls())
            safety_select.on_value_change(lambda e: _refresh_selection_and_controls())
            startup_source_select.on_value_change(lambda e: _refresh_selection_and_controls())
            plan_set_select.on_value_change(lambda e: _refresh_selection_and_controls())
            mode_select.on_value_change(lambda e: _refresh_selection_and_controls())
            reset_scope_select.on_value_change(lambda e: _refresh_selection_and_controls())
            _refresh_selection_and_controls()

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
                completed = sum(1 for v in task_states.values() if "completed" in v.lower()) if task_states else 0
                failed = sum(1 for v in task_states.values() if "failed" in v.lower()) if task_states else 0

                _stat_card("Tasks Completed", f"{completed}/{total}", "task_alt")
                _stat_card("Tasks Failed", str(failed), "error_outline")

                robot_states = bridge.get_robot_states()
                active = sum(1 for r in robot_states.values() if r.get("current_state", "idle") != "idle")
                _stat_card("Active Robots", str(active), "precision_manufacturing")

                safety = bridge.get_safety_state()
                blocked = len(safety.get("blocked_tasks", {}))
                _stat_card("Safety Blocks", str(blocked), "shield")

        _managed_timer(2.0, _refresh_stats)

        def _preview_or_runtime_nodes() -> list[dict]:
            nodes = bridge.get_plan_nodes()
            if nodes or bridge.system_running:
                return nodes
            source = str(startup_source_select.value or "")
            selected_plan_set_id = str(plan_set_select.value or "").strip()
            if source == "Use Verified Plan Set" and selected_plan_set_id:
                return bridge.get_bundle_plan_nodes(selected_plan_set_id)
            return []

        def _preview_or_runtime_task_states(nodes: list[dict]) -> dict[str, str]:
            task_states = bridge.get_task_states()
            if task_states:
                return task_states
            fallback: dict[str, str] = {}
            for node in nodes:
                tid = str(node.get("id") or node.get("task_id") or "").strip()
                if not tid:
                    continue
                fallback[tid] = str(node.get("status", "pending") or "pending")
            return fallback

        # ── Task DAG ────────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Task DAG").classes("text-lg font-semibold mb-2")
            mermaid = ui.mermaid("graph TD\n    empty[No plan loaded]").classes("w-full")

            def _refresh_dag():
                nodes = _preview_or_runtime_nodes()
                task_states = _preview_or_runtime_task_states(nodes)
                mermaid.content = nodes_to_mermaid(nodes, task_states)

            refresh_dag_now = _refresh_dag
            _refresh_dag()
            _managed_timer(2.0, _refresh_dag)

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
                    {"name": "generated_interpretation", "label": "Interpretation", "field": "generated_interpretation"},
                    {"name": "constraint_type", "label": "Type", "field": "constraint_type"},
                ],
                rows=[],
            ).classes("w-full")

            def _refresh_runtime_safety_rules():
                selected_bundle_id = ""
                if str(startup_source_select.value or "") == "Use Verified Plan Set":
                    selected_bundle_id = str(plan_set_select.value or "").strip()

                rules = bridge.get_safety_rules(selected_bundle_id or None)
                if bridge.system_running and bridge.cca and getattr(bridge.cca, "safety_rules", None):
                    runtime_rules_source.text = "Source: live runtime rules from the active controller"
                elif selected_bundle_id:
                    runtime_rules_source.text = f"Source: selected verified plan set {selected_bundle_id}"
                else:
                    active = bridge.get_active_bundle() or {}
                    active_id = str(active.get("bundle_id", "")).strip()
                    runtime_rules_source.text = (
                        f"Source: active verified plan set {active_id}"
                        if active_id
                        else "Source: no runtime or verified bundle safety loaded"
                    )
                rows = []
                for i, r in enumerate(rules):
                    rows.append(
                        {
                            "id": r.get("id", f"R{i}"),
                            "raw_text": r.get("raw_text", r.get("text", str(r))),
                            "constraint_type": r.get("constraint_type", ""),
                            "generated_interpretation": r.get("generated_interpretation", r.get("ltlf_plain_feedback", "")),
                            "ltlf": r.get("ltlf", r.get("formula", "")),
                        }
                    )
                runtime_rules_table.rows = rows

            _managed_timer(5.0, _refresh_runtime_safety_rules)

        # ── Runtime Safety State ────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Runtime Safety State").classes("text-lg font-semibold mb-2")
            runtime_safety_state = ui.code("{}", language="json").classes("w-full")

            def _refresh_runtime_safety_state():
                ss = bridge.get_safety_state()
                runtime_safety_state.content = json.dumps(ss, indent=2, default=str) if ss else "{}"

            _managed_timer(2.0, _refresh_runtime_safety_state)

        # ── Replan / Recovery ───────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Replan / Recovery").classes("text-lg font-semibold mb-2")
            ui.label(
                "Runtime DES recovery stays inside one workflow: DES search, narrow LLM bridge, plan validation, then human intervention if needed."
            ).classes("text-xs text-slate-500 mb-2")
            runtime_recovery_container = ui.column().classes("w-full gap-3")

            guidance_buffers: dict[str, str] = {}

            def _status_color(status: str) -> str:
                key = str(status or "").strip().lower()
                if key == "resolved":
                    return "green"
                if key == "human_required":
                    return "red"
                if key in {"des_search", "llm_bridge", "validating"}:
                    return "blue"
                return "grey"

            def _status_label(status: str) -> str:
                labels = {
                    "idle": "Idle",
                    "des_search": "DES search",
                    "llm_bridge": "LLM bridge",
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
                    ("llm_bridge", "LLM bridge"),
                    ("validating", "Plan validation"),
                    ("human_required", "Human intervention"),
                ]
                for key, label in stages:
                    color = "grey"
                    if status == "resolved":
                        if key == "des_search":
                            color = "green"
                        elif key == "llm_bridge" and used_bridge:
                            color = "green"
                        elif key == "validating":
                            color = "green"
                    elif status == key:
                        color = "red" if key == "human_required" else "blue"
                    elif key == "des_search" and status in {"llm_bridge", "validating", "human_required"}:
                        color = "green"
                    elif key == "llm_bridge" and used_bridge and status in {"validating", "human_required"}:
                        color = "green"
                    elif key == "validating" and status == "human_required":
                        color = "green"
                    badges.append((label, color))
                return badges

            async def _submit_runtime_guidance(product_jid: str) -> None:
                message = str(guidance_buffers.get(product_jid, "")).strip()
                if not message:
                    ui.notify("Operator guidance is empty.", type="warning")
                    return
                try:
                    await asyncio.to_thread(
                        bridge.submit_runtime_recovery_guidance,
                        product_jid,
                        message,
                    )
                    guidance_buffers[product_jid] = ""
                    ui.notify("Operator guidance recorded.", type="positive")
                    _refresh_runtime_recovery_panel()
                except Exception as exc:
                    ui.notify(f"Failed to record guidance: {exc}", type="negative")

            async def _retry_runtime_des(product_jid: str) -> None:
                try:
                    await asyncio.to_thread(bridge.retry_runtime_recovery_des, product_jid)
                    ui.notify("Runtime DES recovery retry started.", type="positive")
                    _refresh_runtime_recovery_panel()
                except Exception as exc:
                    ui.notify(f"Failed to retry DES recovery: {exc}", type="negative")

            def _refresh_runtime_recovery_panel() -> None:
                runtime_recovery_container.clear()
                recoveries = bridge.get_runtime_recoveries()
                if not recoveries:
                    with runtime_recovery_container:
                        ui.label("No runtime recovery sessions.").classes("text-slate-400 italic")
                    return

                with runtime_recovery_container:
                    for recovery in recoveries:
                        if not isinstance(recovery, dict):
                            continue
                        product_jid = str(recovery.get("product_jid", "")).strip()
                        product_name = str(recovery.get("product_name", product_jid) or product_jid or "product")
                        status = str(recovery.get("status", "idle") or "idle")
                        resolution = str(recovery.get("resolution_class", "none") or "none")
                        message = str(recovery.get("message", "")).strip() or "No recovery activity."
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
                        guidance_buffers[product_jid] = operator_guidance

                        with ui.card().classes("w-full bg-slate-50"):
                            with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                                ui.label(product_name).classes("font-semibold")
                                with ui.row().classes("items-center gap-2 flex-wrap"):
                                    ui.badge(_status_label(status)).props(f"color={_status_color(status)}")
                                    ui.badge(_resolution_label(resolution)).props(
                                        f"color={_resolution_color(resolution)}"
                                    )

                            with ui.row().classes("items-center gap-2 flex-wrap text-xs text-slate-600"):
                                ui.label(f"Trigger: {str(recovery.get('trigger', '') or 'n/a')}")
                                ui.label(f"Failed task: {str(recovery.get('failed_task_id', '') or 'n/a')}")
                                if str(recovery.get("replan_mode", "")).strip().lower() == "llm":
                                    ui.badge("Experimental full-LLM mode").props("color=orange")

                            with ui.row().classes("items-center gap-2 flex-wrap mt-1"):
                                for stage_label, stage_color in _stage_badges(recovery):
                                    ui.badge(stage_label).props(f"color={stage_color}")

                            ui.label(message).classes("text-sm mt-2")
                            ui.label(
                                f"Attempts: {attempts_used}/{attempts_max} | "
                                f"Witnesses: {witness_count} | "
                                f"Violated rules: {', '.join(violated_rules) if violated_rules else 'none'}"
                            ).classes("text-xs text-slate-600")

                            history = list(recovery.get("history") or [])
                            if history:
                                with ui.expansion("Recent history", icon="history", value=False).classes("w-full mt-2"):
                                    for item in history[-6:]:
                                        if not isinstance(item, dict):
                                            continue
                                        stamp = str(item.get("timestamp", "")).strip()
                                        item_status = _status_label(str(item.get("status", "")).strip())
                                        item_message = str(item.get("message", "")).strip()
                                        ui.label(f"{stamp} | {item_status} | {item_message}").classes(
                                            "text-xs text-slate-600"
                                        )

                            if status == "human_required":
                                guidance_box = ui.textarea(
                                    label="Operator guidance",
                                    value=operator_guidance,
                                ).props("outlined autogrow").classes("w-full mt-2")
                                guidance_box.on_value_change(
                                    lambda e, jid=product_jid: guidance_buffers.__setitem__(
                                        jid,
                                        str(e.value or ""),
                                    )
                                )
                                with ui.row().classes("gap-2 mt-2 flex-wrap"):
                                    ui.button(
                                        "Retry DES",
                                        on_click=lambda jid=product_jid: asyncio.create_task(
                                            _retry_runtime_des(jid)
                                        ),
                                        icon="restart_alt",
                                    ).props("color=blue")
                                    ui.button(
                                        "Submit operator guidance",
                                        on_click=lambda jid=product_jid: asyncio.create_task(
                                            _submit_runtime_guidance(jid)
                                        ),
                                        icon="person",
                                    ).props("color=amber")

            _refresh_runtime_recovery_panel()
            _managed_timer(3.0, _refresh_runtime_recovery_panel)

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
                    {"name": "task_id", "label": "Task ID", "field": "task_id", "sortable": True},
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
                    node = next((n for n in nodes if (n.get("id") or n.get("task_id")) == tid), None)
                    if not node:
                        continue
                    if all_rows:
                        all_rows.append({"row_id": f"sep_{row_idx}", "key": "———", "value": "———"})
                        row_idx += 1
                    for k, v in node.items():
                        display = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
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
                            (k for k, v in reversed(ts.items())
                             if any(s in v.lower() for s in ("dispatched", "accepted", "completed", "failed"))),
                            None,
                        )
                    if active:
                        _set_node_details([active])

            _managed_timer(2.0, _refresh_tasks)

        # ── Execution Timeline ──────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.expansion("Execution Timeline", icon="timeline", value=False).classes("w-full"):
                ui.label("Expand this panel to inspect the latest execution events.").classes(
                    "text-xs text-slate-500 mb-2"
                )
                timeline_table = ui.table(
                    columns=[
                        {"name": "timestamp", "label": "Time", "field": "timestamp", "sortable": True},
                        {"name": "task_id", "label": "Task", "field": "task_id", "sortable": True},
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
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
                for row in (e.selection or []):
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
    banner.clear()

    def _suppress_sim_ready_note(message: str) -> bool:
        text = str(message or "").strip().lower()
        return text.startswith("perception is still warming up:")

    if bridge.system_running:
        return True  # Already running, don't block.

    if bridge._starting:
        with banner:
            with ui.row().classes("items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded"):
                ui.icon("hourglass_top").classes("text-lg")
                ui.label("System startup in progress...").classes("text-sm font-semibold")
        return False

    if mode == "dry_run":
        # Dry Run has no prerequisites.
        with banner:
            with ui.row().classes("items-center gap-2 text-green-600"):
                ui.icon("check_circle").classes("text-sm")
                ui.label("Dry Run mode — no prerequisites required.").classes("text-sm")
        return True

    if mode == "simulation":
        # Simulation needs Gazebo stack plus MoveIt/services ready.
        statuses = bridge.ros2_all_statuses()
        gazebo_running = any(
            statuses.get(k) == "running"
            for k in ("gazebo_dual", "gazebo_xarm6", "gazebo_ur5e")
        )
        sim_ready = False
        sim_reason = ""
        if gazebo_running and hasattr(bridge, "simulation_start_ready"):
            sim_ready, sim_reason = bridge.simulation_start_ready()
        elif gazebo_running:
            sim_ready = True

        with banner:
            if gazebo_running and sim_ready and sim_reason and not _suppress_sim_ready_note(sim_reason):
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
                        ui.label(sim_reason or "Waiting for ROS services/prewarm to complete...").classes("text-xs")
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

        with banner:
            if not ready:
                with ui.row().classes("items-center gap-2 text-red-700 bg-red-50 p-3 rounded"):
                    ui.icon("error").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Physical mode is blocked.").classes("text-sm font-semibold")
                        ui.label(reason).classes("text-xs")
                        ui.label(_line("xArm6", xarm)).classes("text-xs")
                        ui.label(_line("UR5e", ur5e)).classes("text-xs")
            else:
                with ui.row().classes("items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded"):
                    ui.icon("info").classes("text-lg")
                    with ui.column().classes("gap-1"):
                        ui.label("Physical mode — ensure robots are powered on and controllers are running.").classes("text-sm")
                        ui.label(_line("xArm6", xarm)).classes("text-xs")
                        ui.label(_line("UR5e", ur5e)).classes("text-xs")
        return ready

    return True


def _stat_card(label: str, value: str, icon: str) -> None:
    with ui.column().classes("items-center"):
        ui.icon(icon).classes("text-2xl text-slate-500")
        ui.label(value).classes("text-xl font-bold")
        ui.label(label).classes("text-xs text-slate-500")
