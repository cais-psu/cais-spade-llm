"""Control page: Gazebo/hardware launch, interactive teleop with arrow buttons + keyboard."""

from __future__ import annotations

import asyncio
import logging
import math

from nicegui import context, ui
from nicegui.client import Client
from nicegui.events import KeyEventArguments

from cais_spade_llm.ui.bridge import SystemBridge

log = logging.getLogger(__name__)


# Gazebo launch variants with friendly labels.
_GAZEBO_VARIANTS = {
    "gazebo_dual": ("Dual Robots (xArm6 + UR5e)", "Full dual-robot Gazebo + MoveIt + RViz"),
    "gazebo_xarm6": ("xArm6 Only", "Single xArm6 Gazebo + MoveIt + RViz"),
    "gazebo_ur5e": ("UR5e Only", "Single UR5e + RG2 Gazebo + MoveIt + RViz"),
}

_HARDWARE_STACKS = {
    "xarm6": ("xArm6 Hardware Stack", "Start xArm6 MoveIt realmove stack (includes embedded driver)"),
    "ur5e": ("UR5e Hardware Stack", "Auto sequence: start driver, wait for ready, then start RG2 gripper bridge and MoveIt"),
}
_HARDWARE_PROC_NAMES = (
    "hardware_xarm6_driver",
    "hardware_xarm6_moveit",
    "hardware_ur5e_driver",
    "hardware_ur5e_rg2_gripper",
    "hardware_ur5e_moveit",
)

_SUPPORT_PROCS = {
    "perception": ("Perception", "Simulation-only: part detection via Gazebo ground-truth camera"),
}


def _client_alive(element) -> bool:
    try:
        client = getattr(element, "client", None)
        if client is None:
            return False
        return (client.id in Client.instances) and (not getattr(client, "_deleted", False))
    except RuntimeError:
        return False


def _hardware_status_text(status: dict) -> str:
    robot_bits = []
    for robot in ("xarm6", "ur5e"):
        robot_status = status.get(robot)
        if isinstance(robot_status, dict):
            robot_bits.append(f"{robot}: {_hardware_status_text(robot_status)}")
    if robot_bits:
        return " | ".join(robot_bits)
    driver_state = str(status.get("driver", "stopped"))
    moveit_state = str(status.get("moveit", "stopped"))
    gripper_state = status.get("gripper")
    if gripper_state is None:
        return f"Driver: {driver_state} | MoveIt: {moveit_state}"
    return f"Driver: {driver_state} | Gripper: {str(gripper_state)} | MoveIt: {moveit_state}"


def _ur5e_status_sources(status: dict) -> list[tuple[str, dict]]:
    ur5e_status = status.get("ur5e")
    if isinstance(ur5e_status, dict):
        return [("ur5e ", ur5e_status)]
    return [("", status)]


def _render_ur5e_external_control_and_rg2_status(status: dict) -> None:
    for prefix, source in _ur5e_status_sources(status):
        external_control = str(source.get("external_control") or "").strip()
        if external_control:
            ui.label(f"{prefix}External Control: {external_control}").classes("text-xs text-slate-500")
            external_error = str(source.get("external_control_error") or "").strip()
            if external_error:
                ui.label(f"{prefix}External Control error: {external_error}").classes("text-xs text-red-700")

        gripper_state = source.get("gripper")
        if gripper_state is not None:
            ui.label(f"{prefix}RG2 bridge: {str(gripper_state)}").classes("text-xs text-slate-500")
        gripper_action = str(source.get("gripper_action") or "").strip()
        if gripper_action:
            ui.label(f"{prefix}RG2 bridge: action {gripper_action}").classes("text-xs text-slate-500")


def _install_control_key_scroll_blocker() -> None:
    """Prevent page scrolling when teleop keys are used on the Control page."""
    ui.add_head_html(
        """
<script>
if (!window.__caisTeleopScrollBlockerInstalled) {
  window.__caisTeleopScrollBlockerInstalled = true;
  document.addEventListener('keydown', function(evt) {
    if (!window.location.pathname.includes('/control')) return;
    const focus = document.activeElement;
    const tag = focus && focus.tagName ? focus.tagName.toUpperCase() : '';
    if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(tag)) return;
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', ' ', 'Spacebar'].includes(evt.key)) {
      evt.preventDefault();
    }
  }, { passive: false });
}
</script>
        """
    )


def render(bridge: SystemBridge) -> None:
    _install_control_key_scroll_blocker()
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Control").classes("text-2xl font-bold")

        # ── Gazebo / Hardware Launch ─────────────────────────────────
        _launch_section(bridge)

        # ── Digital Twin Launch ──────────────────────────────────────
        _digital_twin_launch_section(bridge)

        # ── Interactive Teleop ───────────────────────────────────────
        _teleop_section(bridge)


# =====================================================================
# Launch Section
# =====================================================================
def _launch_section(bridge: SystemBridge) -> None:
    with ui.card().classes("w-full"):
        ui.label("Launch Environment").classes("text-lg font-semibold mb-1")
        ui.label(
            "Start Gazebo simulation or connect to physical hardware. "
            "The environment must be running before starting the agent system from the Dashboard."
        ).classes("text-xs text-slate-500 mb-3")
        ui.label(
            "Hardware stacks are combined per arm. UR5e starts Driver, RG2 gripper bridge, then MoveIt; xArm6 MoveIt realmove includes its driver."
        ).classes("text-xs text-slate-500 mb-2")

        hw_refresh_state = {"busy": False}
        ips = bridge.get_hardware_ips()
        with ui.column().classes("w-full gap-2 mb-3"):
            ui.label("Hardware Connectivity (Ping)").classes("text-sm font-semibold text-slate-600")

            with ui.row().classes("items-center gap-2"):
                xarm_ping_icon = ui.icon("circle", color="grey").classes("text-xs")
                xarm_ping = ui.label("xArm6: checking...").classes("text-xs")
            with ui.row().classes("items-center gap-2"):
                ur5e_ping_icon = ui.icon("circle", color="grey").classes("text-xs")
                ur5e_ping = ui.label("UR5e: checking...").classes("text-xs")
            hw_warning = ui.label("").classes("text-xs text-amber-700 bg-amber-50 px-2 py-1 rounded")
            hw_warning.set_visibility(False)

            def _format_ping(name: str, entry: dict) -> str:
                ip = str(entry.get("ip", ""))
                if entry.get("reachable"):
                    latency = entry.get("latency_ms")
                    if latency is not None:
                        return f"{name}: {ip} reachable ({latency:.1f} ms)"
                    return f"{name}: {ip} reachable"
                return f"{name}: {ip} unreachable ({entry.get('message', 'no reply')})"

            def _ping_color(entry: dict) -> str:
                if entry.get("reachable"):
                    return "green"
                msg = str(entry.get("message", "")).strip().lower()
                if msg in {"", "checking...", "no ip configured"}:
                    return "grey"
                return "red"

            with ui.row().classes("items-end gap-3 flex-wrap"):
                xarm_ip_input = ui.input("xArm6 IP", value=ips.get("xarm6", "")).props("dense").classes("w-44")
                ur5e_ip_input = ui.input("UR5e IP", value=ips.get("ur5e", "")).props("dense").classes("w-44")

                def _refresh_ping_labels() -> None:
                    if not _client_alive(xarm_ping):
                        return
                    hw_links = bridge.hardware_connection_statuses_cached()
                    xarm_entry = hw_links.get("xarm6", {})
                    ur5e_entry = hw_links.get("ur5e", {})
                    xarm_ping.set_text(_format_ping("xArm6", xarm_entry))
                    ur5e_ping.set_text(_format_ping("UR5e", ur5e_entry))
                    xarm_ping_icon.props(f"color={_ping_color(xarm_entry)}")
                    ur5e_ping_icon.props(f"color={_ping_color(ur5e_entry)}")
                    all_ok = all((hw_links.get(r, {}).get("reachable", False) for r in ("xarm6", "ur5e")))
                    if all_ok:
                        hw_warning.set_visibility(False)
                    else:
                        hw_warning.set_text("One or more hardware robots are unreachable. Check power/network before launch.")
                        hw_warning.set_visibility(True)

                async def _refresh_ping_async(force: bool = False) -> None:
                    if hw_refresh_state["busy"]:
                        return
                    if not _client_alive(xarm_ping):
                        return
                    hw_refresh_state["busy"] = True
                    try:
                        await asyncio.to_thread(bridge.hardware_connection_statuses, force)
                        if not _client_alive(xarm_ping):
                            return
                        _refresh_ping_labels()
                    finally:
                        hw_refresh_state["busy"] = False

                def _apply_hw_ips():
                    updates = {
                        "xarm6": xarm_ip_input.value,
                        "ur5e": ur5e_ip_input.value,
                    }
                    errors = []
                    for robot, ip in updates.items():
                        err = bridge.set_hardware_ip(robot, str(ip or "").strip())
                        if err:
                            errors.append(err)
                    if errors:
                        ui.notify("; ".join(errors), type="warning", timeout=4000)
                        return
                    ui.notify("Hardware IPs updated", type="positive", timeout=1200)
                    asyncio.create_task(_refresh_ping_async(force=True))

                ui.button("Apply Hardware IPs", on_click=_apply_hw_ips, icon="save").props("outline dense")

            _refresh_ping_labels()

        launch_container = ui.column().classes("w-full gap-3")
        refresh_state = {"signature": None, "busy": False}

        def _launch_signature() -> tuple:
            statuses = bridge.ros2_all_statuses()
            return (
                tuple((name, statuses.get(name, "stopped")) for name in _GAZEBO_VARIANTS),
                tuple((name, statuses.get(name, "stopped")) for name in _HARDWARE_PROC_NAMES),
                tuple((name, statuses.get(name, "stopped")) for name in _SUPPORT_PROCS),
            )

        def _refresh(*, force: bool = False):
            if not _client_alive(launch_container):
                return
            if refresh_state["busy"]:
                return
            refresh_state["busy"] = True
            try:
                signature = _launch_signature()
                if not force and signature == refresh_state["signature"]:
                    asyncio.create_task(_refresh_ping_async())
                    return
                refresh_state["signature"] = signature
                statuses = dict(signature[0] + signature[1] + signature[2])
            finally:
                refresh_state["busy"] = False

            launch_container.clear()
            asyncio.create_task(_refresh_ping_async())
            any_gazebo_running = any(statuses.get(name) == "running" for name in _GAZEBO_VARIANTS)
            any_hardware_running = any(statuses.get(name) == "running" for name in _HARDWARE_PROC_NAMES)

            with launch_container:
                # Gazebo variants.
                ui.label("Gazebo Simulation").classes("text-sm font-semibold text-slate-600")
                for name, (label, desc) in _GAZEBO_VARIANTS.items():
                    blocked_reason = None
                    if any_hardware_running and statuses.get(name, "stopped") != "running":
                        blocked_reason = "Blocked: hardware stack is running. Stop hardware first."
                    _proc_row(bridge, name, label, desc, statuses.get(name, "stopped"), blocked_reason=blocked_reason)

                ui.separator().classes("my-2")

                ui.label("Hardware Launch").classes("text-sm font-semibold text-slate-600")
                for robot, (label, desc) in _HARDWARE_STACKS.items():
                    blocked_reason = None
                    stack_status = bridge.hardware_stack_status(robot)
                    if any_gazebo_running and stack_status.get("overall") != "running":
                        blocked_reason = "Blocked: Gazebo is running. Stop Gazebo first."
                    _hardware_stack_row(bridge, robot, label, desc, stack_status, blocked_reason=blocked_reason)

                ui.separator().classes("my-2")

                # Support processes.
                ui.label("Support Services").classes("text-sm font-semibold text-slate-600")
                for name, (label, desc) in _SUPPORT_PROCS.items():
                    _proc_row(bridge, name, label, desc, statuses.get(name, "stopped"))

                ui.separator().classes("my-2")

                # Utility buttons.
                with ui.row().classes("gap-4"):
                    async def _stop_all_async():
                        await asyncio.to_thread(bridge.ros2_stop_all)
                        ui.notify("Stopped all tracked processes", type="info")
                        _refresh(force=True)

                    def _stop_all():
                        asyncio.create_task(_stop_all_async())

                    async def _reset_gazebo_async() -> None:
                        ok, msg = await asyncio.to_thread(bridge.ros2_reset_gazebo_environment)
                        ui.notify(msg, type=("positive" if ok else "warning"), timeout=4500)
                        _refresh(force=True)

                    def _reset_gazebo():
                        asyncio.create_task(_reset_gazebo_async())

                    async def _cleanup_async():
                        await asyncio.to_thread(bridge.ros2_cleanup_processes)
                        ui.notify("Cleanup complete: removed stale ROS2/MoveIt/driver processes", type="info")
                        _refresh(force=True)

                    def _cleanup():
                        asyncio.create_task(_cleanup_async())

                    ui.button("Stop All", on_click=_stop_all, icon="stop_circle").props("flat dense").classes("text-red-600")
                    ui.button("Reset Gazebo Scene", on_click=_reset_gazebo, icon="restart_alt").props("flat dense").classes("text-blue-700")
                    ui.button("Cleanup", on_click=_cleanup, icon="cleaning_services").props("flat dense").classes("text-amber-700")

        _refresh(force=True)
        ui.timer(3.0, _refresh)


def _proc_row(
    bridge: SystemBridge,
    name: str,
    label: str,
    desc: str,
    status: str,
    blocked_reason: str | None = None,
) -> None:
    with ui.row().classes("items-center gap-4 w-full"):
        color = "green" if status == "running" else "grey"
        ui.icon("circle", color=color).classes("text-xs")

        with ui.column().classes("gap-0 flex-1"):
            ui.label(label).classes("font-semibold text-sm")
            ui.label(desc).classes("text-xs text-slate-400")

        is_running = status == "running"
        start_blocked = bool(blocked_reason)

        async def _start_async(n=name):
            err = await asyncio.to_thread(bridge.ros2_start, n)
            if err:
                ui.notify(err, type="warning")
            else:
                ui.notify(f"Started {label}", type="positive")
            _refresh(force=True)

        async def _stop_async(n=name):
            await asyncio.to_thread(bridge.ros2_stop, n)
            ui.notify(f"Stopped {label}", type="info")
            _refresh(force=True)

        def _start(n=name):
            if blocked_reason:
                ui.notify(blocked_reason, type="warning")
                return
            asyncio.create_task(_start_async(n))

        def _stop(n=name):
            asyncio.create_task(_stop_async(n))

        ui.button("Start", on_click=_start, icon="play_arrow").props(
            "flat dense" + (" disable" if (is_running or start_blocked) else "")
        ).classes("text-green-600")
        ui.button("Stop", on_click=_stop, icon="stop").props(
            "flat dense" + (" disable" if not is_running else "")
        ).classes("text-red-600")


def _hardware_stack_row(
    bridge: SystemBridge,
    robot: str,
    label: str,
    desc: str,
    status: dict,
    blocked_reason: str | None = None,
) -> None:
    overall = str(status.get("overall", "stopped"))
    color = "green" if overall == "running" else ("orange" if overall == "partial" else "grey")

    with ui.row().classes("items-center gap-4 w-full"):
        ui.icon("circle", color=color).classes("text-xs")

        with ui.column().classes("gap-0 flex-1"):
            ui.label(label).classes("font-semibold text-sm")
            ui.label(desc).classes("text-xs text-slate-400")
            ui.label(_hardware_status_text(status)).classes("text-xs text-slate-500")
            if robot == "ur5e":
                _render_ur5e_external_control_and_rg2_status(status)

        start_blocked = bool(blocked_reason) or overall == "running"
        stop_disabled = overall == "stopped"

        async def _start_async() -> None:
            err = await asyncio.to_thread(bridge.ros2_start_hardware_stack, robot)
            if err:
                ui.notify(err, type="warning", timeout=5000)
                return
            ui.notify(f"Started {label}", type="positive")

        async def _stop_async() -> None:
            err = await asyncio.to_thread(bridge.ros2_stop_hardware_stack, robot)
            if err:
                ui.notify(err, type="warning", timeout=3000)
                return
            ui.notify(f"Stopped {label}", type="info")

        def _start() -> None:
            if blocked_reason:
                ui.notify(blocked_reason, type="warning", timeout=3500)
                return
            asyncio.create_task(_start_async())

        def _stop() -> None:
            asyncio.create_task(_stop_async())

        ui.button("Start", on_click=_start, icon="play_arrow").props(
            "flat dense" + (" disable" if start_blocked else "")
        ).classes("text-green-600")
        ui.button("Stop", on_click=_stop, icon="stop").props(
            "flat dense" + (" disable" if stop_disabled else "")
        ).classes("text-red-600")


# =====================================================================
# Digital Twin Launch Section
# =====================================================================
# Remembers whether each target's Record/Replay expansion is open, so a section
# rebuild (from the 3 s status timer) reopens it where the operator left it.
_DT_RECORD_OPEN: dict[str, bool] = {}

# Friendly display labels for the internal sim-mode keys.
_DT_MODE_LABELS = {"monitor": "Monitor", "teach": "Teach"}


def _digital_twin_launch_section(bridge: SystemBridge) -> None:
    with ui.card().classes("w-full"):
        ui.label("digital twin launch").classes("text-lg font-semibold mb-1")
        ui.label(
            "Monitor = sim mirrors the live robot (hardware drives gazebo). "
            "Teach = build motions in the sim MoveIt/RViz, capture waypoints, then replay on "
            "gazebo + hardware together."
        ).classes("text-xs text-slate-500 mb-3")

        container = ui.column().classes("w-full gap-3")
        refresh_state = {"signature": None, "busy": False}

        def _signature() -> tuple:
            rows = bridge.digital_twin_statuses()
            compact_rows = []
            for target, row in rows.items():
                hardware = dict(row.get("hardware") or {})
                compact_rows.append(
                    (
                        target,
                        bool(row.get("supported", False)),
                        str(row.get("blocked_reason", "") or ""),
                        dict(row.get("gazebo") or {}).get("status", "stopped"),
                        dict(row.get("moviet") or {}).get("status", "stopped"),
                        dict(row.get("domains") or {}).get("gazebo"),
                        dict(row.get("domains") or {}).get("hardware"),
                        str(row.get("direction", "")),
                        str(row.get("sim_mode", "")),
                        hardware.get("overall", "unknown"),
                        repr(dict(hardware.get("status") or {})),
                        repr(dict(hardware.get("rg2") or {})),
                        dict(row.get("sync/status") or {}).get("state", "unknown"),
                        dict(row.get("sync/status") or {}).get("process_status", "unknown"),
                        # status_age_ms / latency_ms are intentionally excluded: they change every
                        # cycle and would force a rebuild (collapsing open expansions) 3 s apart.
                        dict(row.get("sync/status") or {}).get("last_error"),
                    )
                )
            return tuple(compact_rows)

        def _refresh(*, force: bool = False) -> None:
            if not _client_alive(container):
                return
            if refresh_state["busy"]:
                return
            refresh_state["busy"] = True
            try:
                signature = _signature()
                if not force and signature == refresh_state["signature"]:
                    return
                refresh_state["signature"] = signature
                rows = bridge.digital_twin_statuses()
            finally:
                refresh_state["busy"] = False

            container.clear()
            with container:
                with ui.row().classes("w-full items-center gap-3 px-2 py-1 bg-slate-50 rounded text-xs font-semibold text-slate-600"):
                    ui.label("target").classes("w-44")
                    ui.label("gazebo").classes("w-48")
                    ui.label("moviet").classes("w-36")
                    ui.label("hardware").classes("w-52")
                    ui.label("sync/status").classes("flex-1 min-w-64")
                for target, row in rows.items():
                    _digital_twin_row(bridge, target, row, _refresh)

        _refresh(force=True)
        ui.timer(3.0, _refresh)


def _digital_twin_status_color(status: str) -> str:
    value = str(status or "").strip().lower()
    if value in {"running", "mirroring", "applied", "teach"}:
        return "green"
    if value in {"partial", "limited", "paused", "waiting", "starting", "stale"}:
        return "orange"
    if value in {"unsupported", "unknown", "mixed", "error", "blocked"}:
        return "red"
    if value in {"gazebo", "hardware", "ready", "stopped"}:
        return "blue"
    return "grey"


def _digital_twin_badge(status: str) -> None:
    ui.badge(str(status or "unknown"), color=_digital_twin_status_color(status)).classes("text-xs")


def _digital_twin_row(bridge: SystemBridge, target: str, row: dict, refresh_callback) -> None:
    gazebo = dict(row.get("gazebo") or {})
    moviet = dict(row.get("moviet") or {})
    hardware = dict(row.get("hardware") or {})
    sync = dict(row.get("sync/status") or {})
    domains = dict(row.get("domains") or {})

    supported = bool(row.get("supported", False))
    blocked_reason = str(row.get("blocked_reason", "") or "").strip()
    gazebo_status = str(gazebo.get("status", "stopped"))
    hardware_status = dict(hardware.get("status") or {})
    hardware_overall = str(hardware.get("overall", "unknown"))
    sync_state = str(sync.get("state", "unknown"))
    sync_process_status = str(sync.get("process_status", "unknown"))
    sync_message = str(sync.get("message", "") or "").strip()
    sim_mode = str(row.get("sim_mode", "monitor") or "monitor")
    sim_modes = list(row.get("sim_modes") or [])
    is_running = (
        gazebo_status == "running"
        or hardware_overall in {"running", "partial"}
        or sync_process_status == "running"
    )

    with ui.row().classes("w-full items-stretch gap-3 border-b border-slate-100 px-2 py-3 flex-wrap"):
        with ui.column().classes("w-44 gap-2"):
            ui.label(target).classes("font-semibold text-sm")
            ui.label("digital twin").classes("text-xs text-slate-400")
            if blocked_reason:
                ui.label(blocked_reason).classes("text-xs text-amber-700")

            async def _start_twin_async() -> None:
                try:
                    err = await asyncio.to_thread(bridge.digital_twin_start, target)
                    if err:
                        ui.notify(err, type="warning", timeout=5000)
                    else:
                        ui.notify(f"Started {target} digital twin", type="positive")
                finally:
                    refresh_callback(force=True)

            async def _stop_twin_async() -> None:
                try:
                    err = await asyncio.to_thread(bridge.digital_twin_stop, target)
                    if err:
                        ui.notify(err, type="warning", timeout=3000)
                    else:
                        ui.notify(f"Stopped {target} digital twin", type="info")
                finally:
                    refresh_callback(force=True)

            def _start_twin() -> None:
                if blocked_reason:
                    ui.notify(blocked_reason, type="warning", timeout=4500)
                    return
                asyncio.create_task(_start_twin_async())

            def _stop_twin() -> None:
                asyncio.create_task(_stop_twin_async())

            retry_sync = is_running and sync_process_status != "running"
            start_disabled = (not supported) or (is_running and not retry_sync) or bool(blocked_reason)
            stop_disabled = (not supported) or not is_running
            with ui.row().classes("gap-1"):
                ui.button("Start Twin", on_click=_start_twin, icon="play_arrow").props(
                    "flat dense" + (" disable" if start_disabled else "")
                ).classes("text-green-600")
                ui.button("Stop Twin", on_click=_stop_twin, icon="stop").props(
                    "flat dense" + (" disable" if stop_disabled else "")
                ).classes("text-red-600")

            if supported and sim_modes:
                def _handle_sim_mode_change(e) -> None:
                    err = bridge.digital_twin_set_sim_mode(target, str(e.value or ""))
                    if err:
                        ui.notify(err, type="warning", timeout=3500)
                    refresh_callback(force=True)

                sim_mode_select = ui.select(
                    {m: _DT_MODE_LABELS.get(m, m) for m in sim_modes},
                    value=sim_mode,
                    label="mode",
                    on_change=_handle_sim_mode_change,
                ).props("dense").classes("w-40")
                if is_running:
                    sim_mode_select.props("disable")
                ui.label(
                    "Teach = build motions in sim MoveIt/RViz, then Capture / Replay (set before Start)"
                    if sim_mode == "teach"
                    else "Monitor = sim mirrors the live robot"
                ).classes("text-xs text-slate-400")

        with ui.column().classes("w-48 gap-1"):
            with ui.row().classes("items-center gap-2"):
                _digital_twin_badge(gazebo_status)
                ui.label(str(gazebo.get("name", ""))).classes("text-xs text-slate-500")
            ui.label(str(gazebo.get("message", ""))).classes("text-xs text-slate-500")
            ui.label(f"process: {str(gazebo.get('process', ''))}").classes("text-xs text-slate-500")
            ui.label(f"ROS_DOMAIN_ID={domains.get('gazebo', '')}").classes("text-xs text-slate-500")

        with ui.column().classes("w-36 gap-1"):
            _digital_twin_badge(str(moviet.get("status", "stopped")))
            ui.label(str(moviet.get("message", ""))).classes("text-xs text-slate-500")

        with ui.column().classes("w-52 gap-1"):
            with ui.row().classes("items-center gap-2"):
                _digital_twin_badge(hardware_overall)
                ui.label("hardware").classes("text-xs text-slate-500")
            if supported:
                ui.label(_hardware_status_text(hardware_status)).classes("text-xs text-slate-500")
                _render_ur5e_external_control_and_rg2_status(hardware_status)
                hardware_domains = dict(hardware.get("domains") or {})
                if hardware_domains:
                    domain_text = " | ".join(
                        f"{robot}: ROS_DOMAIN_ID={domain}"
                        for robot, domain in hardware_domains.items()
                    )
                    ui.label(domain_text).classes("text-xs text-slate-500")
                else:
                    ui.label(f"ROS_DOMAIN_ID={domains.get('hardware', '')}").classes("text-xs text-slate-500")
                rg2 = dict(hardware.get("rg2") or {})
                if rg2:
                    state = str(rg2.get("state") or "unknown")
                    width = rg2.get("width_mm")
                    source = str(rg2.get("source") or "unknown")
                    error = str(rg2.get("error") or "").strip()
                    if width is not None:
                        msg = f"RG2 last {source}: {state}, width {float(width):.1f} mm"
                    else:
                        msg = f"RG2 last {source}: {state}"
                    ui.label(msg).classes("text-xs text-slate-500")
                    if error:
                        ui.label(f"RG2 error: {error}").classes("text-xs text-red-700")
            else:
                ui.label(str(hardware.get("message", ""))).classes("text-xs text-amber-700")

        with ui.column().classes("flex-1 min-w-64 gap-1"):
            with ui.row().classes("items-center gap-2"):
                _digital_twin_badge(sync_state)
                ui.label("sync/status").classes("text-xs text-slate-500")
            ui.label(sync_message).classes("text-xs text-slate-500")
            ui.label(f"sync process: {sync_process_status}").classes("text-xs text-slate-500")

            status_bits = []
            status_age_ms = sync.get("status_age_ms")
            latency_ms = sync.get("latency_ms")
            max_joint_delta_deg = sync.get("max_joint_delta_deg")
            if status_age_ms is not None:
                status_bits.append(f"freshness {float(status_age_ms):.0f} ms")
            if latency_ms is not None:
                status_bits.append(f"latency {float(latency_ms):.1f} ms")
            if max_joint_delta_deg is not None:
                status_bits.append(f"max delta {float(max_joint_delta_deg):.2f} deg")
            if status_bits:
                ui.label(" | ".join(status_bits)).classes("text-xs text-slate-500")
            if sync.get("last_error"):
                ui.label(str(sync.get("last_error"))).classes("text-xs text-red-700")

            # Direction is implied by mode (Monitor = hardware→gazebo, Teach =
            # gazebo→hardware), so there is no direction control.
            if supported and sim_mode == "teach":
                _digital_twin_record_replay(bridge, target)


def _digital_twin_record_replay(bridge: SystemBridge, target: str) -> None:
    """Manual record-in-gazebo / replay-on-hardware panel for one target."""
    is_dual = target == "dual robots"
    expansion = ui.expansion(
        "Record / Replay (gazebo → hardware)",
        icon="fiber_manual_record",
        value=_DT_RECORD_OPEN.get(target, False),
    ).classes("w-full text-xs")
    expansion.on_value_change(lambda e: _DT_RECORD_OPEN.__setitem__(target, bool(e.value)))
    with expansion:
        ui.label(
            "Teach: sim RViz controls Gazebo only. Plan & Execute in the sim RViz window, "
            "Capture Waypoint at each validated pose, then Replay in Twin commits the saved "
            "sim waypoint through hardware MoveIt. Save is optional (keeps a named copy)."
        ).classes("text-xs text-slate-500 mb-1")

        count_label = ui.label(f"waypoints: {bridge.digital_twin_waypoint_count(target)}").classes(
            "text-xs font-semibold"
        )
        waypoints_container = ui.column().classes("w-full gap-0 mb-1")
        replay_source_label = ui.label("").classes("text-xs text-blue-700 font-semibold")
        prepared_label = (
            ui.label("prepared: none").classes("text-xs text-emerald-700")
            if is_dual
            else None
        )
        prepare_state: dict[str, object] = {
            "generation": 0,
            "task": None,
        }

        def _replay_source() -> tuple[str, str]:
            """Return (kind, description) of what Replay/Preview will use right now."""
            n = bridge.digital_twin_waypoint_count(target)
            if n > 0:
                return "buffer", f"current capture ({n} waypoints)"
            sel = str(recordings_select.value or "").strip()
            if sel:
                return "saved", f"saved '{sel}'"
            return "none", "nothing — capture or select a recording"

        def _refresh_replay_source() -> None:
            _kind, desc = _replay_source()
            replay_source_label.text = f"▶ Replay will use: {desc}"

        def _set_prepared_status(text: str, *, failed: bool = False) -> None:
            if prepared_label is None:
                return
            prepared_label.text = text
            prepared_label.classes(
                replace="text-xs text-red-700" if failed else "text-xs text-emerald-700"
            )

        def _mark_prepare_stale(reason: str = "") -> None:
            if not is_dual:
                return
            prepare_state["generation"] = int(prepare_state.get("generation") or 0) + 1
            suffix = f": {reason}" if reason else ""
            _set_prepared_status(f"prepared: stale{suffix}")

        async def _prepare_current_replay(generation: int, *, client: Client | None = None) -> dict[str, object]:
            try:
                kind, _desc = _replay_source()
                if kind == "buffer":
                    result = await asyncio.to_thread(
                        bridge.digital_twin_prepare_replay_buffer,
                        target,
                        replay_target="twin",
                    )
                elif kind == "saved":
                    name = str(recordings_select.value or "").strip()
                    result = await asyncio.to_thread(
                        bridge.digital_twin_prepare_replay,
                        target,
                        name,
                        replay_target="twin",
                    )
                else:
                    result = {"success": False, "message": "capture or select a recording"}
                if generation != int(prepare_state.get("generation") or 0):
                    return result
                if result.get("success"):
                    _set_prepared_status("prepared: ready")
                else:
                    _set_prepared_status(
                        f"prepared: failed: {str(result.get('message') or '')}",
                        failed=True,
                    )
                return result
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin replay preparation failed for %s", target)
                if generation == int(prepare_state.get("generation") or 0):
                    _set_prepared_status(f"prepared: failed: {exc}", failed=True)
                return {"success": False, "message": str(exc)}

        def _start_prepare_background() -> None:
            if not is_dual:
                return
            kind, _desc = _replay_source()
            if kind == "none":
                _set_prepared_status("prepared: none")
                return
            generation = int(prepare_state.get("generation") or 0) + 1
            prepare_state["generation"] = generation
            _set_prepared_status("prepared: preparing")
            prepare_state["task"] = asyncio.create_task(_prepare_current_replay(generation))

        async def _wait_for_prepare_if_running(*, client: Client | None = None) -> None:
            if not is_dual:
                return
            task = prepare_state.get("task")
            if task is not None and not task.done():
                _notify("Preparing replay…", type="ongoing", timeout=1500, client=client)
                await task

        def _current_client() -> Client | None:
            try:
                return context.client
            except RuntimeError:
                return None

        def _notify(
            message: object,
            *,
            type: str | None = None,
            timeout: int = 3500,
            client: Client | None = None,
        ) -> None:
            if client is not None:
                options: dict[str, object] = {"message": str(message), "timeout": timeout}
                if type is not None:
                    options["type"] = type
                try:
                    client.outbox.enqueue_message("notify", options, client.id)
                    return
                except Exception:
                    log.exception("failed to notify captured NiceGUI client")
            try:
                ui.notify(str(message), type=type, timeout=timeout)
            except RuntimeError:
                log.warning("could not notify user because NiceGUI slot was deleted: %s", message)

        def _refresh_count() -> None:
            count_label.text = f"waypoints: {bridge.digital_twin_waypoint_count(target)}"
            _refresh_replay_source()
            _refresh_waypoints()

        def _refresh_waypoints() -> None:
            waypoints = bridge.digital_twin_list_waypoints(target)
            waypoints_container.clear()
            with waypoints_container:
                if not waypoints:
                    ui.label("no waypoints captured yet").classes("text-xs text-slate-400")
                    return
                for wp in waypoints:
                    idx = int(wp["index"])
                    robots = dict(wp.get("robots") or {})
                    if robots:
                        chunks = []
                        for robot in ("xarm6", "ur5e"):
                            body = dict(robots.get(robot) or {})
                            positions = ", ".join(f"{p:.2f}" for p in (body.get("positions") or []))
                            chunks.append(f"{robot} [{positions}]")
                        joints = " | ".join(chunks)
                    else:
                        joints = ", ".join(f"{p:.2f}" for p in (wp.get("positions") or []))
                    with ui.row().classes("items-center gap-1 w-full"):
                        ui.label(f"#{idx + 1}").classes("text-xs font-semibold w-8")
                        ui.label(f"[{joints}]").classes("text-xs text-slate-500 flex-1 truncate")
                        ui.button(icon="arrow_upward", on_click=lambda _e, i=idx: _move_wp(i, -1)).props("flat dense round size=sm")
                        ui.button(icon="arrow_downward", on_click=lambda _e, i=idx: _move_wp(i, 1)).props("flat dense round size=sm")
                        ui.button(icon="my_location", on_click=lambda _e, i=idx: _overwrite_wp(i)).props("flat dense round size=sm").tooltip("overwrite with current sim pose")
                        ui.button(icon="close", on_click=lambda _e, i=idx: _delete_wp(i)).props("flat dense round size=sm").classes("text-red-600")

        def _move_wp(index: int, delta: int) -> None:
            try:
                bridge.digital_twin_move_waypoint(target, index, delta)
                _refresh_count()
                _mark_prepare_stale("waypoints changed")
                _start_prepare_background()
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin move waypoint failed")
                ui.notify(f"Move failed: {exc}", type="negative", timeout=6000)

        def _delete_wp(index: int) -> None:
            try:
                bridge.digital_twin_delete_waypoint(target, index)
                _refresh_count()
                _mark_prepare_stale("waypoints changed")
                _start_prepare_background()
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin delete waypoint failed")
                ui.notify(f"Delete failed: {exc}", type="negative", timeout=6000)

        async def _overwrite_wp(index: int) -> None:
            try:
                ui.notify("Updating waypoint to current sim pose…", type="ongoing", timeout=1500)
                result = await asyncio.to_thread(bridge.digital_twin_overwrite_waypoint, target, index)
                ui.notify(
                    str(result.get("message") or ""),
                    type="positive" if result.get("success") else "warning",
                    timeout=3500,
                )
                _refresh_count()
                _mark_prepare_stale("waypoints changed")
                _start_prepare_background()
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin overwrite waypoint failed")
                ui.notify(f"Overwrite failed: {exc}", type="negative", timeout=6000)

        def _recording_changed(_e) -> None:
            _refresh_replay_source()
            _mark_prepare_stale("recording changed")
            _start_prepare_background()

        def _refresh_recordings() -> None:
            options = bridge.digital_twin_list_recordings(target)
            recordings_select.options = options
            if recordings_select.value not in options:
                recordings_select.value = options[0] if options else None
            recordings_select.update()
            _refresh_replay_source()

        recordings_select = ui.select(
            [], label="saved recording", on_change=_recording_changed
        ).props("dense").classes("w-56")

        async def _capture() -> None:
            try:
                # Immediate feedback — the snapshot subprocess takes ~1-3s.
                ui.notify("Capturing…", type="ongoing", timeout=1500)
                count_label.text = "waypoints: capturing…"
                result = await asyncio.to_thread(bridge.digital_twin_capture_waypoint, target)
                ui.notify(
                    str(result.get("message") or ""),
                    type="positive" if result.get("success") else "warning",
                    timeout=3500,
                )
                _refresh_count()
                _mark_prepare_stale("waypoints changed")
                _start_prepare_background()
            except Exception as exc:  # noqa: BLE001 - surface any failure to the operator
                log.exception("digital twin capture failed for %s", target)
                ui.notify(f"Capture failed: {exc}", type="negative", timeout=6000)
                _refresh_count()

        def _clear() -> None:
            try:
                bridge.digital_twin_clear_waypoints(target)
                _refresh_count()
                _mark_prepare_stale("waypoints cleared")
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin clear failed for %s", target)
                ui.notify(f"Clear failed: {exc}", type="negative", timeout=6000)

        async def _delete_recording() -> None:
            try:
                name = str(recordings_select.value or "").strip()
                if not name:
                    ui.notify("Select a saved recording to delete.", type="warning")
                    return
                result = await asyncio.to_thread(bridge.digital_twin_delete_recording, target, name)
                ui.notify(
                    str(result.get("message") or ""),
                    type="positive" if result.get("success") else "warning",
                    timeout=3500,
                )
                if result.get("success"):
                    _refresh_recordings()
                    _mark_prepare_stale("recording deleted")
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin delete recording failed for %s", target)
                ui.notify(f"Delete failed: {exc}", type="negative", timeout=6000)

        async def _save() -> None:
            try:
                name = str(name_input.value or "").strip()
                if not name:
                    ui.notify("Enter a recording name.", type="warning")
                    return
                result = await asyncio.to_thread(bridge.digital_twin_save_recording, target, name)
                ui.notify(
                    str(result.get("message") or ""),
                    type="positive" if result.get("success") else "warning",
                    timeout=3500,
                )
                if result.get("success"):
                    _refresh_recordings()
                    _mark_prepare_stale("recording saved")
                    _start_prepare_background()
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin save failed for %s", target)
                ui.notify(f"Save failed: {exc}", type="negative", timeout=6000)

        async def _replay(replay_target: str, *, client: Client | None = None) -> dict[str, object]:
            notify_client = client or _current_client()
            try:
                # Prefer the just-captured buffer (no Save needed); fall back to a saved recording.
                kind, desc = _replay_source()
                if kind == "buffer":
                    if replay_target == "twin":
                        await _wait_for_prepare_if_running(client=notify_client)
                    result = await asyncio.to_thread(
                        bridge.digital_twin_replay_buffer, target, replay_target=replay_target
                    )
                elif kind == "saved":
                    name = str(recordings_select.value or "").strip()
                    if replay_target == "twin":
                        await _wait_for_prepare_if_running(client=notify_client)
                    result = await asyncio.to_thread(
                        bridge.digital_twin_replay, target, name, replay_target=replay_target
                    )
                else:
                    _notify(
                        "Capture waypoints (or select a saved recording) first.",
                        type="warning",
                        client=notify_client,
                    )
                    return {"success": False, "message": "Capture waypoints (or select a saved recording) first."}
                _notify(
                    f"{desc} → {str(result.get('message') or '')}",
                    type="positive" if result.get("success") else "warning",
                    timeout=5000,
                    client=notify_client,
                )
                return dict(result)
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin replay failed for %s", target)
                _notify(f"Replay failed: {exc}", type="negative", timeout=6000, client=notify_client)
                return {"success": False, "message": str(exc)}

        async def _preview_gazebo() -> None:
            result = await _replay("gazebo")
            if result.get("success"):
                _start_prepare_background()

        async def _go_home(replay_target: str) -> None:
            try:
                ui.notify("Going home…", type="ongoing", timeout=1500)
                result = await asyncio.to_thread(
                    bridge.digital_twin_go_home, target, replay_target=replay_target
                )
                ui.notify(
                    str(result.get("message") or "home"),
                    type="positive" if result.get("success") else "warning",
                    timeout=5000,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("digital twin go home failed for %s", target)
                ui.notify(f"Go Home failed: {exc}", type="negative", timeout=6000)

        async def _home_sim() -> None:
            await _go_home("gazebo")

        home_confirm = None
        if not is_dual:
            with ui.dialog() as home_dialog, ui.card().classes("gap-3"):
                home_confirm = home_dialog
                ui.label("Send the real robot home?").classes("font-semibold")
                ui.label(
                    "This moves the real robot AND the gazebo model to the initial/home pose. "
                    "The approach is speed-limited for safety."
                ).classes("text-sm text-slate-600")
                with ui.row().classes("justify-end gap-2 w-full"):
                    ui.button("Cancel", on_click=home_dialog.close).props("flat")

                    async def _home_twin_confirmed() -> None:
                        home_dialog.close()
                        await _go_home("twin")

                    ui.button("Go Home", on_click=_home_twin_confirmed, icon="home").props("color=red")

        with ui.row().classes("items-center gap-2 mt-1"):
            ui.button("Capture Waypoint", on_click=_capture, icon="add_location").props(
                "flat dense"
            )
            ui.button("Clear", on_click=_clear, icon="delete").props("flat dense").classes("text-red-600")
            if not is_dual and home_confirm is not None:
                ui.button("Home (sim)", on_click=_home_sim, icon="home").props("flat dense")
                ui.button("Go Home (twin)", on_click=home_confirm.open, icon="home").props(
                    "flat dense"
                ).classes("text-red-600")
        with ui.row().classes("items-center gap-2"):
            name_input = ui.input(label="save as").props("dense").classes("w-40")
            ui.button("Save", on_click=_save, icon="save").props("flat dense")
            ui.button("Delete", on_click=_delete_recording, icon="delete_forever").props(
                "flat dense"
            ).classes("text-red-600")
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Preview in Gazebo",
                on_click=_preview_gazebo,
                icon="visibility",
            ).props("flat dense")

            with ui.dialog() as replay_confirm, ui.card().classes("gap-3"):
                ui.label("Replay in Twin: commit through hardware MoveIt?").classes("font-semibold")
                ui.label(
                    "This commits the saved sim waypoint through hardware MoveIt, keeps grippers "
                    "on their working hardware paths, and resumes hardware -> Gazebo mirroring."
                ).classes("text-sm text-slate-600")
                with ui.row().classes("justify-end gap-2 w-full"):
                    ui.button("Cancel", on_click=replay_confirm.close).props("flat")

                    async def _replay_twin_confirmed() -> None:
                        notify_client = _current_client()
                        replay_confirm.close()
                        await _replay("twin", client=notify_client)

                    ui.button(
                        "Replay",
                        on_click=_replay_twin_confirmed,
                        icon="send",
                    ).props("color=red")

            ui.button(
                "Replay in Twin (commit through hardware MoveIt)",
                on_click=replay_confirm.open,
                icon="precision_manufacturing",
            ).props("outline dense").classes("text-red-600")

        _refresh_recordings()
        _refresh_waypoints()


# =====================================================================
# Teleop Section
# =====================================================================
_AXIS_KEYS = {"x": ("ArrowRight", "ArrowLeft"), "y": ("ArrowUp", "ArrowDown"), "z": ("PageUp", "PageDown")}
_TELEOP_PROFILE_MULTIPLIER = {
    "precision": 0.35,
    "fast": 1.0,
}
_TELEOP_PROFILE_STEPS = {
    "precision": {"cartesian_mm": 2.0, "joint_deg": 0.2},
    "fast": {"cartesian_mm": 10.0, "joint_deg": 2.0},
}


def _teleop_section(bridge: SystemBridge) -> None:
    with ui.card().classes("w-full"):
        ui.label("Interactive Teleop").classes("text-lg font-semibold mb-1")
        ui.label(
            "Control robots via on-screen buttons or keyboard. "
            "Works in both Simulation and Physical modes when the environment is running."
        ).classes("text-xs text-slate-500 mb-3")

        with ui.row().classes("items-center gap-4 mb-2 flex-wrap"):
            with ui.row().classes("items-center gap-2"):
                teleop_backend_icon = ui.icon("circle", color="grey").classes("text-xs")
                teleop_backend_label = ui.label("Teleop backend: checking...").classes("text-xs")
            with ui.row().classes("items-center gap-2"):
                teleop_env_icon = ui.icon("circle", color="grey").classes("text-xs")
                teleop_env_label = ui.label("Teleop environment: checking...").classes("text-xs")
        teleop_warning_label = ui.label("").classes("text-xs text-amber-700")

        # Robot selector.
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.label("Robot:").classes("font-semibold text-sm")
            robot_select = ui.toggle(["xarm6", "ur5e"], value="xarm6").classes("text-sm")

        mode_state = {"mode": "cartesian"}   # cartesian | gripper | joint
        axis_state = {"axis": "y"}           # x | y | z
        joint_state = {"idx": 1}             # 1..6
        profile_state = {"mode": "fast"}     # precision | fast
        velocity_state = {
            "xarm6": {"arm_vel": 1.0, "gripper_vel": 1.0},
            "ur5e": {"arm_vel": 1.0, "gripper_vel": 1.0},
        }
        effective_labels = {}
        profile_note = {"label": None}
        step_inputs = {"cartesian": None, "joint": None}
        save_env_label = {"label": None}
        state_refresh = {"busy": False}
        state_labels = {"status": None, "xyz": None, "rpy": None, "joints": []}

        def _refresh_teleop_status() -> None:
            if not _client_alive(teleop_backend_label):
                return
            status = bridge.teleop_connection_status(str(robot_select.value or "xarm6"))
            connected = bool(status.get("connected", False))
            env = str(status.get("environment", "gazebo")).strip().lower()
            ros_domain_id = status.get("ros_domain_id")
            warning = str(status.get("warning") or "").strip()
            domain_text = f" | ROS_DOMAIN_ID={ros_domain_id}" if ros_domain_id is not None else ""

            teleop_backend_icon.props(f"color={'green' if connected else 'red'}")
            if connected:
                teleop_backend_label.set_text("Teleop backend: connected")
            else:
                teleop_backend_label.set_text(
                    "Teleop backend: disconnected (starts on first teleop command)"
                )

            if env == "real":
                teleop_env_icon.props("color=green")
                teleop_env_label.set_text(f"Teleop environment: hardware (real){domain_text}")
            elif env == "gazebo":
                teleop_env_icon.props("color=blue")
                teleop_env_label.set_text(f"Teleop environment: simulation (gazebo){domain_text}")
            else:
                teleop_env_icon.props("color=grey")
                teleop_env_label.set_text(f"Teleop environment: {env or 'unknown'}{domain_text}")

            teleop_warning_label.set_text(warning)

            target_label = save_env_label["label"]
            if target_label is not None:
                target_label.set_text(f'Save target: "{env or "gazebo"}" block in resource JSON')

        def _set_state_unavailable(reason: str) -> None:
            status_label = state_labels["status"]
            if status_label is not None:
                status_label.set_text(f"State: unavailable ({reason})")
            xyz_label = state_labels["xyz"]
            if xyz_label is not None:
                xyz_label.set_text("X/Y/Z [m]: -- / -- / --")
            rpy_label = state_labels["rpy"]
            if rpy_label is not None:
                rpy_label.set_text("Rx/Ry/Rz [deg]: -- / -- / --")
            for i, label in enumerate(state_labels["joints"], start=1):
                label.set_text(f"J{i}: --")

        def _fmt(value: float | None, digits: int = 3) -> str:
            try:
                return f"{float(value):+.{digits}f}"
            except Exception:
                return "--"

        async def _refresh_teleop_state_async() -> None:
            status_label = state_labels["status"]
            if status_label is None or not _client_alive(status_label):
                return
            if state_refresh["busy"]:
                return
            state_refresh["busy"] = True

            try:
                # Avoid spinning up teleop backend when no environment is running.
                env_running = await asyncio.to_thread(bridge.teleop_environment_running)
                if not env_running:
                    _set_state_unavailable("no environment running")
                    return

                teleop_status = await asyncio.to_thread(
                    bridge.teleop_connection_status,
                    str(robot_select.value or "xarm6"),
                )
                if not bool(teleop_status.get("connected", False)):
                    _set_state_unavailable("teleop backend disconnected")
                    return

                robot = str(robot_select.value or "xarm6")
                ok, msg, state = await asyncio.to_thread(bridge.teleop_state, robot)
                if not _client_alive(status_label):
                    return
                if not ok:
                    _set_state_unavailable(msg)
                    return

                status_label.set_text(f"State: {robot.upper()} live")

                pos = state.get("position", {})
                xyz_label = state_labels["xyz"]
                if xyz_label is not None:
                    xyz_label.set_text(
                        f"X/Y/Z [m]: {_fmt(pos.get('x'))} / {_fmt(pos.get('y'))} / {_fmt(pos.get('z'))}"
                    )

                ori_deg = state.get("orientation_deg")
                if not isinstance(ori_deg, dict):
                    ori_rad = state.get("orientation_rad", {})
                    ori_deg = {
                        "rx": math.degrees(float(ori_rad.get("rx", 0.0))),
                        "ry": math.degrees(float(ori_rad.get("ry", 0.0))),
                        "rz": math.degrees(float(ori_rad.get("rz", 0.0))),
                    }
                rpy_label = state_labels["rpy"]
                if rpy_label is not None:
                    rpy_label.set_text(
                        f"Rx/Ry/Rz [deg]: {_fmt(ori_deg.get('rx'), 2)} / {_fmt(ori_deg.get('ry'), 2)} / {_fmt(ori_deg.get('rz'), 2)}"
                    )

                joints = state.get("joints_deg")
                if not isinstance(joints, list):
                    joints_rad = state.get("joints_rad", [])
                    joints = [math.degrees(float(v)) for v in joints_rad]
                for i, label in enumerate(state_labels["joints"], start=1):
                    val = joints[i - 1] if i - 1 < len(joints) else None
                    label.set_text(f"J{i}: {_fmt(val, 2)}")
            finally:
                state_refresh["busy"] = False

        def _set_velocity(robot: str, key: str, value: float) -> None:
            try:
                velocity_state[robot][key] = float(value)
            except Exception:
                return
            _refresh_effective_labels()

        def _velocity(robot: str, key: str, default: float) -> float:
            try:
                return float(velocity_state.get(robot, {}).get(key, default))
            except Exception:
                return default

        def _effective_velocity(robot: str, key: str, default: float) -> float:
            base = _velocity(robot, key, default)
            multiplier = _TELEOP_PROFILE_MULTIPLIER.get(profile_state["mode"], 1.0)
            return base * multiplier

        def _apply_profile_steps(mode: str) -> None:
            settings = _TELEOP_PROFILE_STEPS.get(mode)
            if not settings:
                return
            cart_input = step_inputs["cartesian"]
            joint_input = step_inputs["joint"]
            if cart_input is not None:
                cart_input.value = float(settings["cartesian_mm"])
            if joint_input is not None:
                joint_input.value = float(settings["joint_deg"])

        def _refresh_effective_labels() -> None:
            multiplier = _TELEOP_PROFILE_MULTIPLIER.get(profile_state["mode"], 1.0)
            note_label = profile_note["label"]
            if note_label is not None:
                note_label.set_text(
                    f'{profile_state["mode"].title()} mode applies velocity x{multiplier:.2f} '
                    "and profile step presets."
                )
            for robot_name, label in effective_labels.items():
                arm = _effective_velocity(robot_name, "arm_vel", 1.0)
                grip = _effective_velocity(robot_name, "gripper_vel", 1.0)
                label.set_text(
                    f'Applied velocity: arm {arm:.2f}, gripper {grip:.2f} (x{multiplier:.2f})'
                )

        with ui.row().classes("items-center gap-3 mb-2"):
            ui.label("Teleop Profile:").classes("font-semibold text-sm")

            def _set_profile(value: str) -> None:
                mode = str(value).strip().lower()
                if mode not in _TELEOP_PROFILE_MULTIPLIER:
                    return
                changed = profile_state["mode"] != mode
                profile_state["mode"] = mode
                _apply_profile_steps(mode)
                _refresh_effective_labels()
                if changed:
                    ui.notify(f"{mode.title()} mode", type="info", position="bottom-right", timeout=900)

            profile_toggle = ui.toggle(
                ["Precision", "Fast"],
                value="Fast",
                on_change=lambda e: _set_profile(e.value),
            ).props("dense")
            profile_note["label"] = ui.label("").classes("text-xs text-slate-500")

        with ui.element("div").classes("w-full columns-1 lg:columns-2 xl:columns-3").style("column-gap: 1.5rem;"):
            # ── Cartesian Jog Pad ────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Cartesian Jog").classes("font-semibold text-sm mb-2")
                step_input = ui.number("Step (mm)", value=10.0, min=0.1, max=100.0, step=0.1).classes("w-32 mb-3")
                step_inputs["cartesian"] = step_input
                ui.label("Switching Precision/Fast also updates this step value.").classes("text-xs text-slate-500 mb-2")

                # X/Y pad (top-down view).
                ui.label("X / Y Axes").classes("text-xs text-slate-500 mb-1")
                with ui.column().classes("items-center gap-1"):
                    _jog_btn(bridge, robot_select, step_input, _effective_velocity, "Y+", "y", 1, "arrow_upward")
                    with ui.row().classes("gap-1"):
                        _jog_btn(bridge, robot_select, step_input, _effective_velocity, "X-", "x", -1, "arrow_back")
                        ui.button(icon="radio_button_unchecked").props("flat dense disable").classes("w-12 h-12")
                        _jog_btn(bridge, robot_select, step_input, _effective_velocity, "X+", "x", 1, "arrow_forward")
                    _jog_btn(bridge, robot_select, step_input, _effective_velocity, "Y-", "y", -1, "arrow_downward")

                # Z axis.
                ui.label("Z Axis").classes("text-xs text-slate-500 mt-3 mb-1")
                with ui.row().classes("gap-2 justify-center"):
                    _jog_btn(bridge, robot_select, step_input, _effective_velocity, "Z+", "z", 1, "expand_less")
                    _jog_btn(bridge, robot_select, step_input, _effective_velocity, "Z-", "z", -1, "expand_more")

            # ── Joint Jog ────────────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Joint Jog").classes("font-semibold text-sm mb-2")
                joint_step_input = ui.number("Step (deg)", value=2.0, min=0.05, max=30.0, step=0.05).classes("w-32 mb-2")
                step_inputs["joint"] = joint_step_input
                _apply_profile_steps(profile_state["mode"])
                selected_joint_label = ui.label("Selected: J1").classes("text-xs text-slate-500 mb-2")

                def _select_joint(idx: int):
                    joint_state["idx"] = idx
                    mode_state["mode"] = "joint"
                    selected_joint_label.set_text(f"Selected: J{idx}")
                    ui.notify(f"Joint mode: J{idx}", type="info", position="bottom-right", timeout=900)

                with ui.row().classes("gap-1 mb-2"):
                    for idx in range(1, 7):
                        ui.button(str(idx), on_click=lambda _=None, j=idx: _select_joint(j)).props("outline dense")

                with ui.row().classes("gap-2"):
                    def _joint_minus():
                        step_deg = joint_step_input.value or 2.0
                        arm_vel = _effective_velocity(robot_select.value, "arm_vel", 1.0)
                        asyncio.create_task(
                            _send_joint(bridge, robot_select.value, joint_state["idx"], -step_deg, arm_vel)
                        )

                    def _joint_plus():
                        step_deg = joint_step_input.value or 2.0
                        arm_vel = _effective_velocity(robot_select.value, "arm_vel", 1.0)
                        asyncio.create_task(
                            _send_joint(bridge, robot_select.value, joint_state["idx"], step_deg, arm_vel)
                        )

                    ui.button("-", on_click=_joint_minus, icon="remove").props("outline")
                    ui.button("+", on_click=_joint_plus, icon="add").props("outline")

            # ── Gripper Control ──────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Gripper").classes("font-semibold text-sm mb-2")

                with ui.row().classes("gap-2 justify-center"):
                    def _full_open():
                        mode_state["mode"] = "gripper"
                        grip_vel = _effective_velocity(robot_select.value, "gripper_vel", 1.0)
                        asyncio.create_task(_send_gripper(bridge, robot_select.value, "open", 1.0, grip_vel))

                    def _full_close():
                        mode_state["mode"] = "gripper"
                        grip_vel = _effective_velocity(robot_select.value, "gripper_vel", 1.0)
                        asyncio.create_task(_send_gripper(bridge, robot_select.value, "close", 1.0, grip_vel))

                    def _step_open():
                        mode_state["mode"] = "gripper"
                        grip_vel = _effective_velocity(robot_select.value, "gripper_vel", 1.0)
                        asyncio.create_task(_send_gripper(bridge, robot_select.value, "open", None, grip_vel))

                    def _step_close():
                        mode_state["mode"] = "gripper"
                        grip_vel = _effective_velocity(robot_select.value, "gripper_vel", 1.0)
                        asyncio.create_task(_send_gripper(bridge, robot_select.value, "close", None, grip_vel))

                    ui.button("Full Open", on_click=_full_open, icon="open_with").props("outline")
                    ui.button("Full Close", on_click=_full_close, icon="close_fullscreen").props("outline")
                with ui.row().classes("gap-2 justify-center"):
                    ui.button("Open", on_click=_step_open, icon="add").props("outline")
                    ui.button("Close", on_click=_step_close, icon="remove").props("outline")

                # Home button.
                ui.separator().classes("my-3")

                def _go_home():
                    asyncio.create_task(_send_home(bridge, robot_select.value))

                ui.button("Move Home", on_click=_go_home, icon="home").props("outline").classes("w-full")

            # ── Keyboard Help ────────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Keyboard Shortcuts").classes("font-semibold text-sm mb-2")
                _kb_help = [
                    ("T", "Switch robot"),
                    ("M", "Cartesian mode"),
                    ("P / F", "Precision / Fast profile (velocity + steps)"),
                    ("X / Y / Z", "Select Cartesian axis"),
                    ("Arrow Left / Right", "Cartesian - / + on X or Y"),
                    ("Arrow Up / Down", "Cartesian Z+ / Z- when axis=Z"),
                    ("Page Up / Down", "Cartesian Z+ / Z-"),
                    ("G", "Gripper mode (arrows)"),
                    ("Arrow Up / Down", "In G mode: Open / Close gripper"),
                    ("J", "Joint mode (arrows)"),
                    ("1..6", "Select joint in J mode"),
                    ("Arrow Up / Down", "In J mode: Joint + / -"),
                    ("H", "Move home"),
                    ("S", "Save current position (uses Name field)"),
                ]
                for key, action in _kb_help:
                    with ui.row().classes("gap-2"):
                        ui.label(key).classes("text-xs font-mono bg-slate-100 px-2 py-0.5 rounded")
                        ui.label(action).classes("text-xs text-slate-600")

            # ── Current Pose/Joints ──────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Current Position").classes("font-semibold text-sm mb-2")
                state_labels["status"] = ui.label("State: checking...").classes("text-xs text-slate-500 mb-1")
                state_labels["xyz"] = ui.label("X/Y/Z [m]: -- / -- / --").classes("text-xs font-mono")
                state_labels["rpy"] = ui.label("Rx/Ry/Rz [deg]: -- / -- / --").classes("text-xs font-mono mb-1")
                with ui.row().classes("gap-3"):
                    for idx in range(1, 4):
                        state_labels["joints"].append(ui.label(f"J{idx}: --").classes("text-xs font-mono"))
                with ui.row().classes("gap-3"):
                    for idx in range(4, 7):
                        state_labels["joints"].append(ui.label(f"J{idx}: --").classes("text-xs font-mono"))

            # ── Speeds ───────────────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Velocity Settings (Per Robot)").classes("font-semibold text-sm mb-2")

                def _velocity_block(robot: str) -> None:
                    with ui.column().classes("gap-1 mb-3"):
                        ui.label(robot).classes("text-xs font-semibold text-slate-600")
                        ui.number(
                            "Arm velocity scale",
                            value=velocity_state[robot]["arm_vel"],
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            on_change=lambda e, r=robot: _set_velocity(r, "arm_vel", e.value),
                        ).classes("w-44")
                        ui.number(
                            "Gripper velocity scale",
                            value=velocity_state[robot]["gripper_vel"],
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            on_change=lambda e, r=robot: _set_velocity(r, "gripper_vel", e.value),
                        ).classes("w-44")
                        effective_labels[robot] = ui.label("").classes("text-xs text-slate-500")

                with ui.row().classes("gap-6 flex-wrap"):
                    _velocity_block("xarm6")
                    _velocity_block("ur5e")
                _refresh_effective_labels()

            # ── Save Position ────────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Save Position").classes("font-semibold text-sm mb-2")
                save_name_input = ui.input("Name", value="home").props("dense").classes("w-48")
                ui.label(
                    "Save current joint positions for selected robot into its resource JSON."
                ).classes("text-xs text-slate-500 mb-2")
                save_env_label["label"] = ui.label("").classes("text-xs text-slate-500 mb-2")

                def _save_current() -> None:
                    name = str(save_name_input.value or "").strip()
                    if not name:
                        ui.notify("Enter a position name", type="warning", position="bottom-right", timeout=1800)
                        return
                    asyncio.create_task(_save_position(bridge, robot_select.value, name))

                with ui.row().classes("gap-2"):
                    ui.button("Save", on_click=_save_current, icon="save").props("outline")

            # ── Named Positions (Go To) ─────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Named Positions").classes("font-semibold text-sm mb-2")
                ui.label(
                    "Select a stored position and press Go to move the robot there."
                ).classes("text-xs text-slate-500 mb-2")
                named_pos_env_label = ui.label("").classes("text-xs text-slate-500 mb-1")
                named_pos_select = ui.select(
                    [], label="Position", value=None,
                ).props("dense").classes("w-48")
                named_pos_busy = {"moving": False}

                def _refresh_named_positions() -> None:
                    robot = robot_select.value
                    positions = bridge.list_named_positions(robot)
                    env = bridge.teleop_target_environment()
                    named_pos_env_label.set_text(f"Environment: {env}")
                    options = list(positions.keys())
                    named_pos_select.options = options
                    if named_pos_select.value not in options:
                        named_pos_select.value = options[0] if options else None
                    named_pos_select.update()

                async def _go_to_named() -> None:
                    name = named_pos_select.value
                    robot = robot_select.value
                    if not name:
                        ui.notify("Select a position first", type="warning", position="bottom-right", timeout=1800)
                        return
                    if named_pos_busy["moving"]:
                        ui.notify("Already moving...", type="info", position="bottom-right", timeout=900)
                        return
                    named_pos_busy["moving"] = True
                    go_btn.props("loading")
                    try:
                        ok, msg = await asyncio.to_thread(bridge.teleop_go_to_position, robot, name)
                        if ok:
                            ui.notify(f"{robot}: moved to '{name}'", type="positive", position="bottom-right", timeout=1800)
                        else:
                            ui.notify(f"{robot}: failed ({msg})", type="negative", position="bottom-right", timeout=3500)
                    except Exception as exc:
                        ui.notify(f"{robot}: error ({exc})", type="negative", position="bottom-right", timeout=3500)
                    finally:
                        named_pos_busy["moving"] = False
                        go_btn.props(remove="loading")

                with ui.row().classes("gap-2 items-center mt-2"):
                    go_btn = ui.button("Go", on_click=_go_to_named, icon="play_arrow").props("color=primary")
                    ui.button("Refresh", on_click=_refresh_named_positions, icon="refresh").props("outline dense")

                _refresh_named_positions()
                robot_select.on_value_change(lambda _: _refresh_named_positions())

        _refresh_teleop_status()
        ui.timer(1.0, _refresh_teleop_status)
        asyncio.create_task(_refresh_teleop_state_async())
        ui.timer(1.0, _refresh_teleop_state_async)

        # ── Keyboard handler ─────────────────────────────────────────
        def _on_key(e: KeyEventArguments):
            if e.action.keydown and not e.action.repeat:
                robot = robot_select.value
                step_mm = step_input.value or 10.0
                step_deg = joint_step_input.value or 2.0
                arm_vel = _effective_velocity(robot, "arm_vel", 1.0)
                grip_vel = _effective_velocity(robot, "gripper_vel", 1.0)
                key_name = e.key.name if hasattr(e.key, "name") else str(e.key)
                key_lower = key_name.lower()

                if key_lower == "t":
                    robot_select.value = "ur5e" if robot == "xarm6" else "xarm6"
                    ui.notify(f"Robot: {robot_select.value}", type="info", position="bottom-right", timeout=900)
                elif key_lower == "m":
                    mode_state["mode"] = "cartesian"
                    ui.notify(
                        f"Cartesian mode: {axis_state['axis'].upper()}",
                        type="info",
                        position="bottom-right",
                        timeout=900,
                    )
                elif key_lower == "p":
                    profile_toggle.value = "Precision"
                    _set_profile("Precision")
                elif key_lower == "f":
                    profile_toggle.value = "Fast"
                    _set_profile("Fast")
                elif key_lower in {"x", "y", "z"}:
                    axis_state["axis"] = key_lower
                    mode_state["mode"] = "cartesian"
                    ui.notify(
                        f"Cartesian axis: {axis_state['axis'].upper()}",
                        type="info",
                        position="bottom-right",
                        timeout=900,
                    )
                elif key_lower == "g":
                    mode_state["mode"] = "gripper"
                    ui.notify("Gripper mode", type="info", position="bottom-right", timeout=900)
                elif key_lower == "j":
                    mode_state["mode"] = "joint"
                    ui.notify(f"Joint mode: J{joint_state['idx']}", type="info", position="bottom-right", timeout=900)
                elif key_lower in {"1", "2", "3", "4", "5", "6"}:
                    joint_state["idx"] = int(key_lower)
                    mode_state["mode"] = "joint"
                    selected_joint_label.set_text(f"Selected: J{joint_state['idx']}")
                    ui.notify(f"Joint mode: J{joint_state['idx']}", type="info", position="bottom-right", timeout=900)
                elif key_lower == "h":
                    asyncio.create_task(_send_home(bridge, robot))
                elif key_lower == "s":
                    name = str(save_name_input.value or "").strip() or "home"
                    save_name_input.value = name
                    asyncio.create_task(_save_position(bridge, robot, name))
                elif key_name == "PageUp":
                    mode_state["mode"] = "cartesian"
                    asyncio.create_task(_send_jog(bridge, robot, "z", step_mm, arm_vel))
                elif key_name == "PageDown":
                    mode_state["mode"] = "cartesian"
                    asyncio.create_task(_send_jog(bridge, robot, "z", -step_mm, arm_vel))
                elif mode_state["mode"] == "gripper":
                    if key_name == "ArrowUp":
                        asyncio.create_task(_send_gripper(bridge, robot, "open", None, grip_vel))
                    elif key_name == "ArrowDown":
                        asyncio.create_task(_send_gripper(bridge, robot, "close", None, grip_vel))
                elif mode_state["mode"] == "joint":
                    if key_name == "ArrowUp":
                        asyncio.create_task(_send_joint(bridge, robot, joint_state["idx"], step_deg, arm_vel))
                    elif key_name == "ArrowDown":
                        asyncio.create_task(_send_joint(bridge, robot, joint_state["idx"], -step_deg, arm_vel))
                elif mode_state["mode"] == "cartesian":
                    # XY uses left/right only, as requested.
                    axis = axis_state["axis"]
                    if axis in {"x", "y"}:
                        if key_name == "ArrowRight":
                            asyncio.create_task(_send_jog(bridge, robot, axis, step_mm, arm_vel))
                        elif key_name == "ArrowLeft":
                            asyncio.create_task(_send_jog(bridge, robot, axis, -step_mm, arm_vel))
                    elif axis == "z":
                        if key_name == "ArrowUp":
                            asyncio.create_task(_send_jog(bridge, robot, "z", step_mm, arm_vel))
                        elif key_name == "ArrowDown":
                            asyncio.create_task(_send_jog(bridge, robot, "z", -step_mm, arm_vel))

        ui.keyboard(on_key=_on_key, ignore=["input", "select", "textarea"])


def _jog_btn(bridge, robot_select, step_input, velocity_getter, label, axis, direction, icon):
    def _on_click(a=axis, d=direction):
        step = (step_input.value or 10.0) * d
        try:
            arm_vel = float(velocity_getter(robot_select.value, "arm_vel", 1.0))
        except Exception:
            arm_vel = 1.0
        asyncio.create_task(_send_jog(bridge, robot_select.value, a, step, arm_vel))

    ui.button(icon=icon, on_click=_on_click).props("flat dense").classes("w-12 h-12").tooltip(label)


# =====================================================================
# ROS2 command senders (publish trajectory messages)
# =====================================================================
async def _send_jog(
    bridge: SystemBridge,
    robot: str,
    axis: str,
    step_mm: float,
    velocity_scale: float = 1.0,
) -> None:
    """Send one Cartesian jog command through the ROS2 teleop backend."""
    ok, msg = await asyncio.to_thread(bridge.teleop_jog, robot, axis, step_mm, velocity_scale)
    if ok:
        ui.notify(f"{robot}: jog {axis} {step_mm:+.0f}mm", type="positive", position="bottom-right", timeout=1200)
        return
    ui.notify(f"{robot}: jog failed ({msg})", type="negative", position="bottom-right", timeout=3500)


async def _send_gripper(
    bridge: SystemBridge,
    robot: str,
    action: str,
    step: float | None = None,
    velocity_scale: float = 1.0,
) -> None:
    """Send one gripper open/close command through the ROS2 teleop backend."""
    try:
        ok, msg = await asyncio.to_thread(bridge.teleop_gripper, robot, action, step, velocity_scale)
        if ok:
            ui.notify(f"{robot}: gripper {action}", type="positive", position="bottom-right", timeout=1200)
            return
        ui.notify(f"{robot}: gripper failed ({msg})", type="negative", position="bottom-right", timeout=3500)
    except Exception as exc:
        ui.notify(f"{robot}: gripper failed ({exc})", type="negative", position="bottom-right", timeout=3500)


async def _send_home(bridge: SystemBridge, robot: str) -> None:
    """Send one move-home command through the ROS2 teleop backend."""
    try:
        ok, msg = await asyncio.to_thread(bridge.teleop_home, robot)
        if ok:
            ui.notify(f"{robot}: moving home", type="positive", position="bottom-right", timeout=1200)
            return
        ui.notify(f"{robot}: move home failed ({msg})", type="negative", position="bottom-right", timeout=3500)
    except Exception as exc:
        ui.notify(f"{robot}: move home failed ({exc})", type="negative", position="bottom-right", timeout=3500)


async def _send_joint(
    bridge: SystemBridge,
    robot: str,
    joint_idx: int,
    delta_deg: float,
    velocity_scale: float = 1.0,
) -> None:
    """Send one joint jog command through the ROS2 teleop backend."""
    ok, msg = await asyncio.to_thread(bridge.teleop_joint, robot, joint_idx, delta_deg, velocity_scale)
    if ok:
        ui.notify(
            f"{robot}: J{joint_idx} {delta_deg:+.2f}deg",
            type="positive",
            position="bottom-right",
            timeout=1200,
        )
        return
    ui.notify(f"{robot}: joint jog failed ({msg})", type="negative", position="bottom-right", timeout=3500)


async def _save_position(bridge: SystemBridge, robot: str, name: str) -> None:
    """Save current robot joint positions under a named entry."""
    ok, msg = await asyncio.to_thread(bridge.teleop_save_position, robot, name)
    if ok:
        ui.notify(f"{robot}: {msg}", type="positive", position="bottom-right", timeout=2200)
        return
    ui.notify(f"{robot}: save failed ({msg})", type="negative", position="bottom-right", timeout=3500)
