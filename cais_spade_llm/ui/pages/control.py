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
    "gazebo_dual": (
        "Dual Robots (xArm6 + UR5e)",
        "No-hardware Gazebo + MoveIt/RViz; RViz controls xArm6 and UR5e planning groups",
    ),
    "gazebo_xarm6": ("xArm6 Only", "Single xArm6 Gazebo + MoveIt + RViz"),
    "gazebo_ur5e": ("UR5e Only", "Single UR5e + RG2 Gazebo + MoveIt + RViz"),
}

_HARDWARE_STACKS = {
    "xarm6": (
        "xArm6 Hardware Stack",
        "Start xArm6 MoveIt realmove stack (includes embedded driver)",
    ),
    "ur5e": (
        "UR5e Hardware Stack",
        "Auto sequence: start RTDE trajectory server, RG2 gripper bridge, and MoveIt",
    ),
}
_HARDWARE_PROC_NAMES = (
    "hardware_xarm6_driver",
    "hardware_xarm6_moveit",
    "hardware_ur5e_rtde_trajectory_server",
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


def _render_ur5e_rtde_and_rg2_status(status: dict) -> None:
    for prefix, source in _ur5e_status_sources(status):
        rtde_state = str(source.get("rtde_trajectory_server") or "").strip()
        if rtde_state:
            rtde_classes = (
                "text-xs text-red-700"
                if rtde_state in {"blocked", "failed"}
                else "text-xs text-slate-500"
            )
            ui.label(f"{prefix}RTDE trajectory server: {rtde_state}").classes(rtde_classes)
        rtde_message = str(source.get("rtde_trajectory_message") or "").strip()
        if rtde_message:
            ui.label(f"{prefix}RTDE trajectory: {rtde_message}").classes("text-xs text-slate-500")
        rtde_blocked = str(source.get("rtde_trajectory_blocked_reason") or "").strip()
        if rtde_blocked:
            ui.label(f"{prefix}RTDE trajectory blocked: {rtde_blocked}").classes(
                "text-xs text-red-700"
            )
        joint_states_fresh = source.get("joint_states_fresh")
        if joint_states_fresh is not None:
            ui.label(f"{prefix}UR5e /joint_states fresh: {bool(joint_states_fresh)}").classes(
                "text-xs text-slate-500"
            )
        max_velocity = source.get("rtde_trajectory_max_segment_velocity_rad_s")
        if max_velocity is not None:
            try:
                value = float(max_velocity)
                ui.label(f"{prefix}RTDE max segment velocity: {value:.3f} rad/s").classes(
                    "text-xs text-slate-500"
                )
            except (TypeError, ValueError):
                ui.label(f"{prefix}RTDE max segment velocity: {max_velocity}").classes(
                    "text-xs text-slate-500"
                )
        time_scale = source.get("rtde_trajectory_time_scale_applied")
        if time_scale is not None:
            try:
                value = float(time_scale)
                ui.label(f"{prefix}RTDE time scale: {value:.3f}").classes("text-xs text-slate-500")
            except (TypeError, ValueError):
                ui.label(f"{prefix}RTDE time scale: {time_scale}").classes("text-xs text-slate-500")

        gripper_state = source.get("gripper")
        if gripper_state is not None:
            ui.label(f"{prefix}RG2 bridge: {str(gripper_state)}").classes("text-xs text-slate-500")
        gripper_action = str(source.get("gripper_action") or "").strip()
        if gripper_action:
            ui.label(f"{prefix}RG2 bridge: action {gripper_action}").classes(
                "text-xs text-slate-500"
            )


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

        # ── Function Record / Replay ─────────────────────────────────
        _function_record_panel(bridge)

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
            hw_warning = ui.label("").classes(
                "text-xs text-amber-700 bg-amber-50 px-2 py-1 rounded"
            )
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
                xarm_ip_input = (
                    ui.input("xArm6 IP", value=ips.get("xarm6", "")).props("dense").classes("w-44")
                )
                ur5e_ip_input = (
                    ui.input("UR5e IP", value=ips.get("ur5e", "")).props("dense").classes("w-44")
                )

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
                    all_ok = all(
                        hw_links.get(r, {}).get("reachable", False) for r in ("xarm6", "ur5e")
                    )
                    if all_ok:
                        hw_warning.set_visibility(False)
                    else:
                        hw_warning.set_text(
                            "One or more hardware robots are unreachable. Check power/network before launch."
                        )
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

                ui.button("Apply Hardware IPs", on_click=_apply_hw_ips, icon="save").props(
                    "outline dense"
                )

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
            any_hardware_running = any(
                statuses.get(name) == "running" for name in _HARDWARE_PROC_NAMES
            )

            with launch_container:
                # Gazebo variants.
                ui.label("Gazebo Simulation (No Hardware)").classes(
                    "text-sm font-semibold text-slate-600"
                )
                for name, (label, desc) in _GAZEBO_VARIANTS.items():
                    blocked_reason = None
                    if any_hardware_running and statuses.get(name, "stopped") != "running":
                        blocked_reason = "Blocked: hardware stack is running. Stop hardware first."
                    _proc_row(
                        bridge,
                        name,
                        label,
                        desc,
                        statuses.get(name, "stopped"),
                        blocked_reason=blocked_reason,
                    )

                ui.separator().classes("my-2")

                ui.label("Hardware Launch").classes("text-sm font-semibold text-slate-600")
                for robot, (label, desc) in _HARDWARE_STACKS.items():
                    blocked_reason = None
                    stack_status = bridge.hardware_stack_status(robot)
                    if any_gazebo_running and stack_status.get("overall") != "running":
                        blocked_reason = "Blocked: Gazebo is running. Stop Gazebo first."
                    _hardware_stack_row(
                        bridge, robot, label, desc, stack_status, blocked_reason=blocked_reason
                    )

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
                        ui.notify(
                            "Cleanup complete: removed stale ROS2/MoveIt/driver processes",
                            type="info",
                        )
                        _refresh(force=True)

                    def _cleanup():
                        asyncio.create_task(_cleanup_async())

                    ui.button("Stop All", on_click=_stop_all, icon="stop_circle").props(
                        "flat dense"
                    ).classes("text-red-600")
                    ui.button(
                        "Reset Gazebo Scene", on_click=_reset_gazebo, icon="restart_alt"
                    ).props("flat dense").classes("text-blue-700")
                    ui.button("Cleanup", on_click=_cleanup, icon="cleaning_services").props(
                        "flat dense"
                    ).classes("text-amber-700")

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
                _render_ur5e_rtde_and_rg2_status(status)

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
            "Use Function Record / Replay to capture, preview, and replay monitor-mode motions."
        ).classes("text-xs text-slate-500 mb-3")

        container = ui.column().classes("w-full gap-3")
        refresh_state = {"signature": None, "busy": False}

        def _signature(rows: dict[str, dict]) -> tuple:
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
                        str(dict(row.get("dual_drag_markers") or {}).get("state", "")),
                        str(dict(row.get("dual_drag_markers") or {}).get("action", "")),
                        str(dict(row.get("dual_drag_markers") or {}).get("stage", "")),
                        str(dict(row.get("dual_drag_markers") or {}).get("message", "")),
                        str(dict(row.get("dual_drag_markers") or {}).get("last_error", "")),
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
                rows = bridge.digital_twin_statuses()
                signature = _signature(rows)
                if not force and signature == refresh_state["signature"]:
                    return
                refresh_state["signature"] = signature
            finally:
                refresh_state["busy"] = False

            container.clear()
            with container:
                active_target = _digital_twin_active_target(rows)
                with ui.row().classes(
                    "w-full items-center gap-3 px-2 py-1 bg-slate-50 rounded text-xs font-semibold text-slate-600"
                ):
                    ui.label("target").classes("w-44")
                    ui.label("gazebo").classes("w-48")
                    ui.label("moviet").classes("w-36")
                    ui.label("hardware").classes("w-52")
                    ui.label("sync/status").classes("flex-1 min-w-64")
                for target, row in rows.items():
                    _digital_twin_row(bridge, target, row, _refresh, active_target=active_target)

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


def _digital_twin_row_is_running(row: dict) -> bool:
    gazebo_status = str(dict(row.get("gazebo") or {}).get("status", "stopped"))
    hardware_overall = str(dict(row.get("hardware") or {}).get("overall", "unknown"))
    sync_process_status = str(dict(row.get("sync/status") or {}).get("process_status", "unknown"))
    return (
        gazebo_status == "running"
        or hardware_overall in {"running", "partial"}
        or sync_process_status == "running"
    )


def _digital_twin_active_target(rows: dict[str, dict]) -> str:
    for target, row in rows.items():
        if _digital_twin_row_is_running(dict(row or {})):
            return str(target)
    return ""


def _digital_twin_row(
    bridge: SystemBridge, target: str, row: dict, refresh_callback, *, active_target: str = ""
) -> None:
    gazebo = dict(row.get("gazebo") or {})
    moviet = dict(row.get("moviet") or {})
    hardware = dict(row.get("hardware") or {})
    sync = dict(row.get("sync/status") or {})
    domains = dict(row.get("domains") or {})

    supported = bool(row.get("supported", False))
    blocked_reason = str(row.get("blocked_reason", "") or "").strip()
    repair_needed = bool(row.get("repair_needed", False))
    repair_reason = str(row.get("repair_reason", "") or "").strip()
    gazebo_status = str(gazebo.get("status", "stopped"))
    hardware_status = dict(hardware.get("status") or {})
    hardware_overall = str(hardware.get("overall", "unknown"))
    hardware_robots = [str(robot).strip().lower() for robot in (hardware.get("robots") or [])]
    sync_state = str(sync.get("state", "unknown"))
    sync_process_status = str(sync.get("process_status", "unknown"))
    sync_message = str(sync.get("message", "") or "").strip()
    dual_drag_markers = dict(row.get("dual_drag_markers") or {})
    sim_mode = str(row.get("sim_mode", "monitor") or "monitor")
    sim_modes = list(row.get("sim_modes") or [])
    is_running = _digital_twin_row_is_running(row)
    other_target_running = bool(active_target and active_target != target)

    with ui.row().classes(
        "w-full items-stretch gap-3 border-b border-slate-100 px-2 py-3 flex-wrap"
    ):
        with ui.column().classes("w-44 gap-2"):
            ui.label(target).classes("font-semibold text-sm")
            ui.label("digital twin").classes("text-xs text-slate-400")
            if blocked_reason:
                ui.label(blocked_reason).classes("text-xs text-amber-700")
            if repair_needed and repair_reason:
                ui.label(f"Repair required: {repair_reason}").classes("text-xs text-red-700")

            async def _start_twin_async() -> None:
                try:
                    err = await asyncio.to_thread(
                        bridge.digital_twin_start,
                        target,
                        repair=repair_needed,
                    )
                    if err:
                        ui.notify(err, type="warning", timeout=5000)
                    else:
                        action = "Repaired" if repair_needed else "Started"
                        ui.notify(f"{action} {target} digital twin", type="positive")
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
                if other_target_running:
                    ui.notify(
                        f"Stop {active_target} digital twin before starting {target}.",
                        type="warning",
                        timeout=4500,
                    )
                    return
                if blocked_reason:
                    ui.notify(blocked_reason, type="warning", timeout=4500)
                    return
                asyncio.create_task(_start_twin_async())

            def _stop_twin() -> None:
                asyncio.create_task(_stop_twin_async())

            start_disabled = (
                (not supported)
                or other_target_running
                or (is_running and not repair_needed)
                or bool(blocked_reason)
            )
            stop_disabled = (not supported) or not is_running or other_target_running
            with ui.row().classes("gap-1"):
                ui.button(
                    "Repair Twin" if repair_needed else "Start Twin",
                    on_click=_start_twin,
                    icon="build" if repair_needed else "play_arrow",
                ).props(
                    "flat dense" + (" disable" if start_disabled else "")
                ).classes("text-green-600")
                ui.button("Stop Twin", on_click=_stop_twin, icon="stop").props(
                    "flat dense" + (" disable" if stop_disabled else "")
                ).classes("text-red-600")

            if supported and len(sim_modes) > 1:

                def _handle_sim_mode_change(e) -> None:
                    err = bridge.digital_twin_set_sim_mode(target, str(e.value or ""))
                    if err:
                        ui.notify(err, type="warning", timeout=3500)
                    refresh_callback(force=True)

                sim_mode_select = (
                    ui.select(
                        {m: _DT_MODE_LABELS.get(m, m) for m in sim_modes},
                        value=sim_mode,
                        label="mode",
                        on_change=_handle_sim_mode_change,
                    )
                    .props("dense")
                    .classes("w-40")
                )
                if is_running:
                    sim_mode_select.props("disable")
                ui.label(
                    "Teach = build motions in sim MoveIt/RViz, then Capture / Replay (set before Start)"
                    if sim_mode == "teach"
                    else "Monitor = sim mirrors the live robot"
                ).classes("text-xs text-slate-400")
            elif supported:
                ui.label("mode: Monitor").classes("text-xs text-slate-500")
                ui.label("Monitor = sim mirrors the live robot").classes("text-xs text-slate-400")

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
                _render_ur5e_rtde_and_rg2_status(hardware_status)
                hardware_domains = dict(hardware.get("domains") or {})
                if hardware_domains:
                    domain_text = " | ".join(
                        f"{robot}: ROS_DOMAIN_ID={domain}"
                        for robot, domain in hardware_domains.items()
                    )
                    ui.label(domain_text).classes("text-xs text-slate-500")
                else:
                    ui.label(f"ROS_DOMAIN_ID={domains.get('hardware', '')}").classes(
                        "text-xs text-slate-500"
                    )
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
            marker_message = str(dual_drag_markers.get("message") or "").strip()
            marker_last_error = str(dual_drag_markers.get("last_error") or "").strip()
            if marker_message:
                ui.label(f"dual_drag_markers: {marker_message}").classes("text-xs text-slate-500")
            if marker_last_error:
                ui.label(f"dual_drag_markers last_error: {marker_last_error}").classes(
                    "text-xs text-red-700"
                )

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

            # Record/replay controls live in the separate Function Record / Replay panel.


def _function_record_panel(bridge: SystemBridge) -> None:
    rows = bridge.digital_twin_statuses()
    targets = [target for target, row in rows.items() if bool(row.get("supported", False))]
    if not targets:
        return
    active_target = _digital_twin_active_target(rows)
    initial_target = (
        active_target
        if active_target in targets
        else ("dual robots" if "dual robots" in targets else targets[0])
    )

    with ui.card().classes("w-full"):
        ui.label("Function Record / Replay").classes("text-lg font-semibold mb-1")
        with ui.row().classes("items-center gap-2 w-full"):
            target_select = (
                ui.select(
                    targets,
                    label="target",
                    value=initial_target,
                )
                .props("dense")
                .classes("w-44")
            )
            if active_target:
                target_select.disable()

        body = ui.column().classes("w-full gap-2")

        def _refresh_body(_e=None) -> None:
            body.clear()
            target = str(target_select.value or "").strip()
            if not target:
                return
            with body:
                _predefined_function_record_body(bridge, target)

        target_select.on_value_change(_refresh_body)
        _refresh_body()

        active_target_refresh = {"busy": False}

        async def _sync_active_target() -> None:
            if not _client_alive(body):
                return
            if active_target_refresh["busy"]:
                return
            active_target_refresh["busy"] = True
            try:
                current_rows = await asyncio.to_thread(bridge.digital_twin_statuses)
                if not _client_alive(body):
                    return
                current_targets = [
                    target
                    for target, row in current_rows.items()
                    if bool(row.get("supported", False))
                ]
                if not current_targets:
                    return
                current_active = _digital_twin_active_target(current_rows)
                target_select.options = current_targets
                desired = (
                    current_active
                    if current_active in current_targets
                    else str(target_select.value or "").strip()
                )
                if desired not in current_targets:
                    desired = (
                        "dual robots" if "dual robots" in current_targets else current_targets[0]
                    )
                changed = target_select.value != desired
                target_select.value = desired
                if current_active:
                    target_select.disable()
                else:
                    target_select.enable()
                target_select.update()
                if changed:
                    _refresh_body()
            finally:
                active_target_refresh["busy"] = False

        ui.timer(3.0, _sync_active_target)


def _predefined_function_record_body(  # noqa: C901, PLR0915 - UI callbacks share selection state.
    bridge: SystemBridge,
    target: str,
) -> None:
    robots = bridge.digital_twin_target_robots(target)
    functions = bridge.digital_twin_function_names()
    readiness: dict[str, object] = {"checked": False, "success": False}
    preview: dict[str, object] = {}
    pick_execution: dict[str, object] = {"busy": False, "active_function": ""}

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

    initial_robot = "ur5e" if "ur5e" in robots else (robots[0] if robots else "")
    initial_function = "pick_approach" if "pick_approach" in functions else functions[0]
    with ui.row().classes("items-center gap-2 w-full"):
        robot_select = (
            ui.select(robots, label="robot", value=initial_robot).props("dense").classes("w-36")
        )
        function_select = (
            ui.select(functions, label="function", value=initial_function)
            .props("dense")
            .classes("w-52")
        )
        location_select = (
            ui.select([], label="location / recording name").props("dense").classes("w-60")
        )
        part_select = ui.select([], label="part_name").props("dense").classes("w-36")

    ui.label(
        "The function and step IDs come from the Python robot task registry. "
        "Only steps marked Position required can be captured."
    ).classes("text-xs text-slate-500")
    info_label = ui.label("").classes("text-xs text-blue-700 font-semibold")

    with ui.row().classes("items-center gap-3 w-full flex-wrap"):
        readiness_label = ui.label("Capture readiness has not been checked.").classes(
            "text-xs text-slate-500"
        )

        async def _check_readiness() -> None:
            _notify("Checking read-only capture readiness...", type="ongoing", timeout=1500)
            result = await asyncio.to_thread(
                bridge.digital_twin_function_capture_readiness,
                target,
                str(robot_select.value or ""),
            )
            readiness.clear()
            readiness.update(result)
            readiness["checked"] = True
            _render_readiness()
            _render_steps()
            _notify(
                "Capture is ready in Local Control."
                if result.get("success")
                else str(result.get("blocked_reason") or "Capture is not ready."),
                type="positive" if result.get("success") else "warning",
                timeout=5000,
            )

        ui.button("Check Capture Readiness", on_click=_check_readiness, icon="fact_check").props(
            "outline dense"
        )

    steps_summary = ui.label("").classes("text-xs font-semibold")
    steps_container = ui.column().classes("w-full gap-2")

    def _current_robot() -> str:
        return str(robot_select.value or "").strip()

    def _current_function() -> str:
        return str(function_select.value or "").strip()

    def _current_name() -> str:
        return str(location_select.value or "").strip()

    def _current_part_name() -> str:
        return str(part_select.value or "").strip()

    def _default_location(function_name: str, options: list[str]) -> str:
        if function_name == "pick_approach" and "prusa-mk4-2" in options:
            return "prusa-mk4-2"
        if function_name == "place_approach" and "assembly_board-v1" in options:
            return "assembly_board-v1"
        return options[0] if options else ""

    def _sync_location() -> None:
        function_name = _current_function()
        location_argument = bridge.digital_twin_function_location_argument(function_name)
        options = bridge.digital_twin_function_location_options(
            _current_robot(),
            function_name,
        )
        if not location_argument:
            options = ["default"]
        location_select.options = options
        if location_select.value not in options:
            location_select.value = _default_location(function_name, options)
        if location_argument:
            location_select.enable()
        else:
            location_select.disable()
        location_select.update()

    def _sync_part_name() -> None:
        function_name = _current_function()
        visible = function_name in {"pick_approach", "place_approach"}
        try:
            options = bridge.digital_twin_function_part_options(function_name)
        except (OSError, TypeError, ValueError):
            log.exception("failed to load part_name options for %s", function_name)
            options = []
        part_select.options = options
        if part_select.value not in options:
            part_select.value = "MG" if "MG" in options else (options[0] if options else "")
        part_select.set_visibility(visible)
        part_select.update()

    def _render_readiness() -> None:
        if not readiness.get("checked"):
            readiness_label.set_text("Capture readiness has not been checked.")
            readiness_label.classes(replace="text-xs text-slate-500")
            return
        values = [
            f"RTDE receive: {'ready' if readiness.get('rtde_receive_connected') else 'not ready'}",
            f"joints: {'fresh' if readiness.get('joint_states_fresh') else 'not fresh'}",
            f"world → tool0: {'ready' if readiness.get('world_tool0_ready') else 'not ready'}",
            (
                "RTDE control: connected (not used for capture)"
                if readiness.get("rtde_control_connected")
                else "RTDE control: disconnected (allowed for capture)"
            ),
        ]
        blocked_reason = str(readiness.get("blocked_reason") or "").strip()
        if blocked_reason:
            values.append(blocked_reason)
        readiness_label.set_text(" | ".join(values))
        readiness_label.classes(
            replace=(
                "text-xs text-green-700" if readiness.get("success") else "text-xs text-amber-700"
            )
        )

    preview_container = ui.column().classes("w-full gap-1")

    def _render_preview() -> None:
        preview_container.clear()
        preview_button.set_visibility(
            _current_robot() == "ur5e" and _current_function() == "pick_approach"
        )
        if _current_part_name() and not pick_execution.get("busy"):
            preview_button.enable()
        else:
            preview_button.disable()
        if _current_function() != "pick_approach":
            return
        with preview_container:
            if not preview:
                ui.label(
                    "Preview Target requests one fresh detection for part_name and computes "
                    "the pick target without robot motion or RTDE control."
                ).classes("text-xs text-slate-500")
                return
            world_pose = dict(preview.get("world_pose") or {})
            confidence = preview.get("confidence")
            age_sec = preview.get("age_sec")
            confidence_text = (
                f"{float(confidence) * 100.0:.1f}%" if confidence is not None else "unavailable"
            )
            age_text = f"{float(age_sec):.2f} s" if age_sec is not None else "unavailable"
            pose_text = "unavailable"
            if world_pose:
                pose_text = (
                    f"x={float(world_pose.get('x', 0.0)):.4f}, "
                    f"y={float(world_pose.get('y', 0.0)):.4f}, "
                    f"z={float(world_pose.get('z', 0.0)):.4f}"
                )
            travel_z = preview.get("travel_z")
            pick_z = preview.get("pick_z")
            travel_text = f"{float(travel_z):.4f}" if travel_z is not None else "unavailable"
            pick_text = f"{float(pick_z):.4f}" if pick_z is not None else "unavailable"
            ready_text = "ready" if preview.get("ready") else "blocked"
            ui.label(
                f"part_name={preview.get('part_name') or _current_part_name()} | "
                f"confidence={confidence_text} | world {pose_text} | "
                f"travel_z={travel_text} | pick_z={pick_text} | "
                f"age={age_text} | readiness={ready_text}"
            ).classes(
                "text-xs text-green-700" if preview.get("ready") else "text-xs text-amber-700"
            )
            message = str(preview.get("blocked_reason") or preview.get("message") or "").strip()
            if message:
                ui.label(message).classes("text-xs text-slate-600")

    async def _preview_target() -> None:
        preview.clear()
        _render_preview()
        _notify("Requesting a fresh read-only pick target...", type="ongoing", timeout=1500)
        result = await asyncio.to_thread(
            bridge.digital_twin_preview_pick_target,
            target,
            _current_robot(),
            _current_part_name(),
        )
        preview.update(result)
        _render_preview()
        _notify(
            str(result.get("message") or ""),
            type="positive" if result.get("success") else "warning",
            timeout=6000,
        )

    preview_button = ui.button(
        "Preview Target",
        on_click=_preview_target,
        icon="visibility",
    ).props("outline dense")
    preview_button.set_visibility(False)

    pick_execution_container = ui.column().classes("w-full gap-2")
    with pick_execution_container:
        ui.label(
            "Preview Target is read-only. Running pick_approach requests a separate fresh "
            "detection after the physical staging movement."
        ).classes("text-xs text-amber-700")
        pick_execution_status = ui.label("").classes("text-xs text-slate-500")
        with ui.row().classes("items-center gap-2 flex-wrap"):
            with ui.dialog() as pick_approach_confirm, ui.card().classes("gap-3 max-w-xl"):
                ui.label("Run pick_approach on the physical UR5e?").classes("font-semibold")
                pick_approach_confirm_text = ui.label("").classes("text-sm")
                ui.label(
                    "This commands physical robot motion. Clear the workcell and switch the "
                    "pendant to Remote Control before continuing."
                ).classes("text-xs text-red-700")
                with ui.row().classes("justify-end gap-2 w-full"):
                    ui.button("Cancel", on_click=pick_approach_confirm.close).props("flat")

                    async def _confirmed_pick_approach() -> None:
                        client = _current_client()
                        pick_approach_confirm.close()
                        if pick_execution.get("busy"):
                            _notify(
                                "A UR5e pick operation is already running.",
                                type="warning",
                                client=client,
                            )
                            return
                        pick_execution.update(
                            {"busy": True, "active_function": "pick_approach"}
                        )
                        pick_approach_button.props("loading")
                        _render_pick_execution()
                        try:
                            result = await bridge.digital_twin_execute_pick_approach(
                                target,
                                _current_robot(),
                                _current_name(),
                                _current_part_name(),
                                confirmed=True,
                            )
                            message = str(result.get("message") or "pick_approach completed.")
                            pick_execution_status.set_text(message)
                            pick_execution_status.classes(
                                replace=(
                                    "text-xs text-green-700"
                                    if result.get("success")
                                    else "text-xs text-red-700"
                                )
                            )
                            _notify(
                                message,
                                type="positive" if result.get("success") else "negative",
                                timeout=7000,
                                client=client,
                            )
                        except Exception as exc:
                            log.exception("pick_approach UI execution failed")
                            message = f"pick_approach failed: {exc}"
                            pick_execution_status.set_text(message)
                            pick_execution_status.classes(replace="text-xs text-red-700")
                            _notify(message, type="negative", timeout=7000, client=client)
                        finally:
                            pick_execution.update({"busy": False, "active_function": ""})
                            pick_approach_button.props(remove="loading")
                            _render_pick_execution()

                    ui.button(
                        "Confirm Run pick_approach",
                        on_click=_confirmed_pick_approach,
                        icon="play_arrow",
                    ).props("color=red")

            def _open_pick_approach_confirmation() -> None:
                pick_approach_confirm_text.set_text(
                    f"The UR5e will move to {_current_name()}, request a fresh "
                    f"{_current_part_name()} detection, open the gripper, move above the "
                    "detected gear, and descend to the computed pick target."
                )
                pick_approach_confirm.open()

            pick_approach_button = (
                ui.button(
                    "Run pick_approach",
                    on_click=_open_pick_approach_confirmation,
                    icon="play_arrow",
                )
                .props("outline dense")
                .classes("text-red-600")
            )

            with ui.dialog() as pick_grasp_confirm, ui.card().classes("gap-3 max-w-xl"):
                ui.label("Run pick_grasp on the physical UR5e?").classes("font-semibold")
                pick_grasp_confirm_text = ui.label("").classes("text-sm")
                ui.label(
                    "This closes the physical RG2 gripper and lifts the gear. Continue only "
                    "after pick_approach has completed at the correct gear."
                ).classes("text-xs text-red-700")
                with ui.row().classes("justify-end gap-2 w-full"):
                    ui.button("Cancel", on_click=pick_grasp_confirm.close).props("flat")

                    async def _confirmed_pick_grasp() -> None:
                        client = _current_client()
                        pick_grasp_confirm.close()
                        if pick_execution.get("busy"):
                            _notify(
                                "A UR5e pick operation is already running.",
                                type="warning",
                                client=client,
                            )
                            return
                        pick_execution.update({"busy": True, "active_function": "pick_grasp"})
                        pick_grasp_button.props("loading")
                        _render_pick_execution()
                        try:
                            result = await bridge.digital_twin_execute_pick_grasp(
                                target,
                                _current_robot(),
                                _current_name(),
                                _current_part_name(),
                                confirmed=True,
                            )
                            message = str(result.get("message") or "pick_grasp completed.")
                            pick_execution_status.set_text(message)
                            pick_execution_status.classes(
                                replace=(
                                    "text-xs text-green-700"
                                    if result.get("success")
                                    else "text-xs text-red-700"
                                )
                            )
                            _notify(
                                message,
                                type="positive" if result.get("success") else "negative",
                                timeout=7000,
                                client=client,
                            )
                        except Exception as exc:
                            log.exception("pick_grasp UI execution failed")
                            message = f"pick_grasp failed: {exc}"
                            pick_execution_status.set_text(message)
                            pick_execution_status.classes(replace="text-xs text-red-700")
                            _notify(message, type="negative", timeout=7000, client=client)
                        finally:
                            pick_execution.update({"busy": False, "active_function": ""})
                            pick_grasp_button.props(remove="loading")
                            _render_pick_execution()

                    ui.button(
                        "Confirm Run pick_grasp",
                        on_click=_confirmed_pick_grasp,
                        icon="pan_tool",
                    ).props("color=red")

            def _open_pick_grasp_confirmation() -> None:
                pick_grasp_confirm_text.set_text(
                    f"The UR5e will grasp and lift {_current_part_name()} from "
                    f"{_current_name()}."
                )
                pick_grasp_confirm.open()

            pick_grasp_button = (
                ui.button(
                    "Run pick_grasp",
                    on_click=_open_pick_grasp_confirmation,
                    icon="pan_tool",
                )
                .props("outline dense")
                .classes("text-red-600")
            )

    def _render_pick_execution() -> None:
        visible = _current_robot() == "ur5e" and _current_function() == "pick_approach"
        pick_execution_container.set_visibility(visible)
        if not visible:
            return
        enabled = bool(
            _current_name() and _current_part_name() and not pick_execution.get("busy")
        )
        pick_approach_button.set_enabled(enabled)
        pick_grasp_button.set_enabled(enabled)
        if pick_execution.get("busy"):
            preview_button.disable()
            active_function = str(pick_execution.get("active_function") or "pick operation")
            pick_execution_status.set_text(f"{active_function} is running...")
            pick_execution_status.classes(replace="text-xs text-amber-700")
        elif _current_part_name():
            preview_button.enable()

    async def _capture_position(step_name: str, primitive: str) -> None:
        result = await asyncio.to_thread(
            bridge.digital_twin_capture_function_step,
            target,
            _current_robot(),
            _current_function(),
            _current_name(),
            step_name,
            primitive,
            part_name=_current_part_name(),
        )
        if isinstance(result.get("readiness"), dict):
            readiness.clear()
            readiness.update(dict(result["readiness"]))
            readiness["checked"] = True
            _render_readiness()
        _notify(
            str(result.get("message") or ""),
            type="positive" if result.get("success") else "warning",
            timeout=5000,
        )
        _render_steps()

    async def _save_position(step_name: str) -> None:
        result = await asyncio.to_thread(
            bridge.digital_twin_save_function_position,
            target,
            _current_robot(),
            _current_function(),
            _current_name(),
            step_name,
            part_name=_current_part_name(),
        )
        _notify(
            str(result.get("message") or ""),
            type="positive" if result.get("success") else "warning",
            timeout=5000,
        )
        _render_steps()

    async def _clear_position(step_name: str) -> None:
        result = await asyncio.to_thread(
            bridge.digital_twin_clear_function_position,
            target,
            _current_robot(),
            _current_function(),
            _current_name(),
            step_name,
            part_name=_current_part_name(),
        )
        _notify(
            str(result.get("message") or ""),
            type="positive" if result.get("success") else "warning",
            timeout=4500,
        )
        _render_steps()

    async def _test_position(step_name: str, *, client: Client | None = None) -> None:
        result = await asyncio.to_thread(
            bridge.digital_twin_test_function_position,
            target,
            _current_robot(),
            _current_function(),
            _current_name(),
            step_name,
            confirmed=True,
            part_name=_current_part_name(),
        )
        _notify(
            str(result.get("message") or ""),
            type="positive" if result.get("success") else "warning",
            timeout=6000,
            client=client,
        )

    def _render_steps() -> None:
        function_name = _current_function()
        name = _current_name()
        template = bridge.digital_twin_function_template(function_name)
        saved_steps = {
            str(step.get("step_name") or ""): dict(step)
            for step in bridge.digital_twin_list_function_file_steps(
                target,
                _current_robot(),
                function_name,
                name,
                part_name=_current_part_name(),
            )
        }
        buffered_steps = {
            str(step.get("step_name") or ""): dict(step)
            for step in bridge.digital_twin_list_function_buffer_steps(
                target,
                _current_robot(),
                function_name,
                name,
                part_name=_current_part_name(),
            )
        }
        required = [step for step in template if bool(step.get("recordable"))]
        saved_required = [
            step
            for step in required
            if str(step.get("step_name") or "") in saved_steps
            and saved_steps[str(step.get("step_name") or "")].get("pose")
        ]
        if required and function_name == "place_approach":
            steps_summary.set_text(
                f"{len(saved_required)}/{len(required)} positions saved for "
                f"{name} / {_current_part_name()}"
            )
        elif required:
            steps_summary.set_text(f"{len(saved_required)}/{len(required)} positions saved")
        else:
            steps_summary.set_text("No physical positions are required for this function.")
        info = bridge.digital_twin_function_info(
            target,
            _current_robot(),
            function_name,
            name,
            _current_part_name(),
        )
        info_label.set_text(
            f"Saving as: {info.get('display_path')}"
            if info.get("success") and required
            else "Function primitives are shown from the Python registry."
        )
        steps_container.clear()
        with steps_container:
            for index, step in enumerate(template, start=1):
                step_name = str(step.get("step_name") or "")
                primitive = str(step.get("primitive") or "")
                recordable = bool(step.get("recordable"))
                saved = saved_steps.get(step_name)
                buffered = buffered_steps.get(step_name)
                with ui.card().classes("w-full p-3"):
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(str(index)).classes("text-xs font-semibold w-5")
                        ui.label(step_name).classes("text-sm font-semibold")
                        ui.label("→").classes("text-xs text-slate-400")
                        ui.label(primitive).classes("text-sm text-blue-700")
                        ui.space()
                        if not recordable:
                            ui.label("No position required").classes("text-xs text-slate-500")
                        elif buffered:
                            ui.label("Captured; not saved").classes("text-xs text-amber-700")
                        elif saved:
                            ui.label("Position saved").classes("text-xs text-green-700")
                        else:
                            ui.label("Position required").classes("text-xs text-red-700")
                    ui.label(
                        f"Parameter source: {str(step.get('parameter_source') or '')}"
                    ).classes("text-xs text-slate-500")
                    if (
                        function_name == "pick_approach"
                        and step_name == "move_to_origin_resource_location"
                    ):
                        ui.label(
                            "Automatic physical staging from `origin_resource_location`; "
                            "no recording required."
                        ).classes("text-xs text-blue-700")
                    position = buffered or saved
                    pose = dict(position.get("pose") or {}) if position else {}
                    if pose:
                        ui.label(
                            "world → tool0: "
                            f"x={float(pose.get('x', 0.0)):.4f}, "
                            f"y={float(pose.get('y', 0.0)):.4f}, "
                            f"z={float(pose.get('z', 0.0)):.4f}, "
                            f"q=({float(pose.get('qx', 0.0)):.4f}, "
                            f"{float(pose.get('qy', 0.0)):.4f}, "
                            f"{float(pose.get('qz', 0.0)):.4f}, "
                            f"{float(pose.get('qw', 1.0)):.4f})"
                        ).classes("text-xs text-slate-600")
                    if not recordable:
                        continue
                    with ui.row().classes("items-center gap-2 mt-1 flex-wrap"):
                        capture_button = ui.button(
                            "Capture Position",
                            on_click=lambda _e, s=step_name, p=primitive: _capture_position(s, p),
                            icon="fiber_manual_record",
                        ).props("flat dense")
                        if (
                            _current_robot() != "ur5e"
                            or not readiness.get("success")
                            or not name
                            or (function_name == "place_approach" and not _current_part_name())
                        ):
                            capture_button.disable()
                        save_button = ui.button(
                            "Save/Replace Position",
                            on_click=lambda _e, s=step_name: _save_position(s),
                            icon="save",
                        ).props("flat dense")
                        if not buffered:
                            save_button.disable()
                        clear_button = (
                            ui.button(
                                "Clear Position",
                                on_click=lambda _e, s=step_name: _clear_position(s),
                                icon="delete",
                            )
                            .props("flat dense")
                            .classes("text-red-600")
                        )
                        if not buffered and not saved:
                            clear_button.disable()

                        with ui.dialog() as test_confirm, ui.card().classes("gap-3"):
                            ui.label(f"Test {function_name}.{step_name}?").classes("font-semibold")
                            ui.label(
                                "This commands physical robot motion through move_cartesian. "
                                "Clear the workcell and switch the pendant to Remote Control first."
                            ).classes("text-xs text-red-700")
                            with ui.row().classes("justify-end gap-2 w-full"):
                                ui.button("Cancel", on_click=test_confirm.close).props("flat")

                                async def _confirmed_test(
                                    step_name=step_name,
                                    dialog=test_confirm,
                                ) -> None:
                                    client = _current_client()
                                    dialog.close()
                                    await _test_position(step_name, client=client)

                                ui.button(
                                    "Test Position",
                                    on_click=_confirmed_test,
                                    icon="send",
                                ).props("color=red")
                        test_button = (
                            ui.button(
                                "Test Position",
                                on_click=test_confirm.open,
                                icon="play_arrow",
                            )
                            .props("outline dense")
                            .classes("text-red-600")
                        )
                        if not saved:
                            test_button.disable()

    def _selection_changed(_e=None) -> None:
        readiness.clear()
        readiness.update({"checked": False, "success": False})
        preview.clear()
        _render_readiness()
        _sync_location()
        _sync_part_name()
        _render_preview()
        _render_pick_execution()
        _render_steps()

    def _part_name_changed(_e=None) -> None:
        preview.clear()
        _render_preview()
        _render_pick_execution()
        _render_steps()

    def _location_changed(_e=None) -> None:
        _render_pick_execution()
        _render_steps()

    robot_select.on_value_change(_selection_changed)
    function_select.on_value_change(_selection_changed)
    location_select.on_value_change(_location_changed)
    part_select.on_value_change(_part_name_changed)
    _sync_location()
    _sync_part_name()
    _render_readiness()
    _render_preview()
    _render_pick_execution()
    _render_steps()
    ui.label(
        "Capture is read-only and works with the UR5e pendant in Local Control. "
        "Testing is a separate motion action and requires Physical mode, Remote Control, "
        "and explicit confirmation."
    ).classes("text-xs text-amber-700 mt-2")


# =====================================================================
# Teleop Section
# =====================================================================
_AXIS_KEYS = {
    "x": ("ArrowRight", "ArrowLeft"),
    "y": ("ArrowUp", "ArrowDown"),
    "z": ("PageUp", "PageDown"),
}
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
        xarm6_initial_target = bridge.teleop_target("xarm6", "state")
        ur5e_initial_target = bridge.teleop_target("ur5e", "state")
        initial_robot = (
            "ur5e"
            if bool(ur5e_initial_target.get("ready"))
            and not bool(xarm6_initial_target.get("ready"))
            else "xarm6"
        )
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.label("Robot:").classes("font-semibold text-sm")
            robot_select = ui.toggle(["xarm6", "ur5e"], value=initial_robot).classes(
                "text-sm"
            )

        mode_state = {"mode": "cartesian"}  # cartesian | gripper | joint
        axis_state = {"axis": "y"}  # x | y | z
        joint_state = {"idx": 1}  # 1..6
        profile_state = {"mode": "fast"}  # precision | fast
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
                    f"{profile_state['mode'].title()} mode applies velocity x{multiplier:.2f} "
                    "and profile step presets."
                )
            for robot_name, label in effective_labels.items():
                arm = _effective_velocity(robot_name, "arm_vel", 1.0)
                grip = _effective_velocity(robot_name, "gripper_vel", 1.0)
                label.set_text(
                    f"Applied velocity: arm {arm:.2f}, gripper {grip:.2f} (x{multiplier:.2f})"
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
                    ui.notify(
                        f"{mode.title()} mode", type="info", position="bottom-right", timeout=900
                    )

            profile_toggle = ui.toggle(
                ["Precision", "Fast"],
                value="Fast",
                on_change=lambda e: _set_profile(e.value),
            ).props("dense")
            profile_note["label"] = ui.label("").classes("text-xs text-slate-500")

        with (
            ui.element("div")
            .classes("w-full columns-1 lg:columns-2 xl:columns-3")
            .style("column-gap: 1.5rem;")
        ):
            # ── Cartesian Jog Pad ────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Cartesian Jog").classes("font-semibold text-sm mb-2")
                step_input = ui.number(
                    "Step (mm)", value=10.0, min=0.1, max=100.0, step=0.1
                ).classes("w-32 mb-3")
                step_inputs["cartesian"] = step_input
                ui.label("Switching Precision/Fast also updates this step value.").classes(
                    "text-xs text-slate-500 mb-2"
                )

                # X/Y pad (top-down view).
                ui.label("X / Y Axes").classes("text-xs text-slate-500 mb-1")
                with ui.column().classes("items-center gap-1"):
                    _jog_btn(
                        bridge,
                        robot_select,
                        step_input,
                        _effective_velocity,
                        "Y+",
                        "y",
                        1,
                        "arrow_upward",
                    )
                    with ui.row().classes("gap-1"):
                        _jog_btn(
                            bridge,
                            robot_select,
                            step_input,
                            _effective_velocity,
                            "X-",
                            "x",
                            -1,
                            "arrow_back",
                        )
                        ui.button(icon="radio_button_unchecked").props(
                            "flat dense disable"
                        ).classes("w-12 h-12")
                        _jog_btn(
                            bridge,
                            robot_select,
                            step_input,
                            _effective_velocity,
                            "X+",
                            "x",
                            1,
                            "arrow_forward",
                        )
                    _jog_btn(
                        bridge,
                        robot_select,
                        step_input,
                        _effective_velocity,
                        "Y-",
                        "y",
                        -1,
                        "arrow_downward",
                    )

                # Z axis.
                ui.label("Z Axis").classes("text-xs text-slate-500 mt-3 mb-1")
                with ui.row().classes("gap-2 justify-center"):
                    _jog_btn(
                        bridge,
                        robot_select,
                        step_input,
                        _effective_velocity,
                        "Z+",
                        "z",
                        1,
                        "expand_less",
                    )
                    _jog_btn(
                        bridge,
                        robot_select,
                        step_input,
                        _effective_velocity,
                        "Z-",
                        "z",
                        -1,
                        "expand_more",
                    )

            # ── Joint Jog ────────────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Joint Jog").classes("font-semibold text-sm mb-2")
                joint_step_input = ui.number(
                    "Step (deg)", value=2.0, min=0.05, max=30.0, step=0.05
                ).classes("w-32 mb-2")
                step_inputs["joint"] = joint_step_input
                _apply_profile_steps(profile_state["mode"])
                selected_joint_label = ui.label("Selected: J1").classes(
                    "text-xs text-slate-500 mb-2"
                )

                def _select_joint(idx: int):
                    joint_state["idx"] = idx
                    mode_state["mode"] = "joint"
                    selected_joint_label.set_text(f"Selected: J{idx}")
                    ui.notify(
                        f"Joint mode: J{idx}", type="info", position="bottom-right", timeout=900
                    )

                with ui.row().classes("gap-1 mb-2"):
                    for idx in range(1, 7):
                        ui.button(str(idx), on_click=lambda _=None, j=idx: _select_joint(j)).props(
                            "outline dense"
                        )

                with ui.row().classes("gap-2"):

                    def _joint_minus():
                        step_deg = joint_step_input.value or 2.0
                        arm_vel = _effective_velocity(robot_select.value, "arm_vel", 1.0)
                        asyncio.create_task(
                            _send_joint(
                                bridge, robot_select.value, joint_state["idx"], -step_deg, arm_vel
                            )
                        )

                    def _joint_plus():
                        step_deg = joint_step_input.value or 2.0
                        arm_vel = _effective_velocity(robot_select.value, "arm_vel", 1.0)
                        asyncio.create_task(
                            _send_joint(
                                bridge, robot_select.value, joint_state["idx"], step_deg, arm_vel
                            )
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
                        asyncio.create_task(
                            _send_gripper(bridge, robot_select.value, "open", 1.0, grip_vel)
                        )

                    def _full_close():
                        mode_state["mode"] = "gripper"
                        grip_vel = _effective_velocity(robot_select.value, "gripper_vel", 1.0)
                        asyncio.create_task(
                            _send_gripper(bridge, robot_select.value, "close", 1.0, grip_vel)
                        )

                    def _step_open():
                        mode_state["mode"] = "gripper"
                        grip_vel = _effective_velocity(robot_select.value, "gripper_vel", 1.0)
                        asyncio.create_task(
                            _send_gripper(bridge, robot_select.value, "open", None, grip_vel)
                        )

                    def _step_close():
                        mode_state["mode"] = "gripper"
                        grip_vel = _effective_velocity(robot_select.value, "gripper_vel", 1.0)
                        asyncio.create_task(
                            _send_gripper(bridge, robot_select.value, "close", None, grip_vel)
                        )

                    ui.button("Full Open", on_click=_full_open, icon="open_with").props("outline")
                    ui.button("Full Close", on_click=_full_close, icon="close_fullscreen").props(
                        "outline"
                    )
                with ui.row().classes("gap-2 justify-center"):
                    ui.button("Open", on_click=_step_open, icon="add").props("outline")
                    ui.button("Close", on_click=_step_close, icon="remove").props("outline")

                # Home button.
                ui.separator().classes("my-3")

                def _go_home():
                    asyncio.create_task(_send_home(bridge, robot_select.value))

                ui.button("Move Home", on_click=_go_home, icon="home").props("outline").classes(
                    "w-full"
                )

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
                state_labels["status"] = ui.label("State: checking...").classes(
                    "text-xs text-slate-500 mb-1"
                )
                state_labels["xyz"] = ui.label("X/Y/Z [m]: -- / -- / --").classes(
                    "text-xs font-mono"
                )
                state_labels["rpy"] = ui.label("Rx/Ry/Rz [deg]: -- / -- / --").classes(
                    "text-xs font-mono mb-1"
                )
                with ui.row().classes("gap-3"):
                    for idx in range(1, 4):
                        state_labels["joints"].append(
                            ui.label(f"J{idx}: --").classes("text-xs font-mono")
                        )
                with ui.row().classes("gap-3"):
                    for idx in range(4, 7):
                        state_labels["joints"].append(
                            ui.label(f"J{idx}: --").classes("text-xs font-mono")
                        )

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
                        ui.notify(
                            "Enter a position name",
                            type="warning",
                            position="bottom-right",
                            timeout=1800,
                        )
                        return
                    asyncio.create_task(_save_position(bridge, robot_select.value, name))

                with ui.row().classes("gap-2"):
                    ui.button("Save", on_click=_save_current, icon="save").props("outline")

            # ── Named Positions (Go To) ─────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Named Positions").classes("font-semibold text-sm mb-2")
                ui.label("Select a stored position and press Go to move the robot there.").classes(
                    "text-xs text-slate-500 mb-2"
                )
                named_pos_env_label = ui.label("").classes("text-xs text-slate-500 mb-1")
                named_pos_select = (
                    ui.select(
                        [],
                        label="Position",
                        value=None,
                    )
                    .props("dense")
                    .classes("w-48")
                )
                named_pos_busy = {"moving": False}
                named_pos_readiness_state = {
                    "busy": False,
                    "ready": False,
                    "robot": "",
                }
                named_pos_readiness_label = ui.label(
                    "Trajectory interface: checking..."
                ).classes("text-xs text-slate-500")

                def _update_named_position_go_enabled() -> None:
                    go_btn.set_enabled(
                        bool(
                            named_pos_readiness_state["ready"]
                            and named_pos_select.value
                            and not named_pos_busy["moving"]
                        )
                    )

                async def _refresh_named_position_readiness() -> None:
                    if named_pos_readiness_state["busy"]:
                        return
                    if not _client_alive(named_pos_readiness_label):
                        return
                    named_pos_readiness_state["busy"] = True
                    robot = str(robot_select.value or "xarm6")
                    if named_pos_readiness_state["robot"] != robot:
                        named_pos_readiness_state["robot"] = robot
                        named_pos_readiness_state["ready"] = False
                        _update_named_position_go_enabled()
                    try:
                        ready, message = await asyncio.to_thread(
                            bridge.teleop_named_position_readiness,
                            robot,
                        )
                        if not _client_alive(named_pos_readiness_label):
                            return
                        if robot != str(robot_select.value or "xarm6"):
                            return
                        named_pos_readiness_state["ready"] = bool(ready)
                        named_pos_readiness_label.set_text(
                            f"Trajectory interface ({robot}): {message}"
                        )
                        named_pos_readiness_label.classes(
                            replace=(
                                "text-xs text-green-700"
                                if ready
                                else "text-xs text-red-700"
                            )
                        )
                        _update_named_position_go_enabled()
                    finally:
                        named_pos_readiness_state["busy"] = False

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
                    _update_named_position_go_enabled()
                    asyncio.create_task(_refresh_named_position_readiness())

                async def _go_to_named() -> None:
                    name = named_pos_select.value
                    robot = robot_select.value
                    if not name:
                        ui.notify(
                            "Select a position first",
                            type="warning",
                            position="bottom-right",
                            timeout=1800,
                        )
                        return
                    if named_pos_busy["moving"]:
                        ui.notify(
                            "Already moving...", type="info", position="bottom-right", timeout=900
                        )
                        return
                    named_pos_busy["moving"] = True
                    _update_named_position_go_enabled()
                    go_btn.props("loading")
                    try:
                        ok, msg = await asyncio.to_thread(bridge.teleop_go_to_position, robot, name)
                        if ok:
                            ui.notify(
                                f"{robot}: moved to '{name}'",
                                type="positive",
                                position="bottom-right",
                                timeout=1800,
                            )
                        else:
                            ui.notify(
                                f"{robot}: failed ({msg})",
                                type="negative",
                                position="bottom-right",
                                timeout=3500,
                            )
                    except Exception as exc:
                        ui.notify(
                            f"{robot}: error ({exc})",
                            type="negative",
                            position="bottom-right",
                            timeout=3500,
                        )
                    finally:
                        named_pos_busy["moving"] = False
                        go_btn.props(remove="loading")
                        _update_named_position_go_enabled()
                        asyncio.create_task(_refresh_named_position_readiness())

                with ui.row().classes("gap-2 items-center mt-2"):
                    go_btn = ui.button("Go", on_click=_go_to_named, icon="play_arrow").props(
                        "color=primary"
                    )
                    go_btn.set_enabled(False)
                    ui.button("Refresh", on_click=_refresh_named_positions, icon="refresh").props(
                        "outline dense"
                    )

                _refresh_named_positions()
                named_pos_select.on_value_change(lambda _: _update_named_position_go_enabled())
                robot_select.on_value_change(lambda _: _refresh_named_positions())
                ui.timer(3.0, _refresh_named_position_readiness)

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
                    ui.notify(
                        f"Robot: {robot_select.value}",
                        type="info",
                        position="bottom-right",
                        timeout=900,
                    )
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
                    ui.notify(
                        f"Joint mode: J{joint_state['idx']}",
                        type="info",
                        position="bottom-right",
                        timeout=900,
                    )
                elif key_lower in {"1", "2", "3", "4", "5", "6"}:
                    joint_state["idx"] = int(key_lower)
                    mode_state["mode"] = "joint"
                    selected_joint_label.set_text(f"Selected: J{joint_state['idx']}")
                    ui.notify(
                        f"Joint mode: J{joint_state['idx']}",
                        type="info",
                        position="bottom-right",
                        timeout=900,
                    )
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
                        asyncio.create_task(
                            _send_joint(bridge, robot, joint_state["idx"], step_deg, arm_vel)
                        )
                    elif key_name == "ArrowDown":
                        asyncio.create_task(
                            _send_joint(bridge, robot, joint_state["idx"], -step_deg, arm_vel)
                        )
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
        ui.notify(
            f"{robot}: jog {axis} {step_mm:+.0f}mm",
            type="positive",
            position="bottom-right",
            timeout=1200,
        )
        return
    ui.notify(
        f"{robot}: jog failed ({msg})", type="negative", position="bottom-right", timeout=3500
    )


async def _send_gripper(
    bridge: SystemBridge,
    robot: str,
    action: str,
    step: float | None = None,
    velocity_scale: float = 1.0,
) -> None:
    """Send one gripper open/close command through the ROS2 teleop backend."""
    try:
        ok, msg = await asyncio.to_thread(
            bridge.teleop_gripper, robot, action, step, velocity_scale
        )
        if ok:
            ui.notify(
                f"{robot}: gripper {action}", type="positive", position="bottom-right", timeout=1200
            )
            return
        ui.notify(
            f"{robot}: gripper failed ({msg})",
            type="negative",
            position="bottom-right",
            timeout=3500,
        )
    except Exception as exc:
        ui.notify(
            f"{robot}: gripper failed ({exc})",
            type="negative",
            position="bottom-right",
            timeout=3500,
        )


async def _send_home(bridge: SystemBridge, robot: str) -> None:
    """Send one move-home command through the ROS2 teleop backend."""
    try:
        ok, msg = await asyncio.to_thread(bridge.teleop_home, robot)
        if ok:
            ui.notify(
                f"{robot}: moving home", type="positive", position="bottom-right", timeout=1200
            )
            return
        ui.notify(
            f"{robot}: move home failed ({msg})",
            type="negative",
            position="bottom-right",
            timeout=3500,
        )
    except Exception as exc:
        ui.notify(
            f"{robot}: move home failed ({exc})",
            type="negative",
            position="bottom-right",
            timeout=3500,
        )


async def _send_joint(
    bridge: SystemBridge,
    robot: str,
    joint_idx: int,
    delta_deg: float,
    velocity_scale: float = 1.0,
) -> None:
    """Send one joint jog command through the ROS2 teleop backend."""
    ok, msg = await asyncio.to_thread(
        bridge.teleop_joint, robot, joint_idx, delta_deg, velocity_scale
    )
    if ok:
        ui.notify(
            f"{robot}: J{joint_idx} {delta_deg:+.2f}deg",
            type="positive",
            position="bottom-right",
            timeout=1200,
        )
        return
    ui.notify(
        f"{robot}: joint jog failed ({msg})", type="negative", position="bottom-right", timeout=3500
    )


async def _save_position(bridge: SystemBridge, robot: str, name: str) -> None:
    """Save current robot joint positions under a named entry."""
    ok, msg = await asyncio.to_thread(bridge.teleop_save_position, robot, name)
    if ok:
        ui.notify(f"{robot}: {msg}", type="positive", position="bottom-right", timeout=2200)
        return
    ui.notify(
        f"{robot}: save failed ({msg})", type="negative", position="bottom-right", timeout=3500
    )
