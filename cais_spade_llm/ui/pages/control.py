"""Control page: Gazebo/hardware launch, interactive teleop with arrow buttons + keyboard."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from datetime import datetime

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
        "Start the xArm6 driver with direct trajectory control; no MoveIt, RViz, or Gazebo",
    ),
    "ur5e": (
        "UR5e Hardware Stack",
        "Start RTDE and RG2 with direct trajectory control; no MoveIt, RViz, or Gazebo",
    ),
    "dual robots": (
        "Dual Hardware Stack",
        "Start xArm6, UR5e RTDE, and RG2 with direct control; no MoveIt, RViz, or Gazebo",
    ),
}
_HARDWARE_PROC_NAMES = (
    "hardware_xarm6_driver",
    "hardware_xarm6_moveit",
    "hardware_ur5e_rtde_trajectory_server",
    "hardware_ur5e_rg2_gripper",
    "hardware_ur5e_moveit",
    "hardware_dual_robots_moveit",
    "hardware_robot_state_publisher",
)


def _assembly_board_v1_tag_currently_visible(status: dict) -> bool:
    """Return whether ID 70 is visible in a fresh live snapshot."""
    if not status.get("visible") or not status.get("valid"):
        return False
    try:
        frame_age_sec = float(status["frame_age_sec"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(frame_age_sec) and 0.0 <= frame_age_sec <= 2.0


def _assembly_board_v1_readiness(status: dict) -> dict[str, object]:
    """Describe whether one arm's accepted assembly_board-v1 pose is usable."""
    accepted = bool(status.get("accepted"))
    accepted_baseline_ready = status.get("accepted_baseline_ready")
    accepted_baseline_error = str(status.get("accepted_baseline_error") or "").strip()
    if accepted_baseline_ready is not None:
        if not bool(accepted_baseline_ready):
            if not accepted:
                if bool(status.get("post_staging_acceptance_allowed")):
                    message = (
                        "Board not accepted, so Capture Pose is blocked. Confirmed Run "
                        "place_approach will move to assembly_board-v1, collect ten fresh "
                        "ArUco ID 70 observations, and accept the board automatically."
                    )
                else:
                    message = accepted_baseline_error or (
                        "Board not accepted, and confirmed Run place_approach cannot accept "
                        "it automatically until the active calibration and 76 mm marker "
                        "configuration are ready."
                    )
            elif bool(status.get("post_staging_acceptance_allowed")):
                message = (
                    "The accepted board baseline must be refreshed, so Capture Pose is "
                    "blocked. Confirmed Run place_approach will move to assembly_board-v1, "
                    "collect ten fresh ArUco ID 70 observations, and reaccept the board "
                    "automatically before approaching it."
                )
            else:
                message = accepted_baseline_error or (
                    "The existing accepted board baseline cannot be refreshed automatically "
                    "during confirmed Run place_approach. Resolve the board calibration or "
                    "configuration warning first."
                )
            message = message.replace(
                "use Locate & Accept Board again.",
                "resolve this before confirmed Run place_approach.",
            ).replace(
                "use Locate & Accept Board",
                "resolve this before confirmed Run place_approach",
            )
            return {
                "usable": False,
                "level": "red",
                "message": message,
            }
        if _assembly_board_v1_tag_currently_visible(status):
            return {
                "usable": True,
                "level": "green",
                "message": "Accepted board baseline is usable and ArUco ID 70 is visible.",
            }
        if status.get("visible"):
            return {
                "usable": True,
                "level": "amber",
                "message": (
                    "Accepted board baseline is usable. The latest ArUco ID 70 snapshot is "
                    "not fresh; Capture Pose remains available and place_approach will "
                    "localize again before descent."
                ),
            }
        return {
            "usable": True,
            "level": "amber",
            "message": (
                "Accepted board baseline is usable. ArUco ID 70 is currently occluded; Capture "
                "Pose remains available and place_approach will localize again before descent."
            ),
        }
    calibration_changed = bool(status.get("calibration_changed"))
    known_live_pose = bool(status.get("visible") and status.get("valid") and status.get("pose"))
    movement_blocked = bool(status.get("movement_blocked") and known_live_pose)
    if not accepted:
        return {
            "usable": False,
            "level": "red",
            "message": (
                "Board not accepted. A fresh stable ArUco ID 70 observation is required "
                "before Capture Pose or Run place_approach."
            ),
        }
    if calibration_changed:
        return {
            "usable": False,
            "level": "red",
            "message": (
                "The accepted board pose uses a different camera calibration. "
                "A fresh stable ArUco ID 70 observation is required."
            ),
        }
    if movement_blocked:
        return {
            "usable": False,
            "level": "red",
            "message": (
                "A valid live observation shows assembly_board-v1 moved beyond 10 mm or "
                "2 deg. A fresh stable ArUco ID 70 observation is required."
            ),
        }
    if _assembly_board_v1_tag_currently_visible(status):
        return {
            "usable": True,
            "level": "green",
            "message": "Accepted board baseline is usable and ArUco ID 70 is visible.",
        }
    return {
        "usable": True,
        "level": "amber",
        "message": (
            "Accepted board baseline is usable. ArUco ID 70 is currently occluded; Capture "
            "Pose remains available and place_approach will localize again before descent."
        ),
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
        state_publisher = status.get("state_publisher")
        if state_publisher is not None:
            robot_bits.append(f"State publisher: {str(state_publisher)}")
        return " | ".join(robot_bits)
    driver_state = str(status.get("driver", "stopped"))
    control = str(status.get("control", "direct"))
    state_publisher = status.get("state_publisher")
    gripper_state = status.get("gripper")
    joint_control = status.get("joint_control")
    cartesian_control = status.get("cartesian_control")
    control_text = ""
    if joint_control is not None or cartesian_control is not None:
        control_text = (
            f" | Joint control: {str(joint_control or 'stopped')}"
            f" | Cartesian control: {str(cartesian_control or 'stopped')}"
        )
    state_text = (
        f" | State publisher: {str(state_publisher)}"
        if state_publisher is not None
        else ""
    )
    if gripper_state is None:
        return f"Driver: {driver_state} | Control: {control}{control_text}{state_text}"
    return (
        f"Driver: {driver_state} | Gripper: {str(gripper_state)} | "
        f"Control: {control}{control_text}{state_text}"
    )


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
        joint_action_ready = source.get("rtde_trajectory_joint_action_ready")
        cartesian_action_ready = source.get("rtde_trajectory_cartesian_action_ready")
        if joint_action_ready is not None:
            ui.label(f"{prefix}RTDE joint control ready: {bool(joint_action_ready)}").classes(
                "text-xs text-slate-500"
            )
        if cartesian_action_ready is not None:
            ui.label(
                f"{prefix}RTDE Cartesian control ready: {bool(cartesian_action_ready)}"
            ).classes("text-xs text-slate-500")
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
    refresh_callbacks: dict[str, Callable[[], None]] = {}
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Control").classes("text-2xl font-bold")

        # ── Gazebo / Hardware Launch ─────────────────────────────────
        _launch_section(bridge)

        # ── Digital Twin Launch ──────────────────────────────────────
        _digital_twin_launch_section(bridge)

        # ── Robot Functions ──────────────────────────────────────────
        _function_record_panel(bridge, refresh_callbacks)

        # ── Interactive Teleop ───────────────────────────────────────
        _teleop_section(bridge, refresh_callbacks)


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
            "Hardware stacks run without MoveIt, RViz, or Gazebo. xArm6 uses its trajectory "
            "controller directly; RTDE is the UR5e arm actuator."
        ).classes("text-xs text-slate-500 mb-2")

        hw_refresh_state = {"busy": False}
        ips = bridge.get_hardware_ips()
        with ui.column().classes("w-full gap-2 mb-3"):
            ui.label("Hardware Connectivity").classes("text-sm font-semibold text-slate-600")

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
                    probe = str(entry.get("probe") or "").strip()
                    probe_text = f" via {probe}" if probe else ""
                    if latency is not None:
                        return f"{name}: {ip} reachable{probe_text} ({latency:.1f} ms)"
                    return f"{name}: {ip} reachable{probe_text}"
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
                statuses = dict(signature[0] + signature[1])
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
                        _refresh,
                        blocked_reason=blocked_reason,
                    )

                ui.separator().classes("my-2")

                ui.label("Hardware Launch").classes("text-sm font-semibold text-slate-600")
                for robot, (label, desc) in _HARDWARE_STACKS.items():
                    blocked_reason = None
                    stack_status = bridge.hardware_stack_status(robot)
                    if any_gazebo_running and stack_status.get("overall") != "running":
                        blocked_reason = "Blocked: Gazebo is running. Stop Gazebo first."
                    else:
                        other_running = next(
                            (
                                other_robot
                                for other_robot in _HARDWARE_STACKS
                                if other_robot != robot
                                and bridge.hardware_stack_status(other_robot).get("overall")
                                == "running"
                            ),
                            "",
                        )
                        if other_running:
                            blocked_reason = (
                                f"Blocked: {other_running} Hardware Stack is running. "
                                "Stop it first."
                            )
                    _hardware_stack_row(
                        bridge,
                        robot,
                        label,
                        desc,
                        stack_status,
                        _refresh,
                        blocked_reason=blocked_reason,
                    )

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
    refresh_callback: Callable[..., None],
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
            refresh_callback(force=True)

        async def _stop_async(n=name):
            await asyncio.to_thread(bridge.ros2_stop, n)
            ui.notify(f"Stopped {label}", type="info")
            refresh_callback(force=True)

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
    refresh_callback: Callable[..., None],
    blocked_reason: str | None = None,
) -> None:
    overall = str(status.get("overall", "stopped"))
    lifecycle_state = str(status.get("lifecycle_state") or "").strip()
    if not lifecycle_state:
        lifecycle_state = "running" if overall == "running" else "stopped"
    color = (
        "green"
        if lifecycle_state == "running" and overall == "running"
        else "red"
        if lifecycle_state == "failed"
        else "orange"
        if lifecycle_state in {"starting", "stopping"} or overall == "partial"
        else "grey"
    )

    with ui.row().classes("items-center gap-4 w-full"):
        ui.icon("circle", color=color).classes("text-xs")

        with ui.column().classes("gap-0 flex-1"):
            ui.label(label).classes("font-semibold text-sm")
            ui.label(desc).classes("text-xs text-slate-400")
            ui.label(f"Lifecycle: {lifecycle_state}").classes("text-xs text-slate-500")
            generation = status.get("lifecycle_generation")
            if generation is not None:
                ui.label(f"Repair generation: {generation}").classes(
                    "text-xs text-slate-500"
                )
            ui.label(_hardware_status_text(status)).classes("text-xs text-slate-500")
            validated_process_pids = dict(
                status.get("validated_process_pids") or {}
            )
            if validated_process_pids:
                ownership = ", ".join(
                    f"{name} PID {process_id}"
                    for name, process_id in validated_process_pids.items()
                )
                ui.label(f"Validated ownership: {ownership}").classes(
                    "text-xs text-slate-500"
                )
            for robot_name, result in dict(
                status.get("stationary_results") or {}
            ).items():
                stationary_ready = bool(dict(result or {}).get("stationary_ready"))
                stationary_message = str(dict(result or {}).get("message") or "")
                ui.label(
                    f"{robot_name} stationary: "
                    f"{'ready' if stationary_ready else 'failed'}"
                    + (f" | {stationary_message}" if stationary_message else "")
                ).classes(
                    "text-xs text-slate-500" if stationary_ready else "text-xs text-red-700"
                )
            for robot_name, reset_result in dict(
                status.get("cartesian_jog_reset_results") or {}
            ).items():
                ui.label(f"{robot_name} Cartesian jog reset: {reset_result}").classes(
                    "text-xs text-slate-500"
                )
            last_error = str(status.get("last_error") or "").strip()
            if last_error:
                ui.label(last_error).classes("text-xs text-red-700")
            if robot in {"ur5e", "dual robots"}:
                _render_ur5e_rtde_and_rg2_status(status)

        repair_needed = lifecycle_state == "failed"
        selected_stack = str(status.get("selected_stack") or "")
        another_stack_selected = bool(selected_stack and selected_stack != robot)
        start_blocked = bool(blocked_reason) or another_stack_selected or lifecycle_state in {
            "starting",
            "running",
            "stopping",
        }
        stop_disabled = another_stack_selected or (
            lifecycle_state == "stopped" and overall == "stopped"
        )
        operation_state = {"busy": False}

        async def _start_async() -> None:
            try:
                operation = (
                    bridge.ros2_repair_hardware_stack
                    if repair_needed
                    else bridge.ros2_start_hardware_stack
                )
                err = await asyncio.to_thread(operation, robot)
                if err:
                    ui.notify(err, type="warning", timeout=5000)
                else:
                    verb = "Repaired" if repair_needed else "Started"
                    ui.notify(f"{verb} {label}", type="positive")
            finally:
                operation_state["busy"] = False
                if _client_alive(start_button):
                    start_button.props(remove="loading")
                refresh_callback(force=True)

        async def _stop_async() -> None:
            try:
                err = await asyncio.to_thread(bridge.ros2_stop_hardware_stack, robot)
                if err:
                    ui.notify(err, type="warning", timeout=3000)
                else:
                    ui.notify(f"Stopped {label}", type="info")
            finally:
                operation_state["busy"] = False
                if _client_alive(stop_button):
                    stop_button.props(remove="loading")
                refresh_callback(force=True)

        def _start() -> None:
            if operation_state["busy"]:
                return
            if blocked_reason:
                ui.notify(blocked_reason, type="warning", timeout=3500)
                return
            operation_state["busy"] = True
            start_button.disable()
            stop_button.disable()
            start_button.props("loading")
            asyncio.create_task(_start_async())

        def _stop() -> None:
            if operation_state["busy"]:
                return
            operation_state["busy"] = True
            start_button.disable()
            stop_button.disable()
            stop_button.props("loading")
            asyncio.create_task(_stop_async())

        start_button = ui.button(
            "Repair Hardware Stack" if repair_needed else "Start",
            on_click=_start,
            icon="build" if repair_needed else "play_arrow",
        ).props(
            "flat dense" + (" disable" if start_blocked else "")
        ).classes("text-green-600")
        stop_button = ui.button("Stop", on_click=_stop, icon="stop").props(
            "flat dense" + (" disable" if stop_disabled else "")
        ).classes("text-red-600")


# =====================================================================
# Digital Twin Launch Section
# =====================================================================
# Friendly display labels for the internal sim-mode keys.
_DT_MODE_LABELS = {"monitor": "Monitor", "teach": "Teach"}


def _digital_twin_launch_section(bridge: SystemBridge) -> None:
    with ui.card().classes("w-full"):
        ui.label("digital twin launch").classes("text-lg font-semibold mb-1")
        ui.label(
            "Monitor = sim mirrors the live robot (hardware drives gazebo). "
            "Use Robot Functions to run exact functions and record required positions."
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

        async def _refresh_async(*, force: bool = False) -> None:
            if not _client_alive(container):
                return
            if refresh_state["busy"]:
                return
            refresh_state["busy"] = True
            try:
                rows = await asyncio.to_thread(bridge.digital_twin_statuses)
                if not _client_alive(container):
                    return
                signature = _signature(rows)
                if not force and signature == refresh_state["signature"]:
                    return
                refresh_state["signature"] = signature
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
                        _digital_twin_row(
                            bridge,
                            target,
                            row,
                            _refresh,
                            active_target=active_target,
                        )
            finally:
                refresh_state["busy"] = False

        def _refresh(*, force: bool = False) -> None:
            asyncio.create_task(_refresh_async(force=force))

        _refresh(force=True)
        ui.timer(3.0, _refresh_async)


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


def _hardware_stack_robot_function_target(bridge: SystemBridge) -> str:
    for stack, target in (
        ("dual robots", "dual robots"),
        ("xarm6", "xarm only"),
        ("ur5e", "ur5e only"),
    ):
        status = bridge.hardware_stack_status(stack)
        selected_stack = str(status.get("selected_stack") or "")
        lifecycle_state = str(status.get("lifecycle_state") or "")
        if selected_stack == stack and lifecycle_state in {
            "starting",
            "running",
            "stopping",
            "failed",
        }:
            return target
        if not selected_stack and str(status.get("overall") or "") == "running":
            return target
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

            # Function execution and position recording live in Robot Functions.


def _function_record_panel(
    bridge: SystemBridge,
    refresh_callbacks: dict[str, Callable[[], None]],
) -> None:
    rows = bridge.digital_twin_statuses()
    targets = [target for target, row in rows.items() if bool(row.get("supported", False))]
    if not targets:
        return
    active_target = _digital_twin_active_target(rows)
    hardware_target = _hardware_stack_robot_function_target(bridge)
    if hardware_target:
        active_target = hardware_target
    initial_target = (
        active_target
        if active_target in targets
        else ("dual robots" if "dual robots" in targets else targets[0])
    )

    with ui.card().classes("w-full"):
        ui.label("Robot Functions").classes("text-lg font-semibold mb-1")
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
        refresh_callbacks["robot_functions"] = _refresh_body

        active_target_refresh: dict[str, object] = {
            "busy": False,
            "hardware_signature": (),
        }

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
                current_hardware_target = await asyncio.to_thread(
                    _hardware_stack_robot_function_target,
                    bridge,
                )
                if current_hardware_target:
                    current_active = current_hardware_target
                hardware_signature: tuple[object, ...] = ()
                if current_hardware_target:
                    stack = {
                        "xarm only": "xarm6",
                        "ur5e only": "ur5e",
                        "dual robots": "dual robots",
                    }[current_hardware_target]
                    stack_status = await asyncio.to_thread(
                        bridge.hardware_stack_status,
                        stack,
                    )
                    hardware_signature = (
                        current_hardware_target,
                        str(stack_status.get("lifecycle_state") or ""),
                        str(stack_status.get("overall") or ""),
                        str(stack_status.get("last_error") or ""),
                    )
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
                lifecycle_changed = (
                    hardware_signature
                    != active_target_refresh.get("hardware_signature")
                )
                active_target_refresh["hardware_signature"] = hardware_signature
                if changed or lifecycle_changed:
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
    execution: dict[str, object] = {
        "busy": False,
        "checking": False,
        "preparing": False,
        "active_function": "",
        "selection_revision": 0,
    }
    pending_execution: dict[str, str] = {}

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
        origin_select = (
            ui.select([], label="origin_resource_location").props("dense").classes("w-60")
        )
        destination_select = (
            ui.select([], label="destination_location").props("dense").classes("w-60")
        )
        part_select = ui.select([], label="part_name").props("dense").classes("w-36")

    ui.label(
        "The function and step IDs come from the Python robot task registry. "
        "Run executes only the exact selected function. This is manual commissioning and does "
        "not require the assembly sequence resource_state or advance ProductAgent/CCA workflow "
        "state. Held-part, gripper, task-context, readiness, and confirmation checks still "
        "apply. Start System is not required for manual Function Execution."
    ).classes("text-xs text-slate-500")

    ui.label("Function Execution").classes("text-sm font-semibold mt-2")
    assembly_board_v1_state: dict[str, object] = {
        "refreshing": False,
        "status_loaded": False,
        "status": {},
        "named_position_exists": False,
        "last_failure": "",
        "selection": (),
    }
    with ui.card().classes("w-full p-3") as assembly_board_v1_panel:
        ui.label("assembly_board-v1 Board Readiness").classes("text-sm font-semibold")
        assembly_board_v1_named_position = ui.label("").classes("text-xs")
        with ui.row().classes("items-center gap-1 w-full"):
            assembly_board_v1_warning_icon = ui.icon("warning", color="red")
            assembly_board_v1_status = ui.label("").classes("text-xs font-semibold")
        assembly_board_v1_acceptance = ui.label("").classes("text-xs text-slate-600")
        assembly_board_v1_calibration = ui.label("").classes("text-xs text-slate-600")
        assembly_board_v1_observation = ui.label("").classes("text-xs text-slate-600")
        assembly_board_v1_movement = ui.label("").classes("text-xs text-slate-600")
        assembly_board_v1_failure = ui.label("").classes("text-xs text-red-700")
        assembly_board_v1_insert_guidance = ui.label(
            "Using the assembly_board-v1 pose frozen by place_approach. Do not move or "
            "re-accept the board."
        ).classes("text-xs font-semibold text-amber-700")
        assembly_board_v1_automatic_status = ui.label("").classes(
            "text-xs font-semibold text-blue-700"
        )
        ui.label(
            "Confirmed Run place_approach first moves to the saved assembly_board-v1 "
            "observation position, collects ten fresh ArUco ID 70 observations, and "
            "automatically accepts or reaccepts the board before any Cartesian approach."
        ).classes("text-xs text-slate-500")
    assembly_board_v1_panel.set_visibility(False)

    execution_status = ui.label("Run readiness is checked automatically without motion.").classes(
        "text-xs text-slate-500"
    )
    execution_blocker = ui.label("").classes("text-xs text-amber-700")
    execution_blocker.set_visibility(False)
    execution_controls = ui.row().classes("items-center gap-2 w-full flex-wrap")

    ui.label("Function Definition").classes("text-sm font-semibold mt-2")
    definition_summary = ui.label("").classes("text-xs text-slate-500")
    definition_container = ui.column().classes("w-full gap-2")

    recording_container = ui.column().classes("w-full gap-2 mt-2")
    with recording_container:
        ui.label("Position Recording").classes("text-sm font-semibold")
        recording_summary = ui.label("").classes("text-xs font-semibold")
        recording_info = ui.label("").classes("text-xs text-blue-700 font-semibold")
        recording_steps = ui.column().classes("w-full gap-2")

    def _current_robot() -> str:
        return str(robot_select.value or "").strip()

    def _current_function() -> str:
        return str(function_select.value or "").strip()

    def _current_origin_resource_location() -> str:
        return str(origin_select.value or "").strip()

    def _current_destination_location() -> str:
        return str(destination_select.value or "").strip()

    def _current_recording_name() -> str:
        location_argument = bridge.digital_twin_function_location_argument(_current_function())
        if location_argument == "origin_resource_location":
            return _current_origin_resource_location()
        if location_argument == "destination_location":
            return _current_destination_location()
        return "default"

    def _current_part_name() -> str:
        return str(part_select.value or "").strip()

    def _assembly_board_v1_selected() -> bool:
        return bool(
            _current_function() in {"place_approach", "place_insert"}
            and _current_destination_location() == "assembly_board-v1"
        )

    def _assembly_board_v1_selection() -> tuple[str, str, str]:
        return (
            _current_robot(),
            _current_function(),
            _current_destination_location(),
        )

    def _assembly_board_v1_accepted_usable() -> bool:
        if (
            not _assembly_board_v1_selected()
            or not assembly_board_v1_state.get("status_loaded")
            or assembly_board_v1_state.get("selection") != _assembly_board_v1_selection()
        ):
            return False
        status = dict(assembly_board_v1_state.get("status") or {})
        return bool(_assembly_board_v1_readiness(status)["usable"])

    def _render_assembly_board_v1_readiness() -> None:
        selected = _assembly_board_v1_selected()
        assembly_board_v1_panel.set_visibility(selected)
        if not selected:
            return
        robot = _current_robot()
        function_name = _current_function()
        state_matches = bool(
            assembly_board_v1_state.get("status_loaded")
            and assembly_board_v1_state.get("selection") == _assembly_board_v1_selection()
        )
        status = dict(assembly_board_v1_state.get("status") or {}) if state_matches else {}
        named_position_exists = bool(
            state_matches and assembly_board_v1_state.get("named_position_exists")
        )
        readiness = _assembly_board_v1_readiness(status)
        assembly_board_v1_named_position.set_text(
            (
                f"Named position assembly_board-v1 exists for {robot}."
                if named_position_exists
                else f"Named position assembly_board-v1 is missing for {robot}."
            )
            if state_matches
            else f"Checking named position assembly_board-v1 for {robot}..."
        )
        assembly_board_v1_named_position.classes(
            replace=(
                "text-xs text-green-700"
                if named_position_exists
                else "text-xs text-red-700"
                if state_matches
                else "text-xs text-slate-500"
            )
        )
        if state_matches:
            level = str(readiness["level"])
            status_classes = {
                "green": "text-xs font-semibold text-green-700",
                "amber": "text-xs font-semibold text-amber-700",
                "red": "text-xs font-semibold text-red-700",
            }[level]
            assembly_board_v1_status.set_text(str(readiness["message"]))
            assembly_board_v1_status.classes(replace=status_classes)
            assembly_board_v1_warning_icon.set_visibility(
                level == "red" or not named_position_exists
            )
        else:
            assembly_board_v1_status.set_text(
                f"Checking the accepted assembly_board-v1 baseline for {robot}..."
            )
            assembly_board_v1_status.classes(replace="text-xs font-semibold text-slate-500")
            assembly_board_v1_warning_icon.set_visibility(False)

        accepted_generation = int(status.get("accepted_generation", 0) or 0)
        accepted_at = status.get("accepted_at")
        accepted_at_text = "not available"
        if accepted_at is not None:
            try:
                accepted_at_text = (
                    datetime.fromtimestamp(float(accepted_at))
                    .astimezone()
                    .strftime("%Y-%m-%d %H:%M:%S %Z")
                )
            except (OSError, OverflowError, TypeError, ValueError):
                accepted_at_text = str(accepted_at)
        assembly_board_v1_acceptance.set_text(
            f"Camera role: {robot} | accepted: {bool(status.get('accepted'))} | "
            f"generation: {accepted_generation} | accepted at: {accepted_at_text}"
        )
        active_calibration_id = str(
            status.get("active_calibration_id")
            or status.get("calibration_id")
            or "unavailable"
        )
        accepted_calibration_id = str(status.get("accepted_calibration_id") or "unavailable")
        assembly_board_v1_calibration.set_text(
            f"Calibration: active={active_calibration_id} | accepted={accepted_calibration_id}"
        )
        sample_count = int(status.get("sample_count", 0) or 0)
        required_sample_count = int(status.get("required_sample_count", 10) or 10)
        tag_currently_visible = _assembly_board_v1_tag_currently_visible(status)
        assembly_board_v1_observation.set_text(
            f"ArUco ID 70 currently visible: {tag_currently_visible} | "
            f"stable: {bool(status.get('stable'))} | "
            f"samples: {sample_count}/{required_sample_count}"
        )
        translation_delta_m = status.get("translation_delta_m")
        rotation_delta_deg = status.get("rotation_delta_deg")
        if (
            status.get("movement_blocked")
            and not status.get("calibration_changed")
            and not status.get("movement_evidence_valid")
        ):
            movement_text = (
                "A prior valid observation proved movement beyond 10 mm or 2 deg; "
                "confirmed Run place_approach will verify it again from the saved "
                "observation position before any board approach."
            )
        elif not tag_currently_visible:
            movement_text = (
                "Movement is unknown without a fresh ArUco ID 70 observation; this is not "
                "evidence that the board moved."
            )
        elif translation_delta_m is None or rotation_delta_deg is None:
            movement_text = "Movement from the accepted board pose is not available."
        else:
            movement_text = (
                f"Movement from accepted pose: {float(translation_delta_m) * 1000.0:.2f} mm, "
                f"{float(rotation_delta_deg):.2f} deg (limits: 10 mm, 2 deg)."
            )
        assembly_board_v1_movement.set_text(movement_text)
        movement_is_blocked = bool(status.get("movement_blocked"))
        assembly_board_v1_movement.classes(
            replace=("text-xs text-red-700" if movement_is_blocked else "text-xs text-slate-600")
        )

        accepted_usable = bool(readiness["usable"])
        last_failure = str(assembly_board_v1_state.get("last_failure") or "").strip()
        post_staging_acceptance_allowed = bool(
            status.get("post_staging_acceptance_allowed")
        )
        if function_name == "place_approach" and not accepted_usable and last_failure:
            failure_text = last_failure
        elif (
            function_name == "place_approach"
            and not accepted_usable
            and post_staging_acceptance_allowed
        ):
            failure_text = (
                "Capture Pose remains blocked. Run place_approach will move to the saved "
                "assembly_board-v1 observation position and obtain the required fresh "
                "10-frame observation automatically."
            )
        elif function_name == "place_approach" and not accepted_usable:
            failure_text = (
                "Confirmed Run place_approach is blocked: "
                + str(readiness["message"])
            )
        else:
            failure_text = ""
        assembly_board_v1_failure.set_text(failure_text)
        assembly_board_v1_failure.classes(
            replace=(
                "text-xs text-amber-700"
                if bool(readiness["usable"])
                else "text-xs text-red-700"
            )
        )
        assembly_board_v1_failure.set_visibility(bool(failure_text))
        assembly_board_v1_insert_guidance.set_visibility(function_name == "place_insert")
        if function_name == "place_insert":
            automatic_text = ""
        elif accepted_usable:
            automatic_text = (
                "The accepted board baseline is available for Capture Pose. Confirmed Run "
                "place_approach will still move to assembly_board-v1, collect ten fresh "
                "post-motion observations, and freeze the resulting board pose."
            )
        elif post_staging_acceptance_allowed:
            automatic_text = (
                "No manual camera staging is required. Confirmed Run place_approach will "
                "stage at assembly_board-v1, collect ten fresh post-motion observations, "
                "and automatically accept or reaccept the board before approaching it."
            )
        else:
            automatic_text = (
                "Confirmed Run place_approach cannot accept the board automatically until "
                "the calibration or configuration warning above is resolved."
            )
        assembly_board_v1_automatic_status.set_text(automatic_text)
        assembly_board_v1_automatic_status.set_visibility(bool(automatic_text))

    async def _refresh_assembly_board_v1_readiness() -> None:
        if not _client_alive(assembly_board_v1_panel):
            return
        if not _assembly_board_v1_selected():
            _render_assembly_board_v1_readiness()
            return
        if assembly_board_v1_state.get("refreshing"):
            return
        selection = _assembly_board_v1_selection()
        robot = selection[0]
        old_gate = (
            _assembly_board_v1_accepted_usable(),
            bool(assembly_board_v1_state.get("named_position_exists")),
        )
        assembly_board_v1_state["refreshing"] = True
        try:
            status_result, named_positions = await asyncio.gather(
                asyncio.to_thread(
                    bridge.perception_assembly_board_v1_aruco_status,
                    robot,
                ),
                asyncio.to_thread(bridge.list_named_positions, robot),
            )
            if selection != _assembly_board_v1_selection():
                return
            assembly_board_v1_state.update(
                {
                    "status_loaded": True,
                    "status": dict(status_result),
                    "named_position_exists": "assembly_board-v1" in named_positions,
                    "selection": selection,
                }
            )
            status = dict(status_result)
            if bool(_assembly_board_v1_readiness(status)["usable"]):
                assembly_board_v1_state["last_failure"] = ""
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if selection != _assembly_board_v1_selection():
                return
            assembly_board_v1_state.update(
                {
                    "status_loaded": False,
                    "status": {},
                    "named_position_exists": False,
                    "last_failure": str(exc),
                    "selection": selection,
                }
            )
        finally:
            assembly_board_v1_state["refreshing"] = False
        if not _client_alive(assembly_board_v1_panel):
            return
        _render_assembly_board_v1_readiness()
        new_gate = (
            _assembly_board_v1_accepted_usable(),
            bool(assembly_board_v1_state.get("named_position_exists")),
        )
        if old_gate != new_gate:
            _render_execution()
            _render_steps()

    def _default_location(function_name: str, options: list[str]) -> str:
        if function_name in {"pick_approach", "pick_grasp"} and "prusa-mk4-2" in options:
            return "prusa-mk4-2"
        if function_name in {"place_approach", "place_insert"} and "assembly_board-v1" in options:
            return "assembly_board-v1"
        return options[0] if options else ""

    def _sync_location() -> None:
        function_name = _current_function()
        location_argument = bridge.digital_twin_function_location_argument(function_name)
        options = bridge.digital_twin_function_location_options(
            _current_robot(),
            function_name,
        )
        selected = _default_location(function_name, options)
        origin_select.options = options if location_argument == "origin_resource_location" else []
        if origin_select.value not in origin_select.options:
            origin_select.value = (
                selected if location_argument == "origin_resource_location" else ""
            )
        origin_select.set_visibility(location_argument == "origin_resource_location")
        origin_select.update()
        destination_select.options = options if location_argument == "destination_location" else []
        if destination_select.value not in destination_select.options:
            destination_select.value = (
                selected if location_argument == "destination_location" else ""
            )
        destination_select.set_visibility(location_argument == "destination_location")
        destination_select.update()

    def _sync_part_name() -> None:
        function_name = _current_function()
        visible = function_name in {
            "pick_approach",
            "pick_grasp",
            "place_approach",
            "place_insert",
        }
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

    def _execution_kwargs() -> dict[str, str]:
        return {
            "origin_resource_location": _current_origin_resource_location(),
            "destination_location": _current_destination_location(),
            "part_name": _current_part_name(),
        }

    def _selection_signature() -> tuple[str, str, str, str, str]:
        return (
            _current_robot(),
            _current_function(),
            _current_origin_resource_location(),
            _current_destination_location(),
            _current_part_name(),
        )

    def _confirmation_description(function_name: str, values: dict[str, str]) -> str:
        robot = _current_robot()
        part_name = values.get("part_name", "")
        if function_name == "pick_approach":
            if robot == "ur5e" and part_name == "MG":
                return (
                    f"The UR5e will stage at {values.get('origin_resource_location', '')}, "
                    "request a fresh MG detection, open the stock RG2, and descend to the "
                    "smooth raised hub target calculated from the actual Gear_Medium.STL. "
                    "The gripper remains open for visual confirmation; the Gazebo MG is not "
                    "used for physical geometry."
                )
            return (
                f"The {robot} will stage at {values.get('origin_resource_location', '')}, request "
                f"a fresh {part_name} detection, open the gripper, move above the detected "
                "gear, and descend to the computed pick target."
            )
        if function_name == "pick_grasp":
            if robot == "ur5e" and part_name == "MG":
                return (
                    "The stock RG2 will close at the STL-calculated width around the MG "
                    "smooth raised hub and then lift. It will not descend again."
                )
            return f"The {robot} will grasp and lift {part_name}."
        if function_name == "place_approach":
            destination_location = values.get("destination_location", "")
            if destination_location == "assembly_board-v1":
                return (
                    f"The {robot} will first move {part_name} to the saved "
                    "assembly_board-v1 observation position, collect ten fresh ArUco ID 70 "
                    "observations, and automatically accept or reaccept the current board "
                    "pose if needed. It will then use that frozen pose for the saved approach "
                    "and descend positions. If localization fails, it remains at the "
                    "observation position and does not approach the board."
                )
            return (
                f"The {robot} will move {part_name} to the saved approach and descend positions "
                f"for {destination_location}."
            )
        if function_name == "place_insert":
            return (
                f"The {robot} will release {part_name} at "
                f"{values.get('destination_location', '')} and lift away. "
                "Releasing the part is irreversible."
            )
        return f"The {robot} will move to the configured home named position."

    with execution_controls:
        with ui.dialog() as run_confirm, ui.card().classes("gap-3 max-w-xl"):
            run_confirm_title = ui.label("").classes("font-semibold")
            run_confirm_text = ui.label("").classes("text-sm")
            ui.label(
                "This commands physical robot motion. Clear the workcell and switch the "
                "pendant to Remote Control before continuing."
            ).classes("text-xs text-red-700")
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("Cancel", on_click=run_confirm.close).props("flat")

                async def _confirmed_run() -> None:
                    client = _current_client()
                    run_confirm.close()
                    if execution.get("busy") or not pending_execution:
                        _notify(
                            "Run readiness must complete before confirmation.",
                            type="warning",
                            client=client,
                        )
                        return
                    values = dict(pending_execution)
                    function_name = values.pop("function_name")
                    robot = values.pop("robot")
                    execution.update({"busy": True, "active_function": function_name})
                    run_button.props("loading")
                    _render_execution()
                    _render_steps()
                    if function_name == "pick_approach":
                        if robot == "ur5e":
                            execution_status.set_text(
                                "Dispatching pick_approach after fresh RTDE, world -> tool0, "
                                "and perception checks; then staging, settling, and detecting."
                            )
                        else:
                            execution_status.set_text(
                                "Dispatching pick_approach after fresh xarm6 hardware, "
                                "world -> link_eef, and perception checks; then staging, "
                                "settling, and detecting."
                            )
                    else:
                        execution_status.set_text(
                            f"Dispatching {function_name} after fresh physical readiness checks."
                        )
                    execution_status.classes(replace="text-xs text-amber-700")
                    try:
                        result = await bridge.digital_twin_execute_robot_function(
                            target,
                            robot,
                            function_name,
                            origin_resource_location=values["origin_resource_location"],
                            destination_location=values["destination_location"],
                            part_name=values["part_name"],
                            confirmed=True,
                        )
                        message = str(result.get("message") or f"{function_name} completed.")
                        if _client_alive(execution_status):
                            execution_status.set_text(message)
                            execution_status.classes(
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
                        log.exception("%s UI execution failed", function_name)
                        message = f"{function_name} failed: {exc}"
                        if _client_alive(execution_status):
                            execution_status.set_text(message)
                            execution_status.classes(replace="text-xs text-red-700")
                        _notify(message, type="negative", timeout=7000, client=client)
                    finally:
                        pending_execution.clear()
                        execution.update({"busy": False, "active_function": ""})
                        if (
                            function_name == "place_approach"
                            and values.get("destination_location") == "assembly_board-v1"
                            and _client_alive(assembly_board_v1_panel)
                        ):
                            await _refresh_assembly_board_v1_readiness()
                        if _client_alive(run_button):
                            run_button.props(remove="loading")
                            _render_execution()
                            _render_steps()

                confirm_run_button = ui.button(
                    "Confirm Run",
                    on_click=_confirmed_run,
                    icon="play_arrow",
                ).props("color=red")

        async def _check_run_readiness() -> None:
            client = _current_client()
            if execution.get("busy") or execution.get("checking"):
                _notify("A robot function check or execution is already active.", type="warning")
                return
            function_name = _current_function()
            robot = _current_robot()
            values = _execution_kwargs()
            selection_signature = _selection_signature()
            selection_revision = int(execution["selection_revision"])
            execution.update({"checking": True, "active_function": function_name})
            pending_execution.clear()
            run_button.props("loading")
            _render_execution()
            _render_steps()
            if robot == "ur5e":
                execution_status.set_text(
                    "Checking fresh RTDE, world -> tool0, and perception readiness..."
                )
            else:
                execution_status.set_text(f"Checking {function_name} readiness without motion...")
            execution_status.classes(replace="text-xs text-amber-700")
            try:
                result = await bridge.digital_twin_robot_function_execution_readiness(
                    target,
                    robot,
                    function_name,
                    origin_resource_location=values["origin_resource_location"],
                    destination_location=values["destination_location"],
                    part_name=values["part_name"],
                )
                if not _client_alive(execution_status):
                    return
                if (
                    selection_revision != int(execution["selection_revision"])
                    or selection_signature != _selection_signature()
                ):
                    _notify(
                        "Selection changed during readiness; the old result was discarded.",
                        type="warning",
                        client=client,
                    )
                    return
                message = str(result.get("message") or "")
                execution_status.set_text(message)
                execution_status.classes(
                    replace=(
                        "text-xs text-green-700"
                        if result.get("success")
                        else "text-xs text-red-700"
                    )
                )
                if not result.get("success"):
                    _notify(message, type="warning", timeout=7000, client=client)
                    return
                pending_execution.update(
                    {
                        "function_name": function_name,
                        "robot": robot,
                        **values,
                    }
                )
                run_confirm_title.set_text(f"Run {function_name} on the physical {robot}?")
                run_confirm_text.set_text(_confirmation_description(function_name, values))
                confirm_run_button.set_text(f"Confirm Run {function_name}")
                run_confirm.open()
            except Exception as exc:
                log.exception("%s readiness check failed", function_name)
                message = f"{function_name} readiness check failed: {exc}"
                if _client_alive(execution_status):
                    execution_status.set_text(message)
                    execution_status.classes(replace="text-xs text-red-700")
                _notify(message, type="negative", timeout=7000, client=client)
            finally:
                execution.update({"checking": False, "active_function": ""})
                if _client_alive(run_button):
                    run_button.props(remove="loading")
                    _render_execution()
                    _render_steps()

        run_button = (
            ui.button("Run", on_click=_check_run_readiness, icon="play_arrow")
            .props("outline dense")
            .classes("text-red-600")
        )

        with ui.dialog() as gripper_close_test_confirm, ui.card().classes("gap-3 max-w-xl"):
            gripper_close_test_title = ui.label("").classes("font-semibold")
            gripper_close_test_text = ui.label("").classes("text-sm")
            ui.label(
                "Clear the gripper area and keep the UR5e at the completed pick_approach pose."
            ).classes("text-xs text-red-700")
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("Cancel", on_click=gripper_close_test_confirm.close).props("flat")

                async def _confirmed_gripper_close_test() -> None:
                    client = _current_client()
                    gripper_close_test_confirm.close()
                    if execution.get("busy") or execution.get("checking"):
                        _notify(
                            "A robot function check or execution is already active.",
                            type="warning",
                            client=client,
                        )
                        return
                    part_name = _current_part_name()
                    execution.update({"busy": True, "active_function": "Gripper Close Test"})
                    gripper_close_test_button.props("loading")
                    _render_execution()
                    _render_steps()
                    execution_status.set_text(
                        "Checking fresh physical readiness, then closing the RG2 without "
                        "moving the arm."
                    )
                    execution_status.classes(replace="text-xs text-amber-700")
                    try:
                        result = await bridge.digital_twin_execute_gripper_close_test(
                            target,
                            _current_robot(),
                            _current_origin_resource_location(),
                            part_name,
                            confirmed=True,
                        )
                        message = str(
                            result.get("message") or f"{part_name} Gripper Close Test completed."
                        )
                        if _client_alive(execution_status):
                            execution_status.set_text(message)
                            execution_status.classes(
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
                        log.exception("Gripper Close Test UI execution failed")
                        message = f"Gripper Close Test failed: {exc}"
                        if _client_alive(execution_status):
                            execution_status.set_text(message)
                            execution_status.classes(replace="text-xs text-red-700")
                        _notify(message, type="negative", timeout=7000, client=client)
                    finally:
                        execution.update({"busy": False, "active_function": ""})
                        if _client_alive(gripper_close_test_button):
                            gripper_close_test_button.props(remove="loading")
                            _render_execution()
                            _render_steps()

                ui.button(
                    "Confirm Gripper Close Test",
                    on_click=_confirmed_gripper_close_test,
                    icon="compare_arrows",
                ).props("color=red")

        def _open_gripper_close_test_confirm() -> None:
            if execution.get("busy") or execution.get("checking"):
                _notify("A robot function check or execution is already active.", type="warning")
                return
            part_name = _current_part_name()
            gripper_close_test_title.set_text(
                f"Run {part_name} Gripper Close Test on the physical ur5e?"
            )
            if part_name == "MG":
                gripper_close_test_text.set_text(
                    "The arm will not move. The RG2 will close once to the retained MG "
                    "position (approximately 0.047), hold for three seconds, and reopen in "
                    "all cases. The stock fingertips move downward approximately 26.16 mm "
                    "at this MG closing width."
                )
            else:
                gripper_close_test_text.set_text(
                    f"The arm will not move. The RG2 will close once to the retained "
                    f"{part_name} gripper_close_position from pick_approach, hold for three "
                    "seconds, and reopen in all cases."
                )
            gripper_close_test_confirm.open()

        gripper_close_test_button = (
            ui.button(
                "Gripper Close Test",
                on_click=_open_gripper_close_test_confirm,
                icon="compare_arrows",
            )
            .props("outline dense")
            .classes("text-amber-700")
        )
        gripper_close_test_help = ui.label(
            "Gripper Close Test only: uses the selected part's retained "
            "gripper_close_position. It never moves or lifts the arm and always attempts "
            "to reopen the RG2."
        ).classes("text-xs text-amber-700")

    def _render_execution() -> None:
        function_name = _current_function()
        run_button.set_text(f"Run {function_name}" if function_name else "Run")
        controls_enabled = (
            not execution.get("busy")
            and not execution.get("checking")
            and not execution.get("preparing")
        )
        for selector in (
            robot_select,
            function_select,
            origin_select,
            destination_select,
            part_select,
        ):
            selector.set_enabled(bool(controls_enabled))
        needs_part = function_name in {
            "pick_approach",
            "pick_grasp",
            "place_approach",
            "place_insert",
        }
        location_argument = bridge.digital_twin_function_location_argument(function_name)
        has_location = (
            bool(_current_origin_resource_location())
            if location_argument == "origin_resource_location"
            else bool(_current_destination_location())
            if location_argument == "destination_location"
            else True
        )
        blocker = ""
        hardware_target = _hardware_stack_robot_function_target(bridge)
        if hardware_target:
            stack = {
                "xarm only": "xarm6",
                "ur5e only": "ur5e",
                "dual robots": "dual robots",
            }[hardware_target]
            stack_status = bridge.hardware_stack_status(stack)
            if hardware_target != target:
                blocker = (
                    f"Selected Function Execution target is {target}, but "
                    f"{hardware_target} owns the Hardware Stack."
                )
            elif str(stack_status.get("lifecycle_state") or "") != "running":
                blocker = str(
                    stack_status.get("last_error")
                    or f"{stack} Hardware Stack lifecycle is "
                    f"{stack_status.get('lifecycle_state') or 'stopped'}."
                )
            elif str(stack_status.get("overall") or "") != "running":
                blocker = f"{stack} Hardware Stack components are not all running."
            else:
                robot_status = (
                    dict(stack_status.get(_current_robot()) or {})
                    if stack == "dual robots"
                    else stack_status
                )
                if robot_status.get("cartesian_function_ready") is False:
                    reason = str(
                        robot_status.get("cartesian_readiness_message")
                        or "Cartesian frame validation has not completed"
                    )
                    blocker = (
                        reason
                        if reason.startswith("Cartesian frame validation failed:")
                        else f"Cartesian frame validation failed: {reason}"
                    )
                elif (
                    _current_robot() == "ur5e"
                    and str(robot_status.get("gripper_action") or "") != "ready"
                ):
                    blocker = "UR5e RG2 gripper action is not ready."
        teleop_readiness = bridge.teleop_cartesian_readiness(_current_robot())
        if teleop_readiness.get("smooth_hold_active"):
            blocker = "Release Cartesian Smooth Hold before Function Execution."
        board_run_blocked = False
        if (
            function_name == "place_approach"
            and _current_destination_location() == "assembly_board-v1"
        ):
            board_status = dict(assembly_board_v1_state.get("status") or {})
            post_staging_acceptance_allowed = bool(
                board_status.get("post_staging_acceptance_allowed")
            )
            board_run_blocked = bool(
                not assembly_board_v1_state.get("named_position_exists")
                or (
                    not _assembly_board_v1_accepted_usable()
                    and not post_staging_acceptance_allowed
                )
            )
            if board_run_blocked:
                if not assembly_board_v1_state.get("status_loaded"):
                    board_blocker = (
                        "Checking assembly_board-v1 Board Readiness before place_approach."
                    )
                elif not assembly_board_v1_state.get("named_position_exists"):
                    board_blocker = (
                        f"Save the assembly_board-v1 named position for {_current_robot()} "
                        "before Run place_approach."
                    )
                else:
                    board_blocker = str(
                        _assembly_board_v1_readiness(board_status)["message"]
                    )
                blocker = f"{blocker} {board_blocker}".strip()
        execution_blocker.set_text(blocker)
        execution_blocker.set_visibility(bool(blocker))
        enabled = bool(
            _current_robot() in {"xarm6", "ur5e"}
            and function_name
            and has_location
            and (not needs_part or _current_part_name())
            and controls_enabled
            and not board_run_blocked
        )
        run_button.set_enabled(enabled)
        show_gripper_close_test = bool(
            _current_robot() == "ur5e"
            and bool(_current_part_name())
            and function_name in {"pick_approach", "pick_grasp"}
        )
        gripper_close_test_button.set_visibility(show_gripper_close_test)
        gripper_close_test_help.set_visibility(show_gripper_close_test)
        gripper_close_test_button.set_enabled(
            bool(
                show_gripper_close_test
                and has_location
                and controls_enabled
            )
        )
        _render_assembly_board_v1_readiness()

    async def _capture_position(step_name: str, primitive: str) -> None:
        client = _current_client()
        if execution.get("busy") or execution.get("checking"):
            _notify("A robot function check or execution is already active.", type="warning")
            return
        robot = _current_robot()
        function_name = _current_function()
        recording_name = _current_recording_name()
        part_name = _current_part_name()
        execution.update(
            {
                "checking": True,
                "active_function": f"{function_name}.{step_name} Capture Pose",
            }
        )
        _render_execution()
        _render_steps()
        try:
            preparation = await bridge.digital_twin_prepare_function_capture(
                target,
                robot,
            )
            if not preparation.get("success"):
                result = preparation
                _notify(
                    str(result.get("message") or ""),
                    type="warning",
                    timeout=6000,
                    client=client,
                )
                return
            result = await asyncio.to_thread(
                bridge.digital_twin_capture_function_step,
                target,
                robot,
                function_name,
                recording_name,
                step_name,
                primitive,
                part_name=part_name,
            )
            _notify(
                str(result.get("message") or ""),
                type="positive" if result.get("success") else "warning",
                timeout=5000,
                client=client,
            )
        finally:
            execution.update({"checking": False, "active_function": ""})
            if _client_alive(run_button):
                _render_execution()
                _render_steps()

    async def _save_position(step_name: str) -> None:
        client = _current_client()
        if execution.get("busy") or execution.get("checking"):
            _notify("A robot function check or execution is already active.", type="warning")
            return
        robot = _current_robot()
        function_name = _current_function()
        recording_name = _current_recording_name()
        part_name = _current_part_name()
        execution.update(
            {
                "checking": True,
                "active_function": f"{function_name}.{step_name} Save/Replace Pose",
            }
        )
        _render_execution()
        _render_steps()
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_save_function_position,
                target,
                robot,
                function_name,
                recording_name,
                step_name,
                part_name=part_name,
            )
            _notify(
                str(result.get("message") or ""),
                type="positive" if result.get("success") else "warning",
                timeout=5000,
                client=client,
            )
        finally:
            execution.update({"checking": False, "active_function": ""})
            if _client_alive(run_button):
                _render_execution()
                _render_steps()

    async def _clear_position(step_name: str) -> None:
        client = _current_client()
        if execution.get("busy") or execution.get("checking"):
            _notify("A robot function check or execution is already active.", type="warning")
            return
        robot = _current_robot()
        function_name = _current_function()
        recording_name = _current_recording_name()
        part_name = _current_part_name()
        execution.update(
            {
                "checking": True,
                "active_function": f"{function_name}.{step_name} Clear Pose",
            }
        )
        _render_execution()
        _render_steps()
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_clear_function_position,
                target,
                robot,
                function_name,
                recording_name,
                step_name,
                part_name=part_name,
            )
            _notify(
                str(result.get("message") or ""),
                type="positive" if result.get("success") else "warning",
                timeout=4500,
                client=client,
            )
        finally:
            execution.update({"checking": False, "active_function": ""})
            if _client_alive(run_button):
                _render_execution()
                _render_steps()

    async def _test_position(
        step_name: str,
        *,
        robot: str,
        function_name: str,
        recording_name: str,
        part_name: str,
        client: Client | None = None,
    ) -> None:
        if execution.get("busy") or execution.get("checking"):
            _notify(
                "A robot function check or execution is already active.",
                type="warning",
                client=client,
            )
            return
        execution.update(
            {
                "busy": True,
                "active_function": f"{function_name}.{step_name} Test Position",
            }
        )
        _render_execution()
        _render_steps()
        try:
            execution_status.set_text(
                f"Preparing the physical {robot} Function Execution runtime; no motion."
            )
            execution_status.classes(replace="text-xs text-amber-700")
            preparation = await bridge.digital_twin_prepare_function_position(
                target,
                robot,
            )
            if not preparation.get("success"):
                message = str(
                    preparation.get("message")
                    or f"Physical {robot} Function Execution preparation failed."
                )
                execution_status.set_text(message)
                execution_status.classes(replace="text-xs text-red-700")
                _notify(message, type="warning", timeout=7000, client=client)
                return
            result = await asyncio.to_thread(
                bridge.digital_twin_test_function_position,
                target,
                robot,
                function_name,
                recording_name,
                step_name,
                confirmed=True,
                part_name=part_name,
            )
            message = str(result.get("message") or "")
            execution_status.set_text(message)
            execution_status.classes(
                replace=(
                    "text-xs text-green-700"
                    if result.get("success")
                    else "text-xs text-red-700"
                )
            )
            _notify(
                message,
                type="positive" if result.get("success") else "warning",
                timeout=6000,
                client=client,
            )
        finally:
            execution.update({"busy": False, "active_function": ""})
            if _client_alive(run_button):
                _render_execution()
                _render_steps()

    def _render_steps() -> None:  # noqa: C901, PLR0912, PLR0915 - operator states stay local.
        function_name = _current_function()
        name = _current_recording_name()
        controls_blocked = bool(
            execution.get("busy")
            or execution.get("checking")
            or execution.get("preparing")
        )
        board_capture_blocked = bool(
            function_name == "place_approach"
            and name == "assembly_board-v1"
            and not _assembly_board_v1_accepted_usable()
        )
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
        recordable_steps = [step for step in template if bool(step.get("recordable"))]
        required_steps = [step for step in recordable_steps if bool(step.get("required"))]
        saved_required = [
            step
            for step in required_steps
            if str(step.get("step_name") or "") in saved_steps
            and saved_steps[str(step.get("step_name") or "")].get("pose")
        ]
        recording_container.set_visibility(bool(recordable_steps))
        definition_summary.set_text(
            f"{len(template)} ordered primitive step{'s' if len(template) != 1 else ''}."
            if template
            else "No function definition is available."
        )
        if required_steps and function_name == "place_approach":
            recording_summary.set_text(
                f"{len(saved_required)}/{len(required_steps)} required positions saved for "
                f"{name} / {_current_part_name()}"
            )
        elif recordable_steps:
            saved_count = sum(
                1
                for step in recordable_steps
                if str(step.get("step_name") or "") in saved_steps
            )
            recording_summary.set_text(
                f"{saved_count}/{len(recordable_steps)} optional Cartesian positions saved"
            )
        info = bridge.digital_twin_function_info(
            target,
            _current_robot(),
            function_name,
            name,
            _current_part_name(),
        )
        recording_info.set_text(
            f"Saving as: {info.get('display_path')}"
            if info.get("success") and recordable_steps
            else ""
        )
        definition_container.clear()
        with definition_container:
            for index, step in enumerate(template, start=1):
                step_name = str(step.get("step_name") or "")
                primitive = str(step.get("primitive") or "")
                recordable = bool(step.get("recordable"))
                position_required = bool(step.get("required"))
                with ui.card().classes("w-full p-3"):
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(str(index)).classes("text-xs font-semibold w-5")
                        ui.label(step_name).classes("text-sm font-semibold")
                        ui.label("→").classes("text-xs text-slate-400")
                        ui.label(primitive).classes("text-sm text-blue-700")
                        ui.space()
                        if not recordable:
                            ui.label("No recorded position required").classes(
                                "text-xs text-slate-500"
                            )
                        elif position_required:
                            ui.label("Recorded position required").classes(
                                "text-xs text-blue-700"
                            )
                        else:
                            ui.label("Optional Cartesian teaching").classes(
                                "text-xs text-amber-700"
                            )
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

        recording_steps.clear()
        with recording_steps:
            for step in recordable_steps:
                step_name = str(step.get("step_name") or "")
                primitive = str(step.get("primitive") or "")
                position_required = bool(step.get("required"))
                saved = saved_steps.get(step_name)
                buffered = buffered_steps.get(step_name)
                position = buffered or saved
                with ui.card().classes("w-full p-3"):
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(step_name).classes("text-sm font-semibold")
                        ui.label("→").classes("text-xs text-slate-400")
                        ui.label(primitive).classes("text-sm text-blue-700")
                        ui.space()
                        if buffered:
                            ui.label("Captured pose; not saved").classes("text-xs text-amber-700")
                        elif saved:
                            ui.label("Pose saved").classes("text-xs text-green-700")
                        elif not position_required:
                            ui.label("Computed live; override optional").classes(
                                "text-xs text-slate-500"
                            )
                        else:
                            ui.label("Position required").classes("text-xs text-red-700")
                    pose = dict(position.get("pose") or {}) if position else {}
                    if pose:
                        ui.label(
                            f"world → {'tool0' if _current_robot() == 'ur5e' else 'link_eef'}: "
                            f"x={float(pose.get('x', 0.0)):.9f}, "
                            f"y={float(pose.get('y', 0.0)):.9f}, "
                            f"z={float(pose.get('z', 0.0)):.9f}, "
                            f"q=({float(pose.get('qx', 0.0)):.9f}, "
                            f"{float(pose.get('qy', 0.0)):.9f}, "
                            f"{float(pose.get('qz', 0.0)):.9f}, "
                            f"{float(pose.get('qw', 1.0)):.9f})"
                        ).classes("text-xs text-slate-600")
                        relative_position = dict(
                            position.get("relative_position_m") or {}
                        )
                        relative_pose = dict(position.get("relative_pose") or {})
                        relative_reference = dict(
                            position.get("relative_reference") or {}
                        )
                        computed_position = dict(
                            position.get("computed_position_m") or {}
                        )
                        if computed_position:
                            ui.label(
                                "Computed pose used for calibration: "
                                f"x={float(computed_position.get('x', 0.0)):.9f}, "
                                f"y={float(computed_position.get('y', 0.0)):.9f}, "
                                f"z={float(computed_position.get('z', 0.0)):.9f} m; "
                                f"source={position.get('computed_source')}, "
                                f"timestamp={position.get('computed_at')}"
                            ).classes("text-xs text-slate-600")
                        if relative_position and relative_reference:
                            ui.label(
                                "Saved calibration XYZ offset: "
                                f"x={float(relative_position.get('x', 0.0)):.9f}, "
                                f"y={float(relative_position.get('y', 0.0)):.9f}, "
                                f"z={float(relative_position.get('z', 0.0)):.9f} m"
                            ).classes("text-xs text-blue-700")
                            reference_position = dict(
                                relative_reference.get("position_m") or {}
                            )
                            ui.label(
                                f"Reference: {relative_reference.get('kind')} "
                                f"{relative_reference.get('name')} in world at "
                                f"x={float(reference_position.get('x', 0.0)):.9f}, "
                                f"y={float(reference_position.get('y', 0.0)):.9f}, "
                                f"z={float(reference_position.get('z', 0.0)):.9f}; "
                                f"source={relative_reference.get('source')}"
                            ).classes("text-xs text-slate-600")
                            if (
                                relative_reference.get("source")
                                == "assembly_board-v1_aruco"
                                and relative_pose
                            ):
                                ui.label(
                                    "Replay composes the current ArUco ID 70 world pose with "
                                    "this saved full relative SE(3) pose, applying board "
                                    "translation and rotation."
                                ).classes("text-xs text-slate-500")
                            else:
                                ui.label(
                                    "Replay adds this saved world-axis XYZ calibration to the "
                                    "current computed pose and "
                                    "keeps the captured quaternion unchanged."
                                ).classes("text-xs text-slate-500")
                    with ui.row().classes("items-center gap-2 mt-1 flex-wrap"):
                        capture_button = ui.button(
                            "Capture Pose",
                            on_click=lambda _e, s=step_name, p=primitive: _capture_position(s, p),
                            icon="fiber_manual_record",
                        ).props("flat dense")
                        if (
                            controls_blocked
                            or board_capture_blocked
                            or not name
                            or (
                                function_name in {"pick_approach", "place_approach"}
                                and not _current_part_name()
                            )
                        ):
                            capture_button.disable()
                        save_button = ui.button(
                            "Save/Replace Pose",
                            on_click=lambda _e, s=step_name: _save_position(s),
                            icon="save",
                        ).props("flat dense")
                        if controls_blocked or not buffered:
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
                        if controls_blocked or (not buffered and not saved):
                            clear_button.disable()

                        test_robot = _current_robot()
                        test_part_name = _current_part_name()
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
                                    robot=test_robot,
                                    function_name=function_name,
                                    recording_name=name,
                                    part_name=test_part_name,
                                    dialog=test_confirm,
                                ) -> None:
                                    client = _current_client()
                                    dialog.close()
                                    await _test_position(
                                        step_name,
                                        robot=robot,
                                        function_name=function_name,
                                        recording_name=recording_name,
                                        part_name=part_name,
                                        client=client,
                                    )

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
                        if controls_blocked or not saved:
                            test_button.disable()

    def _reset_execution_status() -> None:
        message = "Run readiness is checked automatically without motion."
        classes = "text-xs text-slate-500"
        execution_status.set_text(message)
        execution_status.classes(replace=classes)

    def _selection_changed(_e=None) -> None:
        execution["selection_revision"] = int(execution["selection_revision"]) + 1
        pending_execution.clear()
        run_confirm.close()
        assembly_board_v1_state.update(
            {
                "last_failure": "",
            }
        )
        _sync_location()
        _sync_part_name()
        _reset_execution_status()
        _render_execution()
        _render_steps()
        asyncio.create_task(_refresh_assembly_board_v1_readiness())

    def _part_name_changed(_e=None) -> None:
        execution["selection_revision"] = int(execution["selection_revision"]) + 1
        pending_execution.clear()
        run_confirm.close()
        _reset_execution_status()
        _render_execution()
        _render_steps()

    def _location_changed(_e=None) -> None:
        execution["selection_revision"] = int(execution["selection_revision"]) + 1
        pending_execution.clear()
        run_confirm.close()
        assembly_board_v1_state.update(
            {
                "last_failure": "",
            }
        )
        _reset_execution_status()
        _render_execution()
        _render_steps()
        asyncio.create_task(_refresh_assembly_board_v1_readiness())

    robot_select.on_value_change(_selection_changed)
    function_select.on_value_change(_selection_changed)
    origin_select.on_value_change(_location_changed)
    destination_select.on_value_change(_location_changed)
    part_select.on_value_change(_part_name_changed)
    _sync_location()
    _sync_part_name()
    _reset_execution_status()
    _render_execution()
    _render_steps()

    def _refresh_function_execution_progress() -> None:
        if not execution.get("busy"):
            return
        progress = bridge.digital_twin_robot_function_execution_progress()
        if not progress.get("active"):
            return
        message = str(progress.get("message") or "").strip()
        if message and _client_alive(execution_status):
            execution_status.set_text(message)
            execution_status.classes(replace="text-xs text-amber-700")

    ui.timer(0.2, _refresh_function_execution_progress)
    ui.timer(2.0, _refresh_assembly_board_v1_readiness, immediate=True)
    ui.label(
        "Capture checks read-only readiness automatically. Run and Test Position are separate "
        "physical motion actions requiring the selected robot's remote-control mode and "
        "explicit confirmation."
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
_PHYSICAL_CARTESIAN_SPEED_STEP_MM_S = 0.025
_PHYSICAL_JOINT_SPEED_STEP_DEG_S = 0.1


def _teleop_section(
    bridge: SystemBridge,
    refresh_callbacks: dict[str, Callable[[], None]],
) -> None:
    with ui.card().classes("w-full"):
        ui.label("Interactive Teleop").classes("text-lg font-semibold mb-1")
        ui.label(
            "Control robots via on-screen buttons or keyboard. "
            "Works in both Simulation and Physical modes when the environment is running."
        ).classes("text-xs text-slate-500 mb-3")

        rtde_reset_state = {"busy": False}
        named_position_refresh: dict[str, Callable[[], None] | None] = {
            "callback": None,
        }
        with ui.row().classes("items-center gap-4 mb-2 flex-wrap"):
            with ui.row().classes("items-center gap-2"):
                teleop_backend_icon = ui.icon("circle", color="grey").classes("text-xs")
                teleop_backend_label = ui.label("Teleop backend: checking...").classes("text-xs")
            with ui.row().classes("items-center gap-2"):
                teleop_env_icon = ui.icon("circle", color="grey").classes("text-xs")
                teleop_env_label = ui.label("Teleop environment: checking...").classes("text-xs")
            rtde_reset_button = ui.button(
                "Reset UR5e RTDE",
                icon="restart_alt",
            ).props("outline dense")
            rtde_reset_button.disable()
        teleop_warning_label = ui.label("").classes("text-xs text-amber-700")
        rtde_reset_progress_label = ui.label("").classes("text-xs text-amber-700")

        # Robot selector.
        xarm6_initial_target = bridge.teleop_target("xarm6", "state")
        ur5e_initial_target = bridge.teleop_target("ur5e", "state")
        initial_robot = (
            "ur5e"
            if bool(ur5e_initial_target.get("ready"))
            and not bool(xarm6_initial_target.get("ready"))
            else "xarm6"
        )
        selected_robot_state = {"robot": initial_robot, "changing": False}
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.label("Robot:").classes("font-semibold text-sm")
            robot_select = ui.toggle(["xarm6", "ur5e"], value=initial_robot).classes(
                "text-sm"
            )

        mode_state = {"mode": "cartesian"}  # cartesian | gripper | joint
        cartesian_jog_mode = {
            "mode": "step",
            "preparing": False,
            "updating_toggle": False,
        }  # step | smooth
        cartesian_command_state = {"pending": False}
        conflicting_motion_buttons = []
        smooth_hold = {
            "pressed": False,
            "active": False,
            "stopping": False,
            "robot": "",
            "axis": "",
            "speed_mm_s": 0.0,
            "generation": 0,
            "task": None,
        }
        cartesian_jog_buttons = []
        cartesian_jog_controls = {"toggle": None, "readiness": None}
        axis_state = {"axis": "y"}  # x | y | z
        joint_state = {"idx": 1}  # 1..6
        profile_state = {"mode": "fast"}  # precision | fast
        velocity_state = {
            "xarm6": {"arm_vel": 1.0, "gripper_vel": 1.0},
            "ur5e": {"arm_vel": 1.0, "gripper_vel": 1.0},
        }
        motion_settings = {
            robot: bridge.teleop_motion_settings(robot)
            for robot in ("xarm6", "ur5e")
        }
        motion_speed_state = {
            robot: {
                "cartesian_mm_s": float(
                    motion_settings[robot].get("cartesian_speed_default_mm_s", 5.0)
                ),
                "joint_deg_s": float(
                    motion_settings[robot].get("joint_speed_default_deg_s", 0.1)
                ),
            }
            for robot in ("xarm6", "ur5e")
        }
        motion_speed_controls = {
            "title": None,
            "physical": None,
            "simulation": None,
            "cartesian_slider": None,
            "cartesian_number": None,
            "joint_slider": None,
            "joint_number": None,
            "range": None,
            "acceleration": None,
            "applied": None,
            "simulation_scale": None,
        }
        motion_speed_sync = {"busy": False}
        effective_labels = {}
        profile_note = {"label": None}
        step_inputs = {"cartesian": None, "joint": None}
        applied_motion_labels = {"cartesian": None, "joint": None}
        cartesian_status = {"label": None}
        save_env_label = {"label": None}
        state_refresh = {"busy": False}
        state_labels = {"status": None, "xyz": None, "rpy": None, "joints": []}

        def _refresh_rtde_reset_enabled() -> None:
            robot = str(robot_select.value or "xarm6").strip().lower()
            target = bridge.teleop_target(robot, "state")
            source = str(target.get("source") or "")
            enabled = bool(
                robot == "ur5e"
                and str(target.get("environment") or "") == "real"
                and (source == "hardware" or source.startswith("digital_twin:"))
                and not bridge.system_running
                and not bool(getattr(bridge, "_starting", False))
                and not bool(getattr(bridge, "_stopping", False))
                and not rtde_reset_state["busy"]
            )
            rtde_reset_button.set_enabled(enabled)

        async def _reset_ur5e_rtde() -> None:
            if rtde_reset_state["busy"]:
                return
            rtde_reset_state["busy"] = True
            rtde_reset_button.disable()
            rtde_reset_button.props("loading")
            rtde_reset_progress_label.set_text(
                "Resetting the UR5e RTDE connection; no motion."
            )
            try:
                ok, message = await asyncio.to_thread(
                    bridge.teleop_reset_ur5e_rtde_connection
                )
                if not _client_alive(rtde_reset_progress_label):
                    return
                rtde_reset_progress_label.set_text(message)
                rtde_reset_progress_label.classes(
                    replace=(
                        "text-xs text-green-700" if ok else "text-xs text-red-700"
                    )
                )
                ui.notify(
                    message,
                    type="positive" if ok else "negative",
                    position="bottom-right",
                    timeout=8000 if ok else 6000,
                )
                if ok:
                    _refresh_teleop_status()
                    await _refresh_teleop_state_async()
                    refresh_named_positions = named_position_refresh["callback"]
                    if refresh_named_positions is not None:
                        refresh_named_positions()
                    refresh_robot_functions = refresh_callbacks.get("robot_functions")
                    if refresh_robot_functions is not None:
                        refresh_robot_functions()
                    await asyncio.to_thread(bridge.digital_twin_statuses)
            except Exception as exc:
                log.exception("Reset UR5e RTDE failed")
                message = f"Reset UR5e RTDE failed: {type(exc).__name__}: {exc}"
                rtde_reset_progress_label.set_text(message)
                rtde_reset_progress_label.classes(replace="text-xs text-red-700")
                ui.notify(
                    message,
                    type="negative",
                    position="bottom-right",
                    timeout=6000,
                )
            finally:
                rtde_reset_state["busy"] = False
                rtde_reset_button.props(remove="loading")
                _refresh_rtde_reset_enabled()

        rtde_reset_button.on_click(_reset_ur5e_rtde)

        def _refresh_teleop_status() -> None:
            if not _client_alive(teleop_backend_label):
                return
            if smooth_hold["pressed"] or smooth_hold["active"] or smooth_hold["stopping"]:
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
            _refresh_rtde_reset_enabled()

            target_label = save_env_label["label"]
            if target_label is not None:
                target_label.set_text(f'Save target: "{env or "gazebo"}" block in resource JSON')
            if motion_speed_controls["title"] is not None:
                _motion_settings_for(
                    str(robot_select.value or "xarm6"),
                    refresh=True,
                )
                _refresh_motion_speed_labels()

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
            if smooth_hold["pressed"] or smooth_hold["active"] or smooth_hold["stopping"]:
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
            if key != "arm_vel" or _uses_physical_motion_speeds(robot):
                return base
            multiplier = _TELEOP_PROFILE_MULTIPLIER.get(profile_state["mode"], 1.0)
            return base * multiplier

        def _motion_settings_for(robot: str, *, refresh: bool = False) -> dict:
            key = str(robot or "xarm6").strip().lower()
            if refresh:
                latest = bridge.teleop_motion_settings(key)
                if not latest.get("error"):
                    motion_settings[key] = latest
            return motion_settings[key]

        def _uses_physical_motion_speeds(robot: str) -> bool:
            return bool(_motion_settings_for(robot).get("physical_units"))

        def _cartesian_speed_mm_s(robot: str) -> float | None:
            key = str(robot or "xarm6").strip().lower()
            if not _uses_physical_motion_speeds(key):
                return None
            return float(motion_speed_state[key]["cartesian_mm_s"])

        def _joint_speed_deg_s(robot: str) -> float | None:
            key = str(robot or "xarm6").strip().lower()
            if not _uses_physical_motion_speeds(key):
                return None
            return float(motion_speed_state[key]["joint_deg_s"])

        def _aligned_motion_speed(
            value: float,
            minimum: float,
            maximum: float,
            step: float,
        ) -> float:
            steps = round((float(value) - minimum) / step)
            aligned = minimum + steps * step
            return round(min(maximum, max(minimum, aligned)), 6)

        def _set_motion_speed(
            robot: str,
            key: str,
            value: float,
            *,
            source: str = "",
        ) -> None:
            if motion_speed_sync["busy"]:
                return
            robot_key = str(robot or "xarm6").strip().lower()
            settings = motion_settings[robot_key]
            value_key = (
                "cartesian_mm_s" if key == "cartesian" else "joint_deg_s"
            )
            minimum_key = (
                "cartesian_speed_min_mm_s"
                if key == "cartesian"
                else "joint_speed_min_deg_s"
            )
            maximum_key = (
                "cartesian_speed_max_mm_s"
                if key == "cartesian"
                else "joint_speed_max_deg_s"
            )
            current = float(motion_speed_state[robot_key][value_key])
            try:
                requested = float(value)
            except (TypeError, ValueError):
                requested = math.nan
            minimum = float(settings[minimum_key])
            maximum = float(settings[maximum_key])
            step = (
                _PHYSICAL_CARTESIAN_SPEED_STEP_MM_S
                if key == "cartesian"
                else _PHYSICAL_JOINT_SPEED_STEP_DEG_S
            )
            if not math.isfinite(requested) or not minimum <= requested <= maximum:
                ui.notify(
                    f"{robot_key} {key} speed must be within "
                    f"[{minimum:.1f}, {maximum:.1f}]",
                    type="warning",
                    position="bottom-right",
                    timeout=2500,
                )
                requested = current
            else:
                requested = _aligned_motion_speed(
                    requested,
                    minimum,
                    maximum,
                    step,
                )
            motion_speed_state[robot_key][value_key] = requested
            if robot_key == str(robot_select.value or "xarm6"):
                motion_speed_sync["busy"] = True
                try:
                    slider = motion_speed_controls[f"{key}_slider"]
                    number = motion_speed_controls[f"{key}_number"]
                    if slider is not None and source != "slider":
                        slider.set_value(requested)
                        slider.update()
                    if number is not None and source != "number":
                        number.set_value(requested)
                        number.update()
                finally:
                    motion_speed_sync["busy"] = False
            _refresh_motion_speed_labels()

        def _configure_motion_speed_controls(robot: str) -> None:
            robot_key = str(robot or "xarm6").strip().lower()
            settings = _motion_settings_for(robot_key, refresh=True)
            controls = (
                (
                    "cartesian",
                    float(motion_speed_state[robot_key]["cartesian_mm_s"]),
                    float(settings["cartesian_speed_min_mm_s"]),
                    float(settings["cartesian_speed_max_mm_s"]),
                    _PHYSICAL_CARTESIAN_SPEED_STEP_MM_S,
                ),
                (
                    "joint",
                    float(motion_speed_state[robot_key]["joint_deg_s"]),
                    float(settings["joint_speed_min_deg_s"]),
                    float(settings["joint_speed_max_deg_s"]),
                    _PHYSICAL_JOINT_SPEED_STEP_DEG_S,
                ),
            )
            motion_speed_sync["busy"] = True
            try:
                for control_key, value, minimum, maximum, step in controls:
                    slider = motion_speed_controls[f"{control_key}_slider"]
                    number = motion_speed_controls[f"{control_key}_number"]
                    for control in (slider, number):
                        if control is None:
                            continue
                        control.props(
                            f"min={minimum:.6f} max={maximum:.6f} step={step:.6f}"
                        )
                        control.set_value(value)
                        control.update()
                simulation_scale = motion_speed_controls["simulation_scale"]
                if simulation_scale is not None:
                    simulation_scale.value = _velocity(robot_key, "arm_vel", 1.0)
            finally:
                motion_speed_sync["busy"] = False

        def _refresh_motion_speed_labels() -> None:
            robot = str(robot_select.value or "xarm6").strip().lower()
            settings = _motion_settings_for(robot)
            physical = bool(settings.get("physical_units"))
            title = motion_speed_controls["title"]
            if title is not None:
                title.set_text(f"Motion Speed - {robot}")
            physical_group = motion_speed_controls["physical"]
            simulation_group = motion_speed_controls["simulation"]
            if physical_group is not None:
                physical_group.set_visibility(physical)
            if simulation_group is not None:
                simulation_group.set_visibility(not physical)
            if not physical:
                applied = motion_speed_controls["applied"]
                if applied is not None:
                    applied.set_text(
                        "Gazebo uses simulation arm velocity scaling; physical mm/s and "
                        "deg/s are not claimed."
                    )
                simulation_scale_value = _effective_velocity(
                    robot, "arm_vel", 1.0
                )
                if applied_motion_labels["cartesian"] is not None:
                    applied_motion_labels["cartesian"].set_text(
                        "Applied Cartesian control: simulation scale "
                        f"{simulation_scale_value:.2f}"
                    )
                if applied_motion_labels["joint"] is not None:
                    applied_motion_labels["joint"].set_text(
                        "Applied joint jog control: simulation scale "
                        f"{simulation_scale_value:.2f}"
                    )
                return

            cartesian_value = float(motion_speed_state[robot]["cartesian_mm_s"])
            joint_value = float(motion_speed_state[robot]["joint_deg_s"])
            range_label = motion_speed_controls["range"]
            if range_label is not None:
                range_label.set_text(
                    "Cartesian range: "
                    f"{float(settings['cartesian_speed_min_mm_s']):.3f}-"
                    f"{float(settings['cartesian_speed_max_mm_s']):.3f} mm/s"
                )
            acceleration = motion_speed_controls["acceleration"]
            if acceleration is not None:
                acceleration.set_text(
                    "Configured Cartesian acceleration: "
                    f"{float(settings['cartesian_acceleration_mm_s2']):.1f} mm/s²"
                )
            applied = motion_speed_controls["applied"]
            if applied is not None:
                applied.set_text(
                    f"Applied: Cartesian {cartesian_value:.3f} mm/s | "
                    f"Joint {joint_value:.1f} deg/s"
                )
            if applied_motion_labels["cartesian"] is not None:
                applied_motion_labels["cartesian"].set_text(
                    f"Applied Cartesian speed: {cartesian_value:.3f} mm/s"
                )
            if applied_motion_labels["joint"] is not None:
                applied_motion_labels["joint"].set_text(
                    f"Applied joint jog speed: {joint_value:.1f} deg/s"
                )

        def _smooth_speed_mm_s(direction: int) -> float:
            robot = str(robot_select.value or "xarm6")
            configured = _cartesian_speed_mm_s(robot)
            if configured is None:
                configured = 10.0 if profile_state["mode"] == "precision" else 30.0
            return math.copysign(float(configured), direction)

        async def _stop_smooth_hold(*, notify_failure: bool = True) -> None:
            smooth_hold["pressed"] = False
            smooth_hold["generation"] = int(smooth_hold["generation"]) + 1
            if smooth_hold["stopping"]:
                return
            robot = str(smooth_hold["robot"] or robot_select.value or "xarm6")
            axis = str(smooth_hold["axis"] or "x")
            active = bool(smooth_hold["active"])
            smooth_hold["stopping"] = True
            try:
                if active:
                    ok, message = await asyncio.to_thread(
                        bridge.teleop_cartesian_smooth,
                        robot,
                        axis,
                        0.0,
                        "stop",
                    )
                    if not ok and notify_failure and _client_alive(teleop_warning_label):
                        ui.notify(
                            f"{robot}: Smooth Hold stop failed ({message})",
                            type="negative",
                            position="bottom-right",
                            timeout=5000,
                        )
                    elif ok and _client_alive(teleop_warning_label):
                        label = cartesian_status["label"]
                        if label is not None:
                            label.set_text(
                                "Motion stopped; Smooth Hold remains prepared."
                            )
                            label.classes(replace="text-xs text-green-700 mb-2")
            finally:
                smooth_hold["active"] = False
                smooth_hold["robot"] = ""
                smooth_hold["axis"] = ""
                smooth_hold["speed_mm_s"] = 0.0
                smooth_hold["task"] = None
                smooth_hold["stopping"] = False

        async def _run_smooth_hold(
            robot: str,
            axis: str,
            direction: int,
            generation: int,
        ) -> None:
            speed_mm_s = _smooth_speed_mm_s(direction)
            ok, message = await asyncio.to_thread(
                bridge.teleop_cartesian_smooth,
                robot,
                axis,
                speed_mm_s,
                "start",
            )
            if not ok:
                smooth_hold["pressed"] = False
                if _client_alive(teleop_warning_label):
                    ui.notify(
                        f"{robot}: Smooth Hold failed ({message})",
                        type="negative",
                        position="bottom-right",
                        timeout=5000,
                    )
                return
            smooth_hold["active"] = True
            smooth_hold["robot"] = robot
            smooth_hold["axis"] = axis
            smooth_hold["speed_mm_s"] = speed_mm_s
            label = cartesian_status["label"]
            if label is not None:
                label.set_text(
                    f"World {axis.upper()}{'+' if direction > 0 else '-'} moving "
                    f"at {abs(speed_mm_s):.3f} mm/s"
                )
                label.classes(replace="text-xs text-blue-700 mb-2")
            try:
                while (
                    smooth_hold["pressed"]
                    and int(smooth_hold["generation"]) == generation
                    and cartesian_jog_mode["mode"] == "smooth"
                    and str(robot_select.value or "") == robot
                ):
                    await asyncio.sleep(0.1)
                    ok, message = await asyncio.to_thread(
                        bridge.teleop_cartesian_smooth,
                        robot,
                        axis,
                        speed_mm_s,
                        "update",
                    )
                    if not ok:
                        smooth_hold["pressed"] = False
                        if _client_alive(teleop_warning_label):
                            ui.notify(
                                f"{robot}: Smooth Hold stopped ({message})",
                                type="negative",
                                position="bottom-right",
                                timeout=5000,
                            )
                        break
            finally:
                await _stop_smooth_hold(notify_failure=False)

        def _start_smooth_hold(axis: str, direction: int) -> None:
            if cartesian_jog_mode["mode"] != "smooth":
                return
            if smooth_hold["pressed"] or smooth_hold["active"]:
                return
            smooth_hold["pressed"] = True
            smooth_hold["generation"] = int(smooth_hold["generation"]) + 1
            generation = int(smooth_hold["generation"])
            robot = str(robot_select.value or "xarm6")
            smooth_hold["task"] = asyncio.create_task(
                _run_smooth_hold(robot, axis, direction, generation)
            )

        def _request_smooth_stop(_event=None) -> None:
            smooth_hold["pressed"] = False
            if smooth_hold["active"] and not smooth_hold["stopping"]:
                asyncio.create_task(_stop_smooth_hold())

        def _set_cartesian_toggle_value(value: str) -> None:
            toggle = cartesian_jog_controls["toggle"]
            if toggle is None:
                return
            display = "Smooth Hold" if str(value).strip() == "Smooth Hold" else "Step"
            cartesian_jog_mode["updating_toggle"] = True
            try:
                toggle.value = display
            finally:
                cartesian_jog_mode["updating_toggle"] = False

        async def _apply_cartesian_mode(value: str) -> bool:
            if cartesian_jog_mode["preparing"]:
                return False
            requested_label = str(value or "Step").strip()
            requested = requested_label.lower()
            if requested == "smooth hold":
                requested = "smooth"
            if requested not in {"step", "smooth"}:
                return False
            robot = str(robot_select.value or "xarm6").strip().lower()
            if requested == "smooth":
                readiness = bridge.teleop_cartesian_readiness(robot)
                if str(readiness.get("environment") or "") != "real":
                    _set_cartesian_toggle_value("Step")
                    cartesian_jog_mode["mode"] = "step"
                    ui.notify(
                        "Smooth Hold is available only for Hardware Stack.",
                        type="warning",
                        position="bottom-right",
                        timeout=3000,
                    )
                    return False
            previous_mode = str(cartesian_jog_mode["mode"] or "step")
            cartesian_jog_mode["mode"] = requested
            _set_cartesian_toggle_value(
                "Smooth Hold" if requested == "smooth" else "Step"
            )
            if previous_mode == "smooth" and requested != "smooth":
                await _stop_smooth_hold(notify_failure=True)
            cartesian_jog_mode["preparing"] = True
            cartesian_command_state["pending"] = True
            label = cartesian_status["label"] or cartesian_jog_controls["readiness"]
            if label is not None:
                if robot == "xarm6":
                    expected_mode = 0 if requested == "step" else 5
                    label.set_text(f"Preparing Mode {expected_mode}...")
                else:
                    label.set_text(f"Preparing {requested.title()}...")
                label.classes(replace="text-xs text-blue-700 mb-2")
            _refresh_cartesian_controls()
            try:
                ok, message = await asyncio.to_thread(
                    bridge.teleop_cartesian_mode,
                    robot,
                    requested,
                )
                if not _client_alive(cartesian_jog_controls["readiness"]):
                    return False
                if ok:
                    cartesian_jog_mode["mode"] = requested
                    if label is not None:
                        label.set_text(message)
                        label.classes(replace="text-xs text-green-700 mb-2")
                    return True
                else:
                    if label is not None:
                        label.set_text(f"Cartesian mode failed: {message}")
                        label.classes(replace="text-xs text-red-700 mb-2")
                    ui.notify(
                        f"{robot}: {message}",
                        type="negative",
                        position="bottom-right",
                        timeout=5000,
                    )
                    return False
            finally:
                cartesian_jog_mode["preparing"] = False
                cartesian_command_state["pending"] = False
                _refresh_cartesian_controls()

        def _refresh_cartesian_controls() -> None:
            label = cartesian_jog_controls["readiness"]
            toggle = cartesian_jog_controls["toggle"]
            if label is None or toggle is None or not _client_alive(label):
                return
            if smooth_hold["pressed"] or smooth_hold["active"] or smooth_hold["stopping"]:
                for button in cartesian_jog_buttons:
                    button.disable()
                for button in conflicting_motion_buttons:
                    button.disable()
                toggle.disable()
                robot_select.disable()
                return
            if cartesian_jog_mode["preparing"] or cartesian_command_state["pending"]:
                for button in cartesian_jog_buttons:
                    button.disable()
                for button in conflicting_motion_buttons:
                    button.disable()
                toggle.disable()
                robot_select.disable()
                return
            robot = str(robot_select.value or "xarm6")
            readiness = bridge.teleop_cartesian_readiness(robot)
            environment = str(readiness.get("environment") or "")
            selected_mode = str(cartesian_jog_mode["mode"] or "step")
            if environment == "real":
                base_ready = bool(readiness.get("cartesian_jog_ready"))
                message = str(readiness.get("message") or "")
                reported_mode = str(readiness.get("cartesian_mode") or "off")
                if (
                    robot == "xarm6"
                    and selected_mode == "smooth"
                    and reported_mode == "off"
                    and not cartesian_jog_mode["preparing"]
                ):
                    selected_mode = "step"
                    cartesian_jog_mode["mode"] = "step"
                    _set_cartesian_toggle_value("Step")
                mode_ready = bool(
                    selected_mode in {"step", "smooth"}
                    and readiness.get("cartesian_mode_ready") is True
                    and reported_mode == selected_mode
                )
                ready = bool(base_ready and mode_ready)
                state_uncertain = bool(
                    readiness.get("state_uncertain")
                    or readiness.get("cartesian_jog_state_uncertain")
                )
                idle_remaining = float(
                    readiness.get("cartesian_mode_idle_remaining_sec") or 0.0
                )
                if state_uncertain:
                    label.set_text(f"Cartesian motion blocked: {message}")
                    label.classes(replace="text-xs text-red-700 mb-2")
                elif ready:
                    mode_text = (
                        "Mode 0" if selected_mode == "step" else "Mode 5"
                    ) if robot == "xarm6" else selected_mode.title()
                    idle_text = (
                        f"; idle restore in {idle_remaining:.1f}s"
                        if robot == "xarm6" and idle_remaining > 0.0
                        else ""
                    )
                    label.set_text(f"Cartesian {mode_text} ready{idle_text}")
                    label.classes(replace="text-xs text-green-700 mb-2")
                elif base_ready and selected_mode == "step" and robot == "xarm6":
                    label.set_text(
                        "Step selected; Mode 1 active. First World move prepares Mode 0."
                    )
                    label.classes(replace="text-xs text-amber-700 mb-2")
                elif base_ready and selected_mode == "step":
                    label.set_text(
                        "Step selected. First World move prepares Cartesian motion."
                    )
                    label.classes(replace="text-xs text-amber-700 mb-2")
                else:
                    label.set_text(f"Cartesian frame validation failed: {message}")
                    label.classes(replace="text-xs text-red-700 mb-2")
                smooth_available = bool(base_ready)
            else:
                target = bridge.teleop_target(robot, "cartesian")
                base_ready = bool(target.get("ready"))
                smooth_available = False
                ready = bool(base_ready and selected_mode == "step")
                label.set_text("Simulation uses Step Cartesian jog with velocity scaling.")
                label.classes(replace="text-xs text-slate-500 mb-2")
            if not smooth_available and selected_mode == "smooth":
                cartesian_jog_mode["mode"] = "step"
                _set_cartesian_toggle_value("Step")
                _request_smooth_stop()
                selected_mode = "step"
                ready = bool(base_ready)
            controls_enabled = bool(
                (ready or (base_ready and selected_mode == "step"))
                and not (
                    environment == "real"
                    and bool(
                        readiness.get("state_uncertain")
                        or readiness.get("cartesian_jog_state_uncertain")
                    )
                )
                and not cartesian_jog_mode["preparing"]
                and not cartesian_command_state["pending"]
            )
            for button in cartesian_jog_buttons:
                button.set_enabled(controls_enabled)
            for button in conflicting_motion_buttons:
                button.enable()
            toggle.set_enabled(
                not cartesian_jog_mode["preparing"]
                and not cartesian_command_state["pending"]
            )
            robot_select.enable()

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

        def _apply_profile_speeds(mode: str) -> None:
            multiplier = _TELEOP_PROFILE_MULTIPLIER.get(mode, 1.0)
            for robot in ("xarm6", "ur5e"):
                settings = _motion_settings_for(robot)
                cartesian_minimum = float(settings["cartesian_speed_min_mm_s"])
                cartesian_maximum = float(settings["cartesian_speed_max_mm_s"])
                motion_speed_state[robot]["cartesian_mm_s"] = _aligned_motion_speed(
                    float(settings["cartesian_speed_default_mm_s"]) * multiplier,
                    cartesian_minimum,
                    cartesian_maximum,
                    _PHYSICAL_CARTESIAN_SPEED_STEP_MM_S,
                )
                joint_minimum = float(settings["joint_speed_min_deg_s"])
                joint_maximum = float(settings["joint_speed_max_deg_s"])
                motion_speed_state[robot]["joint_deg_s"] = _aligned_motion_speed(
                    float(settings["joint_speed_default_deg_s"]) * multiplier,
                    joint_minimum,
                    joint_maximum,
                    _PHYSICAL_JOINT_SPEED_STEP_DEG_S,
                )
            _configure_motion_speed_controls(
                str(robot_select.value or "xarm6")
            )
            _refresh_motion_speed_labels()

        def _refresh_effective_labels() -> None:
            note_label = profile_note["label"]
            if note_label is not None:
                robot = str(robot_select.value or "xarm6")
                cartesian_speed = motion_speed_state[robot]["cartesian_mm_s"]
                joint_speed = motion_speed_state[robot]["joint_deg_s"]
                note_label.set_text(
                    f"{profile_state['mode'].title()} preset: Cartesian "
                    f"{cartesian_speed:.3f} mm/s, Joint {joint_speed:.1f} deg/s; "
                    "step values are shown below."
                )
            for robot_name, label in effective_labels.items():
                arm = _effective_velocity(robot_name, "arm_vel", 1.0)
                label.set_text(f"Applied simulation arm velocity scale: {arm:.2f}")

        with ui.row().classes("items-center gap-3 mb-2"):
            ui.label("Teleop Profile:").classes("font-semibold text-sm")

            def _set_profile(value: str) -> None:
                mode = str(value).strip().lower()
                if mode not in _TELEOP_PROFILE_MULTIPLIER:
                    return
                changed = profile_state["mode"] != mode
                if changed:
                    _request_smooth_stop()
                profile_state["mode"] = mode
                _apply_profile_steps(mode)
                _apply_profile_speeds(mode)
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

        with ui.card().classes("w-full mb-4 border border-blue-200"):
            motion_speed_controls["title"] = ui.label(
                f"Motion Speed - {initial_robot}"
            ).classes("font-semibold text-sm")
            ui.label(
                "Physical speeds are sent as exact values. xArm6 and UR5e retain "
                "independent page-session settings."
            ).classes("text-xs text-slate-500")
            with ui.column().classes("w-full gap-2") as physical_speed_group:
                with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                    with ui.column().classes("gap-0 grow min-w-64"):
                        ui.label("Cartesian speed (mm/s)").classes(
                            "text-xs font-semibold"
                        )
                        cartesian_speed_slider = ui.slider(
                            min=float(
                                motion_settings[initial_robot][
                                    "cartesian_speed_min_mm_s"
                                ]
                            ),
                            max=float(
                                motion_settings[initial_robot][
                                    "cartesian_speed_max_mm_s"
                                ]
                            ),
                            step=_PHYSICAL_CARTESIAN_SPEED_STEP_MM_S,
                            value=motion_speed_state[initial_robot]["cartesian_mm_s"],
                        ).classes("w-full")
                        cartesian_speed_slider.on(
                            "change",
                            lambda event: _set_motion_speed(
                                str(robot_select.value or "xarm6"),
                                "cartesian",
                                event.args,
                                source="slider",
                            ),
                        )
                        motion_speed_controls["cartesian_slider"] = (
                            cartesian_speed_slider
                        )
                    motion_speed_controls["cartesian_number"] = ui.number(
                        "Cartesian speed (mm/s)",
                        value=motion_speed_state[initial_robot]["cartesian_mm_s"],
                        min=float(
                            motion_settings[initial_robot][
                                "cartesian_speed_min_mm_s"
                            ]
                        ),
                        max=float(
                            motion_settings[initial_robot][
                                "cartesian_speed_max_mm_s"
                            ]
                        ),
                        step=_PHYSICAL_CARTESIAN_SPEED_STEP_MM_S,
                        on_change=lambda event: _set_motion_speed(
                            str(robot_select.value or "xarm6"),
                            "cartesian",
                            event.value,
                            source="number",
                        ),
                    ).classes("w-48")
                with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                    with ui.column().classes("gap-0 grow min-w-64"):
                        ui.label("Joint jog speed (deg/s)").classes(
                            "text-xs font-semibold"
                        )
                        joint_speed_slider = ui.slider(
                            min=float(
                                motion_settings[initial_robot][
                                    "joint_speed_min_deg_s"
                                ]
                            ),
                            max=float(
                                motion_settings[initial_robot][
                                    "joint_speed_max_deg_s"
                                ]
                            ),
                            step=_PHYSICAL_JOINT_SPEED_STEP_DEG_S,
                            value=motion_speed_state[initial_robot]["joint_deg_s"],
                        ).classes("w-full")
                        joint_speed_slider.on(
                            "change",
                            lambda event: _set_motion_speed(
                                str(robot_select.value or "xarm6"),
                                "joint",
                                event.args,
                                source="slider",
                            ),
                        )
                        motion_speed_controls["joint_slider"] = joint_speed_slider
                    motion_speed_controls["joint_number"] = ui.number(
                        "Joint jog speed (deg/s)",
                        value=motion_speed_state[initial_robot]["joint_deg_s"],
                        min=float(
                            motion_settings[initial_robot]["joint_speed_min_deg_s"]
                        ),
                        max=float(
                            motion_settings[initial_robot]["joint_speed_max_deg_s"]
                        ),
                        step=_PHYSICAL_JOINT_SPEED_STEP_DEG_S,
                        on_change=lambda event: _set_motion_speed(
                            str(robot_select.value or "xarm6"),
                            "joint",
                            event.value,
                            source="number",
                        ),
                    ).classes("w-48")
                motion_speed_controls["range"] = ui.label("").classes(
                    "text-xs text-slate-600"
                )
                motion_speed_controls["acceleration"] = ui.label("").classes(
                    "text-xs text-slate-600"
                )
            motion_speed_controls["physical"] = physical_speed_group
            with ui.column().classes("w-full gap-1") as simulation_speed_group:
                ui.label(
                    "Simulation-only control: scaling, not physical mm/s or deg/s."
                ).classes("text-xs text-amber-700")
                motion_speed_controls["simulation_scale"] = ui.number(
                    "Simulation arm velocity scale",
                    value=velocity_state[initial_robot]["arm_vel"],
                    min=0.1,
                    max=3.0,
                    step=0.1,
                    on_change=lambda event: _set_velocity(
                        str(robot_select.value or "xarm6"),
                        "arm_vel",
                        event.value,
                    ),
                ).classes("w-56")
            motion_speed_controls["simulation"] = simulation_speed_group
            motion_speed_controls["applied"] = ui.label("").classes(
                "text-sm font-medium text-blue-800"
            )
            _configure_motion_speed_controls(initial_robot)
            _refresh_motion_speed_labels()
            _refresh_effective_labels()

        with (
            ui.element("div")
            .classes("w-full columns-1 lg:columns-2 xl:columns-3")
            .style("column-gap: 1.5rem;")
        ):
            # ── Cartesian Jog Pad ────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Cartesian Jog").classes("font-semibold text-sm mb-2")
                ui.label(
                    "Commands translation only along the displayed World X, World Y, or "
                    "World Z axis. Rotation is never requested."
                ).classes("text-xs text-amber-700 mb-2")

                def _set_cartesian_jog_mode(value: str) -> None:
                    if cartesian_jog_mode["updating_toggle"]:
                        return
                    asyncio.create_task(_apply_cartesian_mode(str(value or "Step")))

                cartesian_jog_controls["toggle"] = ui.toggle(
                    ["Step", "Smooth Hold"],
                    value="Step",
                    on_change=lambda event: _set_cartesian_jog_mode(event.value),
                ).props("dense")
                cartesian_jog_controls["readiness"] = ui.label(
                    "Cartesian frame validation: checking..."
                ).classes("text-xs text-slate-500 mb-2")
                cartesian_status["label"] = cartesian_jog_controls["readiness"]
                step_input = ui.number(
                    "Step (mm)", value=10.0, min=0.1, max=100.0, step=0.1
                ).classes("w-32 mb-3")
                step_inputs["cartesian"] = step_input
                applied_motion_labels["cartesian"] = ui.label("").classes(
                    "text-xs font-medium text-blue-800 mb-1"
                )
                ui.label("Switching Precision/Fast also updates this step value.").classes(
                    "text-xs text-slate-500 mb-2"
                )

                async def _send_cartesian_step(axis: str, direction: int) -> None:
                    if cartesian_jog_mode["mode"] != "step":
                        return
                    robot = str(robot_select.value or "xarm6")
                    readiness = bridge.teleop_cartesian_readiness(robot)
                    if not (
                        str(readiness.get("cartesian_mode") or "off") == "step"
                        and readiness.get("cartesian_mode_ready") is True
                    ):
                        prepared = await _apply_cartesian_mode("Step")
                        if not prepared:
                            return
                    step_mm = math.copysign(float(step_input.value or 10.0), direction)
                    speed_mm_s = _cartesian_speed_mm_s(robot)
                    velocity_scale = _effective_velocity(robot, "arm_vel", 1.0)
                    cartesian_command_state["pending"] = True
                    if speed_mm_s is not None:
                        cartesian_status["label"].set_text(
                            f"World {axis.upper()}{'+' if direction > 0 else '-'} "
                            f"moving at {speed_mm_s:.3f} mm/s"
                        )
                    else:
                        cartesian_status["label"].set_text(
                            f"World {axis.upper()}{'+' if direction > 0 else '-'} moving "
                            f"at simulation scale {velocity_scale:.2f}"
                        )
                    cartesian_status["label"].classes(
                        replace="text-xs text-blue-700 mb-2"
                    )
                    _refresh_cartesian_controls()
                    try:
                        await _send_jog(
                            bridge,
                            robot,
                            axis,
                            step_mm,
                            velocity_scale,
                            speed_mm_s=speed_mm_s,
                        )
                    finally:
                        cartesian_command_state["pending"] = False
                        _refresh_cartesian_controls()

                def _cartesian_jog_button(
                    label: str,
                    axis: str,
                    direction: int,
                    icon: str,
                ):
                    button = _jog_btn(
                        bridge,
                        robot_select,
                        step_input,
                        _effective_velocity,
                        label,
                        axis,
                        direction,
                        icon,
                        mode_getter=lambda: cartesian_jog_mode["mode"],
                        step_sender=_send_cartesian_step,
                        smooth_start=_start_smooth_hold,
                        smooth_stop=_request_smooth_stop,
                    )
                    cartesian_jog_buttons.append(button)
                    return button

                # X/Y pad (top-down view).
                ui.label("World X / World Y").classes("text-xs text-slate-500 mb-1")
                with ui.column().classes("items-center gap-1"):
                    _cartesian_jog_button("World Y+", "y", 1, "arrow_upward")
                    with ui.row().classes("gap-1"):
                        _cartesian_jog_button("World X-", "x", -1, "arrow_back")
                        ui.button(icon="radio_button_unchecked").props(
                            "flat dense disable"
                        ).classes("w-12 h-12")
                        _cartesian_jog_button("World X+", "x", 1, "arrow_forward")
                    _cartesian_jog_button("World Y-", "y", -1, "arrow_downward")

                # Z axis.
                ui.label("World Z").classes("text-xs text-slate-500 mt-3 mb-1")
                with ui.row().classes("gap-2 justify-center"):
                    _cartesian_jog_button("World Z+", "z", 1, "expand_less")
                    _cartesian_jog_button("World Z-", "z", -1, "expand_more")

            # ── Joint Jog ────────────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Joint Jog").classes("font-semibold text-sm mb-2")
                joint_step_input = ui.number(
                    "Step (deg)", value=2.0, min=0.05, max=30.0, step=0.05
                ).classes("w-32 mb-2")
                step_inputs["joint"] = joint_step_input
                applied_motion_labels["joint"] = ui.label("").classes(
                    "text-xs font-medium text-blue-800 mb-1"
                )
                _refresh_motion_speed_labels()
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
                        speed_deg_s = _joint_speed_deg_s(robot_select.value)
                        asyncio.create_task(
                            _send_joint(
                                bridge,
                                robot_select.value,
                                joint_state["idx"],
                                -step_deg,
                                arm_vel,
                                speed_deg_s=speed_deg_s,
                            )
                        )

                    def _joint_plus():
                        step_deg = joint_step_input.value or 2.0
                        arm_vel = _effective_velocity(robot_select.value, "arm_vel", 1.0)
                        speed_deg_s = _joint_speed_deg_s(robot_select.value)
                        asyncio.create_task(
                            _send_joint(
                                bridge,
                                robot_select.value,
                                joint_state["idx"],
                                step_deg,
                                arm_vel,
                                speed_deg_s=speed_deg_s,
                            )
                        )

                    conflicting_motion_buttons.append(
                        ui.button("-", on_click=_joint_minus, icon="remove").props(
                            "outline"
                        )
                    )
                    conflicting_motion_buttons.append(
                        ui.button("+", on_click=_joint_plus, icon="add").props(
                            "outline"
                        )
                    )

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

                    conflicting_motion_buttons.append(
                        ui.button(
                            "Full Open", on_click=_full_open, icon="open_with"
                        ).props("outline")
                    )
                    conflicting_motion_buttons.append(
                        ui.button(
                            "Full Close",
                            on_click=_full_close,
                            icon="close_fullscreen",
                        ).props("outline")
                    )
                with ui.row().classes("gap-2 justify-center"):
                    conflicting_motion_buttons.append(
                        ui.button("Open", on_click=_step_open, icon="add").props(
                            "outline"
                        )
                    )
                    conflicting_motion_buttons.append(
                        ui.button("Close", on_click=_step_close, icon="remove").props(
                            "outline"
                        )
                    )

                # Home button.
                ui.separator().classes("my-3")

                def _go_home():
                    asyncio.create_task(_send_home(bridge, robot_select.value))

                conflicting_motion_buttons.append(
                    ui.button("Move Home", on_click=_go_home, icon="home")
                    .props("outline")
                    .classes("w-full")
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

            # ── Gripper Settings ─────────────────────────────────────
            with ui.card().classes("w-full mb-6 break-inside-avoid"):
                ui.label("Gripper Settings (Per Robot)").classes(
                    "font-semibold text-sm mb-2"
                )
                ui.label(
                    "Gripper command scale is separate from arm motion speed and is not "
                    "a physical velocity."
                ).classes("text-xs text-slate-500 mb-2")

                def _velocity_block(robot: str) -> None:
                    with ui.column().classes("gap-1 mb-3"):
                        ui.label(robot).classes("text-xs font-semibold text-slate-600")
                        ui.number(
                            "Gripper command scale",
                            value=velocity_state[robot]["gripper_vel"],
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            on_change=lambda e, r=robot: _set_velocity(r, "gripper_vel", e.value),
                        ).classes("w-44")

                with ui.row().classes("gap-6 flex-wrap"):
                    _velocity_block("xarm6")
                    _velocity_block("ur5e")

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

                named_position_refresh["callback"] = _refresh_named_positions

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
                            recovery_suffix = (
                                " Robot Function recovery state cleared."
                                if robot == "ur5e" and name == "home"
                                else ""
                            )
                            ui.notify(
                                f"{robot}: moved to '{name}'.{recovery_suffix}",
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

        async def _close_cartesian_mode_for_robot_change(
            previous_robot: str,
            selected_robot: str,
        ) -> None:
            selected_robot_state["changing"] = True
            cartesian_jog_mode["preparing"] = True
            try:
                await _stop_smooth_hold(notify_failure=False)
                await asyncio.to_thread(
                    bridge.teleop_cartesian_mode,
                    previous_robot,
                    "off",
                )
            finally:
                selected_robot_state["robot"] = selected_robot
                selected_robot_state["changing"] = False
                cartesian_jog_mode["preparing"] = False
                cartesian_jog_mode["mode"] = "step"
                _set_cartesian_toggle_value("Step")
                _configure_motion_speed_controls(selected_robot)
                _refresh_motion_speed_labels()
                _refresh_effective_labels()
                _refresh_cartesian_controls()

        def _robot_selection_changed(_event=None) -> None:
            selected_robot = str(robot_select.value or "xarm6")
            previous_robot = str(selected_robot_state["robot"] or "xarm6")
            if selected_robot_state["changing"] or selected_robot == previous_robot:
                _refresh_motion_speed_labels()
                return
            asyncio.create_task(
                _close_cartesian_mode_for_robot_change(
                    previous_robot,
                    selected_robot,
                )
            )

        robot_select.on_value_change(_robot_selection_changed)
        ui.on("cais_cartesian_smooth_stop", _request_smooth_stop)
        ui.run_javascript(
            """
            if (!window.__caisCartesianSmoothStopInstalled) {
              window.__caisCartesianSmoothStopInstalled = true;
              const stop = () => emitEvent('cais_cartesian_smooth_stop');
              window.addEventListener('pointerup', stop);
              window.addEventListener('pointercancel', stop);
              window.addEventListener('blur', stop);
              document.addEventListener('visibilitychange', () => {
                if (document.hidden) stop();
              });
            }
            """
        )

        async def _client_disconnected() -> None:
            await _stop_smooth_hold(notify_failure=False)
            await asyncio.to_thread(
                bridge.teleop_cartesian_mode,
                str(robot_select.value or "xarm6"),
                "off",
            )

        ui.context.client.on_disconnect(_client_disconnected)

        _refresh_teleop_status()
        _refresh_cartesian_controls()
        ui.timer(1.0, _refresh_teleop_status)
        ui.timer(0.5, _refresh_cartesian_controls)
        asyncio.create_task(_refresh_teleop_state_async())
        ui.timer(1.0, _refresh_teleop_state_async)

        # ── Keyboard handler ─────────────────────────────────────────
        def _keyboard_cartesian(
            robot: str,
            axis: str,
            direction: int,
            step_mm: float,
            arm_vel: float,
        ) -> None:
            if cartesian_jog_mode["mode"] == "smooth":
                _start_smooth_hold(axis, direction)
            elif cartesian_jog_mode["mode"] == "step":
                asyncio.create_task(_send_cartesian_step(axis, direction))
            else:
                ui.notify(
                    "Select Step or Smooth Hold first.",
                    type="warning",
                    position="bottom-right",
                    timeout=1800,
                )

        def _on_key(e: KeyEventArguments):
            key_name = e.key.name if hasattr(e.key, "name") else str(e.key)
            if e.action.keyup:
                if key_name in {
                    "ArrowLeft",
                    "ArrowRight",
                    "ArrowUp",
                    "ArrowDown",
                    "PageUp",
                    "PageDown",
                }:
                    _request_smooth_stop()
                return
            if e.action.keydown and not e.action.repeat:
                robot = robot_select.value
                step_mm = step_input.value or 10.0
                step_deg = joint_step_input.value or 2.0
                arm_vel = _effective_velocity(robot, "arm_vel", 1.0)
                grip_vel = _effective_velocity(robot, "gripper_vel", 1.0)
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
                    _keyboard_cartesian(robot, "z", 1, step_mm, arm_vel)
                elif key_name == "PageDown":
                    mode_state["mode"] = "cartesian"
                    _keyboard_cartesian(robot, "z", -1, step_mm, arm_vel)
                elif mode_state["mode"] == "gripper":
                    if key_name == "ArrowUp":
                        asyncio.create_task(_send_gripper(bridge, robot, "open", None, grip_vel))
                    elif key_name == "ArrowDown":
                        asyncio.create_task(_send_gripper(bridge, robot, "close", None, grip_vel))
                elif mode_state["mode"] == "joint":
                    speed_deg_s = _joint_speed_deg_s(robot)
                    if key_name == "ArrowUp":
                        asyncio.create_task(
                            _send_joint(
                                bridge,
                                robot,
                                joint_state["idx"],
                                step_deg,
                                arm_vel,
                                speed_deg_s=speed_deg_s,
                            )
                        )
                    elif key_name == "ArrowDown":
                        asyncio.create_task(
                            _send_joint(
                                bridge,
                                robot,
                                joint_state["idx"],
                                -step_deg,
                                arm_vel,
                                speed_deg_s=speed_deg_s,
                            )
                        )
                elif mode_state["mode"] == "cartesian":
                    # XY uses left/right only, as requested.
                    axis = axis_state["axis"]
                    if axis in {"x", "y"}:
                        if key_name == "ArrowRight":
                            _keyboard_cartesian(robot, axis, 1, step_mm, arm_vel)
                        elif key_name == "ArrowLeft":
                            _keyboard_cartesian(robot, axis, -1, step_mm, arm_vel)
                    elif axis == "z":
                        if key_name == "ArrowUp":
                            _keyboard_cartesian(robot, "z", 1, step_mm, arm_vel)
                        elif key_name == "ArrowDown":
                            _keyboard_cartesian(robot, "z", -1, step_mm, arm_vel)

        ui.keyboard(on_key=_on_key, ignore=["input", "select", "textarea"])


def _jog_btn(
    bridge,
    robot_select,
    step_input,
    velocity_getter,
    label,
    axis,
    direction,
    icon,
    *,
    mode_getter=lambda: "step",
    step_sender=None,
    smooth_start=None,
    smooth_stop=None,
):
    def _on_click(a=axis, d=direction):
        if str(mode_getter() or "step") != "step":
            return
        if step_sender is not None:
            asyncio.create_task(step_sender(a, d))
            return
        step = (step_input.value or 10.0) * d
        try:
            arm_vel = float(velocity_getter(robot_select.value, "arm_vel", 1.0))
        except Exception:
            arm_vel = 1.0
        asyncio.create_task(_send_jog(bridge, robot_select.value, a, step, arm_vel))

    button = (
        ui.button(icon=icon, on_click=_on_click)
        .props("flat dense")
        .classes("w-12 h-12")
        .style("touch-action: none; user-select: none;")
        .tooltip(label)
    )
    if smooth_start is not None:
        button.on(
            "pointerdown",
            lambda _event=None, a=axis, d=direction: smooth_start(a, d),
            js_handler=(
                "(event) => { event.preventDefault(); "
                "event.currentTarget.setPointerCapture(event.pointerId); emit(); }"
            ),
        )
    if smooth_stop is not None:
        for event_name in ("pointerup", "pointercancel"):
            button.on(
                event_name,
                smooth_stop,
                js_handler=(
                    "(event) => { event.preventDefault(); "
                    "if (event.currentTarget.hasPointerCapture(event.pointerId)) "
                    "event.currentTarget.releasePointerCapture(event.pointerId); emit(); }"
                ),
            )
    return button


# =====================================================================
# ROS2 command senders (publish trajectory messages)
# =====================================================================
async def _send_jog(
    bridge: SystemBridge,
    robot: str,
    axis: str,
    step_mm: float,
    velocity_scale: float = 1.0,
    *,
    speed_mm_s: float | None = None,
) -> tuple[bool, str]:
    """Send one Cartesian jog command through the ROS2 teleop backend."""
    ok, msg = await asyncio.to_thread(
        bridge.teleop_jog,
        robot,
        axis,
        step_mm,
        velocity_scale,
        speed_mm_s=speed_mm_s,
    )
    if ok:
        ui.notify(
            f"{robot}: jog {axis} {step_mm:+.0f}mm",
            type="positive",
            position="bottom-right",
            timeout=1200,
        )
        return True, msg
    ui.notify(
        f"{robot}: jog failed ({msg})", type="negative", position="bottom-right", timeout=3500
    )
    return False, msg


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
    *,
    speed_deg_s: float | None = None,
) -> tuple[bool, str]:
    """Send one joint jog command through the ROS2 teleop backend."""
    ok, msg = await asyncio.to_thread(
        bridge.teleop_joint,
        robot,
        joint_idx,
        delta_deg,
        velocity_scale,
        speed_deg_s=speed_deg_s,
    )
    if ok:
        ui.notify(
            f"{robot}: J{joint_idx} {delta_deg:+.2f}deg",
            type="positive",
            position="bottom-right",
            timeout=1200,
        )
        return True, msg
    ui.notify(
        f"{robot}: joint jog failed ({msg})", type="negative", position="bottom-right", timeout=3500
    )
    return False, msg


async def _save_position(bridge: SystemBridge, robot: str, name: str) -> None:
    """Save current robot joint positions under a named entry."""
    ok, msg = await asyncio.to_thread(bridge.teleop_save_position, robot, name)
    if ok:
        ui.notify(f"{robot}: {msg}", type="positive", position="bottom-right", timeout=2200)
        return
    ui.notify(
        f"{robot}: save failed ({msg})", type="negative", position="bottom-right", timeout=3500
    )
