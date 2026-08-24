"""Control page: Gazebo/hardware launch, interactive teleop with arrow buttons + keyboard."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

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
_MOVE_INSERT_SUPPORTED_PARTS = ("SG", "MG", "LG", "SCP", "MCP", "LCP")


def _assembly_board_v1_tag_currently_visible(status: dict) -> bool:
    """Return whether ID 70 is visible in a fresh live snapshot."""
    if not status.get("visible") or not status.get("valid"):
        return False
    try:
        frame_age_sec = float(status["frame_age_sec"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(frame_age_sec) and 0.0 <= frame_age_sec <= 2.0


def _move_insert_expected_start_diagnostic(status: dict) -> str:
    """Format the no-motion pre-insertion pose comparison for Control."""
    raw_delta = status.get("expected_start_delta_m")
    details: list[str] = []
    if isinstance(raw_delta, dict):
        try:
            delta = {
                field: float(raw_delta[field])
                for field in ("x", "y", "z")
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            delta = {}
        if delta and all(math.isfinite(value) for value in delta.values()):
            details.append(
                "Current to retained pre-insertion pose: "
                f"ΔX {1000.0 * delta['x']:+.2f} mm, "
                f"ΔY {1000.0 * delta['y']:+.2f} mm, "
                f"ΔZ {1000.0 * delta['z']:+.2f} mm."
            )

    try:
        position_error_m = float(status["expected_start_position_error_m"])
        position_tolerance_m = float(
            status["expected_start_position_tolerance_m"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        position_error_m = math.nan
        position_tolerance_m = math.nan
    if math.isfinite(position_error_m) and math.isfinite(position_tolerance_m):
        details.append(
            f"Distance {1000.0 * position_error_m:.2f} mm / "
            f"limit {1000.0 * position_tolerance_m:.2f} mm."
        )

    try:
        rotation_error_rad = float(status["expected_start_rotation_error_rad"])
        rotation_tolerance_rad = float(
            status["expected_start_orientation_tolerance_rad"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        rotation_error_rad = math.nan
        rotation_tolerance_rad = math.nan
    if math.isfinite(rotation_error_rad) and math.isfinite(rotation_tolerance_rad):
        details.append(
            f"Rotation {math.degrees(rotation_error_rad):.2f} deg / "
            f"limit {math.degrees(rotation_tolerance_rad):.2f} deg."
        )

    try:
        tf_age_sec = float(status["expected_start_tf_age_sec"])
    except (KeyError, TypeError, ValueError, OverflowError):
        tf_age_sec = math.nan
    if math.isfinite(tf_age_sec):
        details.append(f"TF age {tf_age_sec:.2f} s.")
    if details and status.get("dispatch_attempted") is False:
        details.append("Readiness only: move_insert was not dispatched.")
    return " ".join(details)


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

        def _launch_snapshot() -> tuple[tuple, dict[str, str], dict[str, dict]]:
            statuses = bridge.ros2_all_statuses()
            hardware_statuses = {
                robot: bridge.hardware_stack_status(robot)
                for robot in _HARDWARE_STACKS
            }
            signature = (
                tuple(
                    (name, statuses.get(name, "stopped"))
                    for name in _GAZEBO_VARIANTS
                ),
                tuple(
                    (name, statuses.get(name, "stopped"))
                    for name in _HARDWARE_PROC_NAMES
                ),
                tuple(
                    (
                        robot,
                        str(status.get("overall") or "stopped"),
                        str(status.get("lifecycle_state") or "stopped"),
                        status.get("lifecycle_generation"),
                        str(status.get("selected_stack") or ""),
                        str(status.get("last_error") or ""),
                    )
                    for robot, status in hardware_statuses.items()
                ),
            )
            return signature, statuses, hardware_statuses

        async def _refresh_async(*, force: bool = False) -> None:
            if not _client_alive(launch_container):
                return
            if refresh_state["busy"]:
                return
            refresh_state["busy"] = True
            try:
                signature, statuses, hardware_statuses = await asyncio.to_thread(
                    _launch_snapshot
                )
                if not force and signature == refresh_state["signature"]:
                    asyncio.create_task(_refresh_ping_async())
                    return
                refresh_state["signature"] = signature
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
                    stack_status = hardware_statuses[robot]
                    if any_gazebo_running and stack_status.get("overall") != "running":
                        blocked_reason = "Blocked: Gazebo is running. Stop Gazebo first."
                    else:
                        other_running = next(
                            (
                                other_robot
                                for other_robot in _HARDWARE_STACKS
                                if other_robot != robot
                                and hardware_statuses[other_robot].get("overall")
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

        def _refresh(*, force: bool = False) -> None:
            asyncio.create_task(_refresh_async(force=force))

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
            if blocked_reason:
                ui.label(blocked_reason).classes("text-xs text-amber-700")
            if robot in {"ur5e", "dual robots"}:
                _render_ur5e_rtde_and_rg2_status(status)

        repair_needed = lifecycle_state == "failed"
        selected_stack = str(status.get("selected_stack") or "")
        another_stack_selected = bool(
            selected_stack and selected_stack != robot and not repair_needed
        )
        start_blocked = (
            bool(blocked_reason)
            or another_stack_selected
            or lifecycle_state in {"starting", "stopping"}
            or (lifecycle_state == "running" and not repair_needed)
        )
        stop_disabled = repair_needed or another_stack_selected or (
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


def _hardware_stack_robot_function_target(
    bridge: SystemBridge,
    statuses: dict[str, dict] | None = None,
) -> str:
    for stack, target in (
        ("dual robots", "dual robots"),
        ("xarm6", "xarm only"),
        ("ur5e", "ur5e only"),
    ):
        status = (
            dict(statuses.get(stack) or {})
            if statuses is not None
            else bridge.hardware_stack_status(stack)
        )
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


def _function_record_panel(  # noqa: C901 - refresh and cleanup share panel state.
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
        body_cleanup: dict[str, Callable[[], None]] = {"callback": lambda: None}

        def _refresh_body(_e=None) -> None:
            body_cleanup["callback"]()
            body_cleanup["callback"] = lambda: None
            body.clear()
            target = str(target_select.value or "").strip()
            if not target:
                return
            with body:
                body_cleanup["callback"] = _predefined_function_record_body(
                    bridge,
                    target,
                )

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
) -> Callable[[], None]:
    robots = bridge.digital_twin_target_robots(target)
    functions = bridge.digital_twin_function_names()
    execution: dict[str, object] = {
        "busy": False,
        "checking": False,
        "preparing": False,
        "active_function": "",
        "selection_revision": 0,
        "assembly_selection_revision": 0,
        "assembly_task": None,
    }
    move_insert_trial: dict[str, object] = {
        "loading": False,
        "active": False,
        "stop_requested": False,
        "selection": (),
        "status": {},
        "trial_id": "",
        "trial_task": None,
        "readiness_revision": 0,
        "readiness_task": None,
        "background_readiness_key": (),
    }
    insertion_demonstration: dict[str, object] = {
        "loading": False,
        "selection": (),
        "status": {},
        "recording_id": "",
    }
    selection_update: dict[str, object] = {
        "active": False,
        "readiness_task": None,
        "readiness_selection": (),
    }
    hardware_status_cache: dict[str, object] = {
        "updated_at": 0.0,
        "target": "",
        "statuses": {},
    }
    pending_execution: dict[str, object] = {}
    pending_assembly: dict[str, str] = {}
    pending_move_insert_trial: dict[str, str] = {}
    pending_move_insert_recovery: dict[str, str] = {}
    assembly_functions = (
        "pick_approach",
        "pick_grasp",
        "place_approach",
        "place_insert",
        "move_home",
    )

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
        "apply when a pick/place context is active. Independent place_approach may run with "
        "held_part empty. place_insert at assembly_board-v1 requires the held part and a "
        "confirmed supervised move_insert trial. Start System is not required for manual "
        "Function Execution."
    ).classes("text-xs text-slate-500")

    ui.label("Function Execution").classes("text-sm font-semibold mt-2")
    assembly_board_v1_state: dict[str, object] = {
        "refreshing": False,
        "status_loaded": False,
        "status": {},
        "named_position_exists": False,
        "last_failure": "",
        "accept_message": "",
        "accept_success": None,
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
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            assembly_board_v1_accept_button = ui.button(
                "Locate & Accept Board",
                on_click=lambda: _locate_and_accept_assembly_board_v1(),
                icon="location_on",
            ).props("dense outline color=primary")
            assembly_board_v1_accept_guidance = ui.label("").classes(
                "text-xs text-slate-500"
            )
        assembly_board_v1_accept_result = ui.label("").classes(
            "text-xs font-semibold"
        )
        assembly_board_v1_frozen_pose_guidance = ui.label(
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
    assembly_container = ui.column().classes("w-full gap-2 mt-2")

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

    def _robot_function_hardware_snapshot() -> tuple[str, dict[str, dict]]:
        now = time.monotonic()
        cached_statuses = hardware_status_cache.get("statuses")
        if (
            isinstance(cached_statuses, dict)
            and now - float(hardware_status_cache.get("updated_at") or 0.0) <= 0.35
        ):
            return str(hardware_status_cache.get("target") or ""), cached_statuses
        statuses = {
            stack: bridge.hardware_stack_status(stack)
            for stack in ("dual robots", "xarm6", "ur5e")
        }
        hardware_target = _hardware_stack_robot_function_target(
            bridge,
            statuses,
        )
        hardware_status_cache.update(
            {
                "updated_at": now,
                "target": hardware_target,
                "statuses": statuses,
            }
        )
        return hardware_target, statuses

    def _physical_function_execution_selected() -> bool:
        hardware_target, _hardware_statuses = _robot_function_hardware_snapshot()
        return bool(
            str(getattr(bridge, "execution_mode", "") or "").strip().lower()
            == "physical"
            or str(getattr(bridge, "robot_env", "") or "").strip().lower()
            == "real"
            or hardware_target
        )

    def _operator_held_part_option_visible() -> bool:
        return bool(
            _physical_function_execution_selected()
            and _current_robot() == "ur5e"
            and _current_function() == "place_approach"
            and _current_destination_location() == "assembly_board-v1"
            and _current_part_name() in _MOVE_INSERT_SUPPORTED_PARTS
            and not bridge.digital_twin_function_held_part("ur5e")
        )

    def _operator_confirmed_held_part() -> bool:
        return _operator_held_part_option_visible()

    def _operator_held_part_origin_resource_location() -> str:
        try:
            options = bridge.digital_twin_function_location_options(
                _current_robot(),
                "pick_approach",
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            log.exception(
                "failed to resolve operator-confirmed %s pick recording origin",
                _current_part_name(),
            )
            return ""
        if "prusa-mk4-2" in options:
            return "prusa-mk4-2"
        return str(options[0] if options else "")

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
        accept_visible = function_name == "place_approach"
        assembly_board_v1_accept_button.set_visibility(accept_visible)
        assembly_board_v1_accept_guidance.set_visibility(accept_visible)
        assembly_board_v1_accept_result.set_visibility(
            bool(accept_visible and assembly_board_v1_state.get("accept_message"))
        )
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
        ready_to_accept = bool(state_matches and status.get("ready_to_accept"))
        accept_controls_idle = bool(
            not execution.get("busy")
            and not execution.get("checking")
            and not execution.get("preparing")
            and not assembly_board_v1_state.get("refreshing")
        )
        assembly_board_v1_accept_button.set_enabled(
            bool(accept_visible and ready_to_accept and accept_controls_idle)
        )
        assembly_board_v1_accept_guidance.set_text(
            "Ready to accept the current stable 10-frame ArUco ID 70 pose. "
            "This does not move the robot."
            if ready_to_accept
            else (
                "Make ArUco ID 70 visible and wait for a fresh stable "
                f"{required_sample_count}/{required_sample_count}-sample pose before "
                "accepting it for Capture Pose."
                if state_matches
                else "Checking whether ArUco ID 70 is ready to accept..."
            )
        )
        accept_message = str(
            assembly_board_v1_state.get("accept_message") or ""
        ).strip()
        assembly_board_v1_accept_result.set_text(accept_message)
        assembly_board_v1_accept_result.classes(
            replace=(
                "text-xs font-semibold text-green-700"
                if assembly_board_v1_state.get("accept_success") is True
                else "text-xs font-semibold text-red-700"
            )
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
            if status.get("excessive_movement") and not status.get(
                "movement_blocked"
            ):
                movement_text += (
                    " This cross-view difference is diagnostic and does not replace the "
                    "accepted board pose. Use Locate & Accept Board if the board actually "
                    "moved."
                )
        assembly_board_v1_movement.set_text(movement_text)
        movement_is_blocked = bool(status.get("movement_blocked"))
        movement_is_diagnostic = bool(
            status.get("excessive_movement") and not movement_is_blocked
        )
        assembly_board_v1_movement.classes(
            replace=(
                "text-xs text-red-700"
                if movement_is_blocked
                else (
                    "text-xs text-amber-700"
                    if movement_is_diagnostic
                    else "text-xs text-slate-600"
                )
            )
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
        assembly_board_v1_frozen_pose_guidance.set_visibility(
            function_name == "place_insert"
        )
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
        _render_assembly_board_v1_readiness()
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

    async def _locate_and_accept_assembly_board_v1() -> None:
        client = _current_client()
        if not (
            _current_function() == "place_approach"
            and _current_destination_location() == "assembly_board-v1"
        ):
            return
        if (
            execution.get("busy")
            or execution.get("checking")
            or execution.get("preparing")
        ):
            _notify(
                "Locate & Accept Board is unavailable during another check or execution.",
                type="warning",
                client=client,
            )
            return
        selection = _assembly_board_v1_selection()
        status = dict(assembly_board_v1_state.get("status") or {})
        if (
            not assembly_board_v1_state.get("status_loaded")
            or assembly_board_v1_state.get("selection") != selection
            or not status.get("ready_to_accept")
        ):
            _notify(
                "ArUco ID 70 needs a fresh stable 10-frame pose before acceptance.",
                type="warning",
                client=client,
            )
            return
        robot = selection[0]
        part_name = _current_part_name()
        execution.update(
            {
                "checking": True,
                "active_function": "Locate & Accept Board",
            }
        )
        assembly_board_v1_accept_button.props("loading")
        assembly_board_v1_state.update(
            {
                "accept_message": "Accepting the current ArUco ID 70 pose...",
                "accept_success": None,
            }
        )
        _render_execution()
        try:
            result = await asyncio.to_thread(
                bridge.perception_locate_and_accept_assembly_board_v1,
                robot,
            )
            if selection != _assembly_board_v1_selection():
                _notify(
                    str(result.get("message") or "Board pose accepted."),
                    type="positive",
                    timeout=6000,
                    client=client,
                )
                return
            if not result.get("success"):
                raise RuntimeError(
                    str(result.get("message") or "Board pose was not accepted.")
                )
            bridge.digital_twin_clear_function_steps(
                target,
                robot,
                "place_approach",
                "assembly_board-v1",
                part_name=part_name,
            )
            execution["selection_revision"] = int(
                execution["selection_revision"]
            ) + 1
            execution["assembly_selection_revision"] = int(
                execution["assembly_selection_revision"]
            ) + 1
            pending_execution.clear()
            pending_assembly.clear()
            run_confirm.close()
            assembly_confirm.close()
            _invalidate_move_insert_trial()
            message = str(result.get("message") or "Board pose accepted.")
            assembly_board_v1_state.update(
                {
                    "status_loaded": True,
                    "status": dict(result),
                    "last_failure": "",
                    "accept_message": message,
                    "accept_success": True,
                    "selection": selection,
                }
            )
            _notify(message, type="positive", timeout=6000, client=client)
            await _refresh_assembly_board_v1_readiness()
            await _load_move_insert_trial_readiness()
            _render_steps()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if selection == _assembly_board_v1_selection():
                assembly_board_v1_state.update(
                    {
                        "accept_message": str(exc),
                        "accept_success": False,
                    }
                )
            _notify(str(exc), type="warning", timeout=7000, client=client)
        finally:
            execution.update({"checking": False, "active_function": ""})
            if _client_alive(assembly_board_v1_accept_button):
                assembly_board_v1_accept_button.props(remove="loading")
                _render_execution()
                _render_steps()

    def _default_location(function_name: str, options: list[str]) -> str:
        if function_name in {"pick_approach", "pick_grasp"} and "prusa-mk4-2" in options:
            return "prusa-mk4-2"
        if function_name in {"place_approach", "place_insert"} and "assembly_board-v1" in options:
            return "assembly_board-v1"
        return options[0] if options else ""

    def _sync_location() -> None:
        previous_active = bool(selection_update["active"])
        selection_update["active"] = True
        try:
            function_name = _current_function()
            location_argument = bridge.digital_twin_function_location_argument(
                function_name
            )
            options = bridge.digital_twin_function_location_options(
                _current_robot(),
                function_name,
            )
            selected = _default_location(function_name, options)
            origin_select.options = (
                options if location_argument == "origin_resource_location" else []
            )
            if origin_select.value not in origin_select.options:
                origin_select.value = (
                    selected
                    if location_argument == "origin_resource_location"
                    else ""
                )
            origin_select.set_visibility(
                location_argument == "origin_resource_location"
            )
            origin_select.update()
            destination_select.options = (
                options if location_argument == "destination_location" else []
            )
            if destination_select.value not in destination_select.options:
                destination_select.value = (
                    selected
                    if location_argument == "destination_location"
                    else ""
                )
            destination_select.set_visibility(
                location_argument == "destination_location"
            )
            destination_select.update()
        finally:
            selection_update["active"] = previous_active

    def _sync_part_name() -> None:
        previous_active = bool(selection_update["active"])
        selection_update["active"] = True
        try:
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
            held_part = (
                bridge.digital_twin_function_held_part(_current_robot())
                if function_name in {"place_approach", "place_insert"}
                else ""
            )
            if held_part in options:
                part_select.value = held_part
            elif part_select.value not in options:
                part_select.value = (
                    "MG" if "MG" in options else (options[0] if options else "")
                )
            part_select.set_visibility(visible)
            part_select.update()
        finally:
            selection_update["active"] = previous_active

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

    def _confirmation_description(
        function_name: str,
        values: dict[str, object],
        *,
        operator_confirmed_held_part: bool = False,
        operator_handoff_origin_resource_location: str = "",
    ) -> str:
        robot = _current_robot()
        part_name = str(values.get("part_name") or "")
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
            destination_location = str(values.get("destination_location") or "")
            if operator_confirmed_held_part:
                return (
                    f"This standalone place_approach run assumes {part_name} is physically "
                    "clamped in "
                    "the UR5e gripper. The UR5e will verify the confirmed "
                    "pick_approach.descend handoff and fresh robot TF using the confirmed "
                    "pick recording "
                    f"from {operator_handoff_origin_resource_location or 'unavailable'} before "
                    f"motion. It will then rerun place_approach and retain the exact {part_name} "
                    "handoff. On success, return to place_insert and use Supervised Test "
                    f"move_insert. This does not insert, release, or qualify {part_name}, and Assembly "
                    "never uses this custody recovery."
                )
            if not bridge.digital_twin_function_held_part(robot):
                if destination_location == "assembly_board-v1":
                    return (
                        f"The {robot} will run place_approach independently with held_part "
                        "empty. It will use the configured assembly_board-v1 observation "
                        "position, collect ten fresh ArUco ID 70 observations, and then move "
                        "through freshly computed approach and descend targets without a part."
                    )
                return (
                    f"The {robot} will run place_approach independently with held_part "
                    f"empty and move through computed targets for {destination_location} "
                    "without a part."
                )
            if destination_location == "assembly_board-v1":
                return (
                    f"The {robot} will first stage {part_name} at the configured "
                    "assembly_board-v1 observation position, collect ten fresh ArUco ID 70 "
                    "observations, and automatically accept or reaccept the current board "
                    "pose if needed. It will then use that frozen pose for freshly computed "
                    "approach and descend targets. If localization fails, it remains at the "
                    "observation position and does not approach the board."
                )
            return (
                f"The {robot} will move {part_name} to computed approach and descend targets "
                f"for {destination_location}."
            )
        if function_name == "place_insert":
            if not bridge.digital_twin_function_held_part(robot):
                if values.get("destination_location") == "assembly_board-v1":
                    return (
                        f"The {robot} cannot run place_insert at assembly_board-v1 with "
                        "held_part empty because Function Execution has no retained "
                        "pick_grasp handoff. Operator-confirmed place_approach transport "
                        "does not authorize Supervised Test move_insert."
                    )
                return (
                    f"The {robot} will run place_insert independently with held_part empty. "
                    "It will open the empty gripper and retreat 0.08 m without advancing "
                    "the pick/place sequence."
                )
            return (
                f"The {robot} will keep {part_name} clamped while the internal move_insert "
                f"step searches and seats it at {values.get('destination_location', '')}. "
                "Only after move_insert succeeds will place_insert release the part "
                "and lift away. Releasing the part is irreversible."
            )
        return f"The {robot} will move to the configured home named position."

    def _current_assembly_origin_resource_location() -> str:
        return str(assembly_origin_select.value or "").strip()

    def _current_assembly_destination_location() -> str:
        return str(assembly_destination_select.value or "").strip()

    def _current_assembly_part_name() -> str:
        return str(assembly_part_select.value or "").strip()

    def _assembly_kwargs() -> dict[str, str]:
        return {
            "origin_resource_location": _current_assembly_origin_resource_location(),
            "destination_location": _current_assembly_destination_location(),
            "part_name": _current_assembly_part_name(),
        }

    def _assembly_selection_signature() -> tuple[str, str, str, str]:
        return (
            _current_robot(),
            _current_assembly_origin_resource_location(),
            _current_assembly_destination_location(),
            _current_assembly_part_name(),
        )

    def _assembly_correction_states(
        function_name: str,
        name: str,
    ) -> tuple[str, str]:
        robot = _current_robot()
        part_name = _current_assembly_part_name()
        buffered_steps = {
            str(step.get("step_name") or "<unnamed>"): dict(step)
            for step in bridge.digital_twin_list_function_buffer_steps(
                target,
                robot,
                function_name,
                name,
                part_name=part_name,
            )
        }
        saved_steps = {
            str(step.get("step_name") or "<unnamed>"): dict(step)
            for step in bridge.digital_twin_list_function_file_steps(
                target,
                robot,
                function_name,
                name,
                part_name=part_name,
            )
        }
        template_step_names = [
            str(step.get("step_name") or "")
            for step in bridge.digital_twin_function_template(function_name)
            if bool(step.get("recordable")) and str(step.get("step_name") or "")
        ]
        extra_step_names = sorted(
            (set(buffered_steps) | set(saved_steps)) - set(template_step_names)
        )
        step_states: list[tuple[str, str]] = []
        for step_name in [*template_step_names, *extra_step_names]:
            if step_name in buffered_steps:
                state = "buffered"
            elif (
                step_name in saved_steps
                and saved_steps[step_name].get("confirmed") is not True
            ):
                state = "unconfirmed"
            elif step_name in saved_steps:
                state = "active"
            else:
                state = "missing"
            step_states.append((step_name, state))

        states = {state for _step_name, state in step_states}
        if "buffered" in states:
            overall_state = "buffered"
        elif "unconfirmed" in states:
            overall_state = "unconfirmed"
        elif states == {"active"}:
            overall_state = "active"
        else:
            overall_state = "missing"
        message = " | ".join(
            f"{function_name}.{step_name}: {state}"
            for step_name, state in step_states
        )
        if states & {"buffered", "unconfirmed"}:
            message += (
                ". Assembly is blocked; use Save/Replace Pose or Clear Position."
            )
        elif "missing" in states:
            message += ". Missing optional robot corrections are allowed."
        else:
            message += "."
        return overall_state, message

    def _refresh_assembly_correction_status() -> None:
        rows = (
            (
                "pick_approach",
                _current_assembly_origin_resource_location(),
                assembly_pick_approach_correction_status,
            ),
            (
                "place_approach",
                _current_assembly_destination_location(),
                assembly_place_approach_correction_status,
            ),
        )
        state_classes = {
            "active": "text-xs text-green-700",
            "missing": "text-xs text-blue-700",
            "buffered": "text-xs text-amber-700",
            "unconfirmed": "text-xs text-amber-700",
        }
        for function_name, name, label in rows:
            try:
                state, message = _assembly_correction_states(function_name, name)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                log.exception("failed to load %s Assembly correction state", function_name)
                state = "unconfirmed"
                message = (
                    f"{function_name}: correction state unavailable ({exc}); Assembly "
                    "readiness remains authoritative."
                )
            label.set_text(message)
            label.classes(replace=state_classes[state])

    def _sync_assembly_options() -> None:
        previous_active = bool(selection_update["active"])
        selection_update["active"] = True
        try:
            robot = _current_robot()
            try:
                origin_options = bridge.digital_twin_function_location_options(
                    robot,
                    "pick_approach",
                )
                destination_options = bridge.digital_twin_function_location_options(
                    robot,
                    "place_approach",
                )
                part_options = bridge.digital_twin_function_part_options(
                    "pick_approach"
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                log.exception("failed to load Assembly options for %s", robot)
                origin_options = []
                destination_options = []
                part_options = []

            assembly_origin_select.options = origin_options
            if assembly_origin_select.value not in origin_options:
                assembly_origin_select.value = (
                    "prusa-mk4-2"
                    if "prusa-mk4-2" in origin_options
                    else (origin_options[0] if origin_options else "")
                )
            assembly_origin_select.update()

            assembly_destination_select.options = destination_options
            if assembly_destination_select.value not in destination_options:
                assembly_destination_select.value = (
                    "assembly_board-v1"
                    if "assembly_board-v1" in destination_options
                    else (destination_options[0] if destination_options else "")
                )
            assembly_destination_select.update()

            assembly_part_select.options = part_options
            if assembly_part_select.value not in part_options:
                assembly_part_select.value = (
                    "MG"
                    if "MG" in part_options
                    else (part_options[0] if part_options else "")
                )
            assembly_part_select.update()
            _refresh_assembly_correction_status()
        finally:
            selection_update["active"] = previous_active

    def _reset_assembly_status() -> None:
        assembly_status.set_text("Assembly readiness is checked automatically without motion.")
        assembly_status.classes(replace="text-xs text-slate-500")
        assembly_progress_status.set_text(
            "No Assembly is active. Completed functions: none."
        )
        assembly_progress_status.classes(replace="text-xs text-slate-500")

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
                    operator_confirmed_held_part = bool(
                        values.pop("operator_confirmed_held_part", False)
                    )
                    operator_handoff_origin_resource_location = str(
                        values.pop("operator_handoff_origin_resource_location", "") or ""
                    )
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
                            origin_resource_location=(
                                operator_handoff_origin_resource_location
                                if operator_confirmed_held_part
                                else values["origin_resource_location"]
                            ),
                            destination_location=values["destination_location"],
                            part_name=values["part_name"],
                            operator_confirmed_held_part=(
                                operator_confirmed_held_part
                            ),
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
            _sync_part_name()
            values = _execution_kwargs()
            operator_confirmed_held_part = _operator_confirmed_held_part()
            operator_handoff_origin_resource_location = (
                _operator_held_part_origin_resource_location()
                if operator_confirmed_held_part
                else ""
            )
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
                    origin_resource_location=(
                        operator_handoff_origin_resource_location
                        if operator_confirmed_held_part
                        else values["origin_resource_location"]
                    ),
                    destination_location=values["destination_location"],
                    part_name=values["part_name"],
                    operator_confirmed_held_part=operator_confirmed_held_part,
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
                        "operator_confirmed_held_part": operator_confirmed_held_part,
                        "operator_handoff_origin_resource_location": (
                            operator_handoff_origin_resource_location
                        ),
                        **values,
                    }
                )
                run_confirm_title.set_text(f"Run {function_name} on the physical {robot}?")
                run_confirm_text.set_text(
                    _confirmation_description(
                        function_name,
                        values,
                        operator_confirmed_held_part=operator_confirmed_held_part,
                        operator_handoff_origin_resource_location=(
                            operator_handoff_origin_resource_location
                        ),
                    )
                )
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

    def _insertion_demonstration_selection() -> tuple[str, str, str, str, str]:
        return (
            target,
            _current_robot(),
            _current_destination_location(),
            _current_part_name(),
            str(getattr(bridge, "execution_mode", "") or ""),
        )

    def _apply_insertion_demonstration_status(
        result: dict[str, object],
        selection: tuple[str, str, str, str, str],
    ) -> None:
        if selection != _insertion_demonstration_selection():
            return
        insertion_demonstration["selection"] = selection
        insertion_demonstration["status"] = dict(result)
        insertion_demonstration["recording_id"] = str(
            result.get("recording_id")
            or insertion_demonstration.get("recording_id")
            or ""
        )

    async def _load_insertion_demonstration_readiness() -> None:
        if _current_function() != "place_insert":
            return
        selection = _insertion_demonstration_selection()
        insertion_demonstration["loading"] = True
        _render_insertion_demonstration()
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_insertion_recording_status,
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                recording_id=str(
                    insertion_demonstration.get("recording_id") or ""
                ),
            )
            if selection == _insertion_demonstration_selection():
                _apply_insertion_demonstration_status(dict(result), selection)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            log.exception("insertion demonstration readiness failed")
            if selection == _insertion_demonstration_selection():
                _apply_insertion_demonstration_status(
                    {
                        "success": False,
                        "ready": False,
                        "active": False,
                        "state": "not_recorded",
                        "message": f"Insertion demonstration readiness failed: {exc}",
                    },
                    selection,
                )
        finally:
            insertion_demonstration["loading"] = False
            if _client_alive(insertion_demonstration_container):
                _render_insertion_demonstration()
                _render_execution()

    async def _start_insertion_recording() -> None:
        client = _current_client()
        selection = _insertion_demonstration_selection()
        insertion_demonstration["loading"] = True
        start_insertion_recording_button.props("loading")
        _render_insertion_demonstration()
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_start_insertion_recording,
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                confirmed=True,
            )
            _apply_insertion_demonstration_status(dict(result), selection)
            _notify(
                str(result.get("message") or "Insertion recording start processed."),
                type="positive" if result.get("success") else "negative",
                timeout=7000,
                client=client,
            )
        finally:
            insertion_demonstration["loading"] = False
            if _client_alive(start_insertion_recording_button):
                start_insertion_recording_button.props(remove="loading")
                _render_insertion_demonstration()
                _render_execution()

    async def _save_insertion_recording() -> None:
        client = _current_client()
        selection = _insertion_demonstration_selection()
        save_insertion_recording_button.props("loading")
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_save_insertion_recording,
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                recording_id=str(insertion_demonstration.get("recording_id") or ""),
            )
            _apply_insertion_demonstration_status(dict(result), selection)
            if (
                result.get("success")
                and str(result.get("state") or "")
                == "recording_saved_return_to_pre_insertion"
            ):
                _invalidate_move_insert_trial()
                await _load_move_insert_trial_readiness()
            _notify(
                str(result.get("message") or "Save Recording processed."),
                type="positive" if result.get("success") else "warning",
                timeout=7000,
                client=client,
            )
        finally:
            if _client_alive(save_insertion_recording_button):
                save_insertion_recording_button.props(remove="loading")
                _render_insertion_demonstration()
                _render_execution()

    async def _cancel_insertion_recording() -> None:
        client = _current_client()
        selection = _insertion_demonstration_selection()
        cancel_insertion_recording_button.props("loading")
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_cancel_insertion_recording,
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                recording_id=str(insertion_demonstration.get("recording_id") or ""),
                note="",
            )
            _apply_insertion_demonstration_status(dict(result), selection)
            _notify(
                str(result.get("message") or "Cancel Recording processed."),
                type="warning",
                timeout=7000,
                client=client,
            )
        finally:
            if _client_alive(cancel_insertion_recording_button):
                cancel_insertion_recording_button.props(remove="loading")
                _render_insertion_demonstration()
                _render_execution()

    async def _delete_insertion_recording() -> None:
        client = _current_client()
        selection = _insertion_demonstration_selection()
        delete_insertion_recording_confirm.close()
        delete_insertion_recording_button.props("loading")
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_delete_insertion_recording,
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                confirmed=True,
            )
            _apply_insertion_demonstration_status(dict(result), selection)
            _notify(
                str(result.get("message") or "Delete Previous Recording processed."),
                type="positive" if result.get("success") else "negative",
                timeout=7000,
                client=client,
            )
        finally:
            if _client_alive(delete_insertion_recording_button):
                delete_insertion_recording_button.props(remove="loading")
                _render_insertion_demonstration()
                _render_execution()

    async def _reanalyze_insertion_recording() -> None:
        client = _current_client()
        selection = _insertion_demonstration_selection()
        reanalyze_insertion_recording_button.props("loading")
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_reanalyze_insertion_recording,
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                recording_id=str(
                    insertion_demonstration.get("recording_id") or ""
                ),
            )
            _apply_insertion_demonstration_status(dict(result), selection)
            if (
                result.get("success")
                and str(result.get("state") or "")
                == "recording_saved_return_to_pre_insertion"
            ):
                _invalidate_move_insert_trial()
                await _load_move_insert_trial_readiness()
            _notify(
                str(
                    result.get("message")
                    or "Reanalyze Saved Recording processed."
                ),
                type="positive" if result.get("success") else "warning",
                timeout=7000,
                client=client,
            )
        finally:
            if _client_alive(reanalyze_insertion_recording_button):
                reanalyze_insertion_recording_button.props(remove="loading")
                _render_insertion_demonstration()
                _render_execution()

    def _download_insertion_recording_bundle() -> None:
        status = dict(insertion_demonstration.get("status") or {})
        raw_path = str(
            status.get("download_path")
            or status.get("diagnostic_bundle_path")
            or ""
        ).strip()
        allowed_root = (
            Path("~/.local/share/cais-spade-llm/move_insert_demonstrations")
            .expanduser()
            .resolve()
        )
        try:
            bundle_path = Path(raw_path).expanduser().resolve(strict=True)
            bundle_path.relative_to(allowed_root)
        except (OSError, RuntimeError, ValueError):
            _notify("The insertion recording bundle path is unavailable or unsafe.", type="negative")
            return
        if bundle_path.name != "diagnostic_bundle.zip":
            _notify("The insertion recording bundle is invalid.", type="negative")
            return
        recording_id = str(
            insertion_demonstration.get("recording_id") or "insertion_demonstration"
        )
        ui.download(bundle_path, f"{recording_id}-diagnostic_bundle.zip")

    def _move_insert_trial_selection() -> tuple[str, str, str, str, str]:
        return (
            target,
            _current_robot(),
            _current_destination_location(),
            _current_part_name(),
            str(getattr(bridge, "execution_mode", "") or ""),
        )

    def _move_insert_trial_state() -> str:
        status = dict(move_insert_trial.get("status") or {})
        state = str(
            status.get("state")
            or status.get("qualification_state")
            or "not_confirmed"
        ).strip()
        if state not in {
            "not_confirmed",
            "ready_to_test",
            "testing",
            "awaiting_visual_confirmation",
            "confirmation_progress",
            "confirmed",
            "failure_recorded",
        }:
            return "not_confirmed"
        if state == "testing" and not bool(
            move_insert_trial.get("active") or status.get("active")
        ):
            return "not_confirmed"
        return state

    def _move_insert_trial_is_current() -> bool:
        return bool(
            move_insert_trial.get("selection") == _move_insert_trial_selection()
        )

    def _move_insert_trial_is_qualified() -> bool:
        status = dict(move_insert_trial.get("status") or {})
        return bool(_move_insert_trial_is_current() and status.get("qualified"))

    def _apply_move_insert_trial_status(
        result: dict[str, object],
        selection: tuple[str, str, str, str, str],
    ) -> bool:
        if selection != _move_insert_trial_selection():
            return False
        previous = (
            move_insert_trial.get("selection"),
            dict(move_insert_trial.get("status") or {}),
            str(move_insert_trial.get("trial_id") or ""),
            bool(move_insert_trial.get("active")),
        )
        move_insert_trial["selection"] = selection
        move_insert_trial["status"] = dict(result)
        trial_id = str(result.get("trial_id") or move_insert_trial.get("trial_id") or "")
        move_insert_trial["trial_id"] = trial_id
        move_insert_trial["active"] = bool(result.get("active"))
        current = (
            move_insert_trial.get("selection"),
            dict(move_insert_trial.get("status") or {}),
            str(move_insert_trial.get("trial_id") or ""),
            bool(move_insert_trial.get("active")),
        )
        return current != previous

    def _move_insert_trial_progress_message(status: dict[str, object]) -> str:
        phase_status = dict(status.get("controller_status") or {})
        phase = str(
            status.get("insert_phase")
            or status.get("phase")
            or phase_status.get("insert_phase")
            or phase_status.get("phase")
            or ""
        ).strip()
        try:
            relief_cycle = int(
                status.get("relief_cycle_count")
                or phase_status.get("relief_cycle_count")
                or 0
            )
        except (TypeError, ValueError, OverflowError):
            relief_cycle = 0
        try:
            disengagement_cycle = int(
                status.get("disengagement_cycle_count")
                or phase_status.get("disengagement_cycle_count")
                or 0
            )
        except (TypeError, ValueError, OverflowError):
            disengagement_cycle = 0
        cycle_text = f" (cycle {max(1, relief_cycle)}/3)"
        disengagement_cycle_text = (
            f" (cycle {max(1, disengagement_cycle)}/6)"
        )
        tactile_center_valid = bool(
            status.get("tactile_center_valid")
            or phase_status.get("tactile_center_valid")
        )
        search_peck_state = str(
            status.get("search_peck_state")
            or status.get("insert_search_peck_state")
            or phase_status.get("search_peck_state")
            or phase_status.get("insert_search_peck_state")
            or ""
        )
        phase_messages = {
            "searching": (
                "Searching for pin center — local spiral. The part remains clamped."
            ),
            "expanded_searching": (
                "No entry detected — expanding touch search through 3 mm, 5 mm, "
                "then 10 mm. The part remains clamped."
            ),
            "cocked": (
                f"{_current_part_name()} appears cocked — withdrawing completely"
                f"{disengagement_cycle_text}. The part remains clamped."
            ),
            "disengaging": (
                f"{_current_part_name()} appears cocked — withdrawing completely"
                f"{disengagement_cycle_text}. The part remains clamped."
            ),
            "recentering": (
                "Returning above tactile center"
                f"{disengagement_cycle_text}. The part remains clamped."
            ),
            "retaring": (
                "Returning above tactile center and retaring"
                f"{disengagement_cycle_text}. The part remains clamped."
            ),
            "retrying": (
                "Alignment normal — retrying insertion"
                f"{disengagement_cycle_text}. The part remains clamped."
            ),
            "relieving": (
                "Soft load limit detected — unloading force first"
                f"{cycle_text}. The part remains clamped."
            ),
            "backing_off": (
                "Load persisted — performing the bounded micro-backoff"
                f"{cycle_text}. The part remains clamped."
            ),
            "resuming": (
                "Load cleared — retrying direct insertion"
                f"{cycle_text}. The part remains clamped."
            ),
            "seating": (
                "Checking engagement and seating. The part remains clamped."
            ),
            "settling": (
                "Checking engagement and stable seating. The part remains clamped."
            ),
        }
        if phase == "seating" and tactile_center_valid:
            return (
                "Pin capture detected — inserting straight. The part remains clamped."
            )
        if phase in {"searching", "expanded_searching"}:
            if search_peck_state == "unloading":
                return (
                    "Pin entry stalled — lifting slightly while advancing the "
                    "spiral. The part remains clamped."
                )
            if search_peck_state == "descending":
                return (
                    "Trying the next spiral position with low downward preload. "
                    "The part remains clamped."
                )
        return phase_messages.get(phase, str(status.get("message") or ""))

    async def _load_move_insert_trial_readiness(  # noqa: C901 - exact pending recovery states.
        *,
        open_confirmation: bool = False,
        background: bool = False,
    ) -> dict[str, object]:
        client = _current_client()
        selection = _move_insert_trial_selection()
        if _current_function() != "place_insert":
            return {}
        readiness_revision = int(move_insert_trial["readiness_revision"]) + 1
        move_insert_trial["readiness_revision"] = readiness_revision
        status_changed = False
        if not background:
            move_insert_trial["loading"] = True
            _render_move_insert_trial()
        try:
            current_status = await asyncio.to_thread(
                bridge.digital_twin_move_insert_trial_status,
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                trial_id=str(move_insert_trial.get("trial_id") or ""),
            )
            if (
                selection != _move_insert_trial_selection()
                or readiness_revision != move_insert_trial["readiness_revision"]
            ):
                return {}
            current_state = str(
                current_status.get("state")
                or current_status.get("qualification_state")
                or "not_confirmed"
            )
            pending_review = bool(
                current_status.get("active")
                or current_status.get("review_required")
                or current_status.get("recovery_required")
                or current_status.get("hardware_stack_repair_required")
                or current_status.get("normal_repair_required")
                or current_state == "awaiting_visual_confirmation"
            )
            if current_status.get("qualified") or pending_review:
                status_changed = _apply_move_insert_trial_status(
                    dict(current_status), selection
                )
                if open_confirmation:
                    _notify(
                        str(
                            current_status.get("message")
                            or "Finish the current move_insert review before another test."
                        ),
                        type="warning",
                        timeout=7000,
                        client=client,
                    )
                return dict(current_status)
            result = await bridge.digital_twin_move_insert_trial_readiness(
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
            )
            if (
                selection != _move_insert_trial_selection()
                or readiness_revision != move_insert_trial["readiness_revision"]
            ):
                return {}
            status_changed = _apply_move_insert_trial_status(
                dict(result), selection
            )
            message = str(result.get("message") or "")
            if not result.get("ready"):
                pending_move_insert_trial.clear()
                move_insert_trial_confirm.close()
            if open_confirmation:
                if not result.get("success") or not result.get("ready"):
                    _notify(message, type="warning", timeout=7000, client=client)
                else:
                    pending_move_insert_trial.clear()
                    pending_move_insert_trial.update(
                        {
                            "robot": _current_robot(),
                            "destination_location": _current_destination_location(),
                            "part_name": _current_part_name(),
                        }
                    )
                    move_insert_trial_confirm_title.set_text(
                        "Run Supervised Test move_insert on the physical "
                        f"{_current_robot()}?"
                    )
                    move_insert_trial_confirm.open()
            return dict(result)
        except Exception as exc:
            log.exception("move_insert trial readiness check failed")
            result = {
                "success": False,
                "ready": False,
                "state": "not_confirmed",
                "qualified": False,
                "message": f"move_insert trial readiness failed: {exc}",
            }
            if (
                selection == _move_insert_trial_selection()
                and readiness_revision == move_insert_trial["readiness_revision"]
            ):
                status_changed = _apply_move_insert_trial_status(result, selection)
            if open_confirmation:
                _notify(
                    result["message"],
                    type="negative",
                    timeout=7000,
                    client=client,
                )
            return result
        finally:
            if readiness_revision == move_insert_trial["readiness_revision"]:
                if not background:
                    move_insert_trial["loading"] = False
                if (
                    _client_alive(move_insert_trial_container)
                    and (not background or status_changed)
                ):
                    _render_move_insert_trial()
                    _render_execution()

    async def _check_move_insert_trial_readiness() -> None:
        if execution.get("busy") or execution.get("checking"):
            _notify("A robot function check or execution is already active.", type="warning")
            return
        execution.update({"checking": True, "active_function": "move_insert trial readiness"})
        supervised_move_insert_button.props("loading")
        _render_execution()
        try:
            await _load_move_insert_trial_readiness(open_confirmation=True)
        finally:
            execution.update({"checking": False, "active_function": ""})
            if _client_alive(supervised_move_insert_button):
                supervised_move_insert_button.props(remove="loading")
                _render_execution()

    async def _confirmed_execute_move_insert_trial() -> None:
        client = _current_client()
        move_insert_trial_confirm.close()
        if execution.get("busy") or execution.get("checking"):
            _notify("A robot function check or execution is already active.", type="warning")
            return
        values = dict(pending_move_insert_trial)
        if not values or values != {
            "robot": _current_robot(),
            "destination_location": _current_destination_location(),
            "part_name": _current_part_name(),
        }:
            _notify(
                "The move_insert selection changed. Check readiness again.",
                type="warning",
                client=client,
            )
            return
        selection = _move_insert_trial_selection()
        execution.update({"busy": True, "active_function": "move_insert trial"})
        move_insert_trial.update(
            {
                "active": True,
                "stop_requested": False,
                "selection": selection,
            }
        )
        status = dict(move_insert_trial.get("status") or {})
        status.update(
            {
                "state": "testing",
                "qualification_state": "testing",
                "active": True,
                "message": "Supervised move_insert is running; the part remains clamped.",
            }
        )
        move_insert_trial["status"] = status
        supervised_move_insert_button.props("loading")
        _render_execution()
        task: asyncio.Task | None = None
        backend_may_be_active = True
        try:
            task = asyncio.create_task(
                bridge.digital_twin_execute_move_insert_trial(
                    target,
                    values["robot"],
                    destination_location=values["destination_location"],
                    part_name=values["part_name"],
                    confirmed=True,
                )
            )
            move_insert_trial["trial_task"] = task
            result = await task
            backend_may_be_active = bool(result.get("active"))
            if selection == _move_insert_trial_selection():
                _apply_move_insert_trial_status(dict(result), selection)
                message = str(result.get("message") or "move_insert trial completed.")
                _notify(
                    message,
                    type=(
                        "positive"
                        if result.get("success") and result.get("completion_eligible")
                        else "warning"
                    ),
                    timeout=7000,
                    client=client,
                )
        except asyncio.CancelledError:
            # The bridge shields an accepted physical goal and retains its lock until
            # settlement. Do not make the UI claim that the trial became terminal.
            backend_may_be_active = True
            raise
        except Exception as exc:
            log.exception("Supervised move_insert UI execution failed")
            if selection == _move_insert_trial_selection():
                try:
                    status = await asyncio.to_thread(
                        bridge.digital_twin_move_insert_trial_status,
                        target,
                        values["robot"],
                        destination_location=values["destination_location"],
                        part_name=values["part_name"],
                        trial_id=str(move_insert_trial.get("trial_id") or ""),
                    )
                except Exception:  # noqa: BLE001 - UI must remain fail-closed.
                    log.exception(
                        "Could not refresh supervised move_insert after UI failure"
                    )
                    status = {
                        **dict(move_insert_trial.get("status") or {}),
                        "success": False,
                        "state": "testing",
                        "qualification_state": "testing",
                        "qualified": False,
                        "active": True,
                        "trial_id": str(move_insert_trial.get("trial_id") or ""),
                        "message": (
                            f"Supervised move_insert UI failed: {exc}. Backend "
                            "settlement is unknown; use Stop Supervised move_insert "
                            "and inspect the robot."
                        ),
                    }
                backend_may_be_active = bool(status.get("active"))
                _apply_move_insert_trial_status(status, selection)
                _notify(
                    str(status.get("message") or f"Supervised move_insert failed: {exc}"),
                    type="negative",
                    timeout=7000,
                    client=client,
                )
        finally:
            if move_insert_trial.get("trial_task") is task:
                move_insert_trial["trial_task"] = None
            pending_move_insert_trial.clear()
            current_status = dict(move_insert_trial.get("status") or {})
            move_insert_trial["active"] = bool(
                backend_may_be_active or current_status.get("active")
            )
            execution.update({"busy": False, "active_function": ""})
            if _client_alive(supervised_move_insert_button):
                supervised_move_insert_button.props(remove="loading")
                _render_execution()
                _render_steps()

    async def _stop_move_insert_trial() -> None:
        client = _current_client()
        selection = _move_insert_trial_selection()
        move_insert_trial["stop_requested"] = True
        stop_move_insert_button.props("loading")
        _render_move_insert_trial()
        try:
            task = move_insert_trial.get("trial_task")
            deadline = asyncio.get_running_loop().time() + 15.0
            while True:
                status = await asyncio.to_thread(
                    bridge.digital_twin_move_insert_trial_status,
                    target,
                    _current_robot(),
                    destination_location=_current_destination_location(),
                    part_name=_current_part_name(),
                    trial_id="",
                )
                if bool(status.get("active")):
                    break
                if not isinstance(task, asyncio.Task) or task.done():
                    _apply_move_insert_trial_status(dict(status), selection)
                    _notify(
                        "Supervised move_insert is already terminal; no active "
                        "force or search motion remains to stop.",
                        type="warning",
                        client=client,
                    )
                    return
                if asyncio.get_running_loop().time() >= deadline:
                    _notify(
                        "Stop Supervised move_insert is waiting for the locked "
                        "readiness check to either finish without dispatch or publish "
                        "the exact active trial. Use the physical emergency stop if "
                        "motion is unsafe.",
                        type="negative",
                        timeout=7000,
                        client=client,
                    )
                    return
                await asyncio.sleep(0.1)
            trial_id = str(status.get("trial_id") or "")
            _apply_move_insert_trial_status(dict(status), selection)
            result = await bridge.digital_twin_cancel_move_insert_trial(
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                trial_id=trial_id,
            )
            _apply_move_insert_trial_status(dict(result), selection)
            _notify(
                str(
                    result.get("message")
                    or "Supervised move_insert cancellation requested."
                ),
                type="warning" if result.get("success") else "negative",
                timeout=7000,
                client=client,
            )
        finally:
            move_insert_trial["stop_requested"] = False
            if _client_alive(stop_move_insert_button):
                stop_move_insert_button.props(remove="loading")
                _render_move_insert_trial()

    def _open_move_insert_recovery_confirmation() -> None:
        status = (
            dict(move_insert_trial.get("status") or {})
            if _move_insert_trial_is_current()
            else {}
        )
        trial_id = str(status.get("trial_id") or move_insert_trial.get("trial_id") or "")
        if (
            not status.get("recovery_required")
            or status.get("active")
            or status.get("completion_motion_active")
            or not trial_id
        ):
            _notify(
                "Confirm Physical Recovery is available only for the exact inactive "
                "current supervised move_insert trial with recovery_required.",
                type="warning",
            )
            return
        pending_move_insert_recovery.clear()
        pending_move_insert_recovery.update(
            {
                "target": target,
                "robot": _current_robot(),
                "destination_location": _current_destination_location(),
                "part_name": _current_part_name(),
                "trial_id": trial_id,
            }
        )
        move_insert_recovery_confirm_title.set_text(
            f"Confirm Physical Recovery for {_current_part_name()}?"
        )
        move_insert_recovery_confirm.open()

    async def _confirmed_move_insert_recovery() -> None:
        client = _current_client()
        move_insert_recovery_confirm.close()
        if execution.get("busy") or execution.get("checking"):
            _notify(
                "A robot function check or execution is already active.",
                type="warning",
            )
            return
        values = dict(pending_move_insert_recovery)
        expected_values = {
            "target": target,
            "robot": _current_robot(),
            "destination_location": _current_destination_location(),
            "part_name": _current_part_name(),
            "trial_id": str(move_insert_trial.get("trial_id") or ""),
        }
        if not values or values != expected_values:
            _notify(
                "The exact move_insert recovery selection changed. Open Confirm "
                "Physical Recovery again.",
                type="warning",
                client=client,
            )
            return
        selection = _move_insert_trial_selection()
        execution.update(
            {
                "checking": True,
                "active_function": "move_insert physical recovery confirmation",
            }
        )
        confirm_move_insert_recovery_button.props("loading")
        _render_execution()
        try:
            result = await asyncio.to_thread(
                bridge.digital_twin_confirm_move_insert_recovery,
                values["target"],
                values["robot"],
                destination_location=values["destination_location"],
                part_name=values["part_name"],
                trial_id=values["trial_id"],
                confirmed=True,
            )
            _apply_move_insert_trial_status(dict(result), selection)
            recovery_confirmed = bool(
                result.get("recovery_confirmed_at")
                and not result.get("recovery_required")
            )
            _notify(
                str(
                    result.get("message")
                    or "move_insert physical recovery confirmation was processed."
                ),
                type="positive" if recovery_confirmed else "warning",
                timeout=7000,
                client=client,
            )
        except Exception as exc:
            log.exception("move_insert physical recovery confirmation failed")
            _notify(
                f"move_insert physical recovery confirmation failed: {exc}",
                type="negative",
                timeout=7000,
                client=client,
            )
        finally:
            pending_move_insert_recovery.clear()
            execution.update({"checking": False, "active_function": ""})
            if _client_alive(confirm_move_insert_recovery_button):
                confirm_move_insert_recovery_button.props(remove="loading")
                _render_execution()
                _render_steps()

    async def _confirmed_move_insert_completion() -> None:
        client = _current_client()
        move_insert_completion_confirm.close()
        if execution.get("busy") or execution.get("checking"):
            _notify("A robot function check or execution is already active.", type="warning")
            return
        trial_id = str(move_insert_trial.get("trial_id") or "")
        selection = _move_insert_trial_selection()
        execution.update({"busy": True, "active_function": "move_insert completion"})
        confirm_move_insert_completion_button.props("loading")
        _render_execution()
        try:
            result = await bridge.digital_twin_confirm_move_insert_completion(
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                trial_id=trial_id,
                confirmed=True,
            )
            _apply_move_insert_trial_status(dict(result), selection)
            message = str(result.get("message") or "move_insert completion processed.")
            _notify(
                message,
                type="positive" if result.get("success") else "negative",
                timeout=7000,
                client=client,
            )
        except Exception as exc:
            log.exception("move_insert completion confirmation failed")
            _notify(
                f"move_insert completion confirmation failed: {exc}",
                type="negative",
                timeout=7000,
                client=client,
            )
        finally:
            execution.update({"busy": False, "active_function": ""})
            if _client_alive(confirm_move_insert_completion_button):
                confirm_move_insert_completion_button.props(remove="loading")
                _render_execution()
                _render_steps()

    def _download_move_insert_diagnostic_bundle() -> None:
        status = dict(move_insert_trial.get("status") or {})
        raw_path = str(
            status.get("download_path")
            or status.get("diagnostic_bundle_path")
            or ""
        ).strip()
        if not raw_path:
            _notify("No move_insert diagnostic bundle is available.", type="warning")
            return
        allowed_root = (
            Path("~/.local/share/cais-spade-llm/move_insert_trials")
            .expanduser()
            .resolve()
        )
        try:
            bundle_path = Path(raw_path).expanduser().resolve(strict=True)
            bundle_path.relative_to(allowed_root)
        except (OSError, RuntimeError, ValueError):
            _notify(
                "The move_insert diagnostic bundle path is unavailable or unsafe.",
                type="negative",
                timeout=7000,
            )
            return
        if not bundle_path.is_file() or bundle_path.name != "diagnostic_bundle.zip":
            _notify(
                "The move_insert diagnostic bundle is missing or invalid.",
                type="negative",
                timeout=7000,
            )
            return
        trial_id = str(move_insert_trial.get("trial_id") or "move_insert_trial")
        ui.download(bundle_path, f"{trial_id}-diagnostic_bundle.zip")

    with (
        ui.column().classes("w-full gap-2 mt-2") as insertion_demonstration_container,
        ui.card().classes("w-full p-3"),
    ):
        ui.label("Insertion Demonstration").classes("text-sm font-semibold")
        ui.label(
            "Record one manually guided insertion with CAIS Cartesian jog. Manual jog "
            "speed is retained as diagnostic evidence but does not set the automatic "
            "insertion speed; the jog trace is never replayed as robot motion."
        ).classes("text-xs text-slate-500")
        insertion_demonstration_selection_label = ui.label("").classes(
            "text-xs text-slate-600"
        )
        insertion_demonstration_state_label = ui.label("Not recorded").classes(
            "text-xs font-semibold text-amber-700"
        )
        insertion_demonstration_message = ui.label("").classes(
            "text-xs text-slate-600"
        )
        insertion_demonstration_path = ui.label("").classes(
            "text-xs text-slate-500 break-all"
        )
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            start_insertion_recording_button = ui.button(
                "Start Recording",
                on_click=_start_insertion_recording,
                icon="fiber_manual_record",
            ).props("outline dense")
            save_insertion_recording_button = ui.button(
                "Save Recording",
                on_click=_save_insertion_recording,
                icon="save",
            ).props("outline dense")
            cancel_insertion_recording_button = (
                ui.button(
                    "Cancel Recording",
                    on_click=_cancel_insertion_recording,
                    icon="cancel",
                )
                .props("outline dense")
                .classes("text-red-700")
            )
            download_insertion_recording_button = ui.button(
                "Download Recording Bundle",
                on_click=_download_insertion_recording_bundle,
                icon="download",
            ).props("flat dense")
            reanalyze_insertion_recording_button = ui.button(
                "Reanalyze Saved Recording",
                on_click=_reanalyze_insertion_recording,
                icon="analytics",
            ).props("outline dense")
            delete_insertion_recording_button = ui.button(
                "Delete Previous Recording",
                on_click=lambda: delete_insertion_recording_confirm.open(),
                icon="delete",
            ).props("outline dense color=red")

    with ui.dialog() as delete_insertion_recording_confirm, ui.card().classes(
        "gap-3 max-w-xl"
    ):
        ui.label("Delete previous insertion recording?").classes("font-semibold")
        ui.label(
            "This permanently deletes the exact selected part's recording bundle and "
            "learned recipe, and revokes its qualification. It cannot be recovered."
        ).classes("text-sm text-red-700")
        ui.label(
            "No robot motion is commanded and the physical part remains clamped."
        ).classes("text-xs text-slate-600")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button(
                "Cancel",
                on_click=delete_insertion_recording_confirm.close,
            ).props("flat")
            ui.button(
                "Delete Previous Recording",
                on_click=_delete_insertion_recording,
                icon="delete",
            ).props("color=red")

    def _render_insertion_demonstration() -> None:
        selected = _current_function() == "place_insert"
        insertion_demonstration_container.set_visibility(selected)
        if not selected:
            return
        current = bool(
            insertion_demonstration.get("selection")
            == _insertion_demonstration_selection()
        )
        status = (
            dict(insertion_demonstration.get("status") or {}) if current else {}
        )
        state = str(status.get("state") or "not_recorded")
        trial_current = bool(
            move_insert_trial.get("selection")
            == _move_insert_trial_selection()
        )
        trial_status = (
            dict(move_insert_trial.get("status") or {}) if trial_current else {}
        )
        if (
            state == "recording_saved_return_to_pre_insertion"
            and bool(trial_status.get("ready"))
        ):
            state = "ready_for_supervised_test"
        state_labels = {
            "not_recorded": "Not recorded",
            "recording_baseline": "Preparing force baseline",
            "recording_insertion": "Recording insertion — Cartesian jog ready",
            "saving_recording": "Saving recording",
            "cancelling": "Cancelling",
            "recording_saved_return_to_pre_insertion": (
                "Recording saved — return to pre-insertion"
            ),
            "recording_saved_mg_hard_cap_review_required": (
                f"Recording saved — {_current_part_name()} hard-cap review required"
            ),
            "ready_for_supervised_test": "Ready for supervised test",
        }
        if state not in state_labels:
            state = "not_recorded"
        state_class = (
            "text-xs font-semibold text-green-700"
            if state == "ready_for_supervised_test"
            else "text-xs font-semibold text-blue-700"
            if state in {
                "recording_baseline",
                "recording_insertion",
                "saving_recording",
                "cancelling",
            }
            else "text-xs font-semibold text-amber-700"
        )
        insertion_demonstration_selection_label.set_text(
            f"Selected: robot {_current_robot()}; destination "
            f"{_current_destination_location()}; part {_current_part_name()}."
        )
        insertion_demonstration_state_label.set_text(state_labels[state])
        insertion_demonstration_state_label.classes(replace=state_class)
        insertion_demonstration_message.set_text(
            "Checking insertion demonstration readiness without motion..."
            if insertion_demonstration.get("loading")
            else str(
                status.get("message")
                or "Run place_approach, then Start Recording."
            )
        )
        bundle_path = str(
            status.get("download_path")
            or status.get("diagnostic_bundle_path")
            or ""
        )
        insertion_demonstration_path.set_text(
            f"Recording bundle: {bundle_path}" if bundle_path else ""
        )
        insertion_demonstration_path.set_visibility(bool(bundle_path))
        active = bool(status.get("active"))
        controls_idle = bool(
            not execution.get("busy")
            and not execution.get("checking")
            and not execution.get("preparing")
            and not insertion_demonstration.get("loading")
        )
        start_insertion_recording_button.set_enabled(
            bool(status.get("ready") and controls_idle and not active)
        )
        save_insertion_recording_button.set_enabled(
            bool(
                active
                and state == "recording_insertion"
                and controls_idle
            )
        )
        cancel_insertion_recording_button.set_enabled(active)
        download_insertion_recording_button.set_visibility(bool(bundle_path))
        download_insertion_recording_button.set_enabled(bool(bundle_path))
        reanalyze_visible = bool(
            bundle_path
            and (
                status.get("hard_cap_review_required")
                or status.get("reanalyze_available")
            )
            and not status.get("candidate_recipe")
        )
        reanalyze_insertion_recording_button.set_visibility(reanalyze_visible)
        reanalyze_insertion_recording_button.set_enabled(
            bool(reanalyze_visible and controls_idle and not active)
        )
        delete_insertion_recording_button.set_visibility(
            bool(bundle_path or status.get("candidate_recipe"))
        )
        delete_insertion_recording_button.set_enabled(
            bool(controls_idle and not active)
        )

    with (
        ui.column().classes("w-full gap-2 mt-2") as move_insert_trial_container,
        ui.card().classes("w-full p-3"),
    ):
            ui.label("Supervised move_insert").classes("text-sm font-semibold")
            ui.label(
                "move_insert remains internal to place_insert. There are no operator tuning "
                "values: complete one supervised automatic insertion and visually confirm "
                "it. "
                "A failed test saves diagnostics automatically; delete the previous recording "
                "only when you want to relearn it."
            ).classes("text-xs text-slate-500")
            move_insert_trial_selection_label = ui.label("").classes(
                "text-xs text-slate-600"
            )
            move_insert_trial_state_label = ui.label("Not confirmed").classes(
                "text-xs font-semibold text-amber-700"
            )
            move_insert_trial_message = ui.label("").classes("text-xs text-slate-600")
            move_insert_missing_handoff = ui.label(
                "Function Execution has no retained selected-part handoff. Select "
                "place_approach, then Confirm Run place_approach once. That standalone "
                "run assumes the selected part is "
                "already physically clamped. Then return to place_insert; "
                "Supervised Test move_insert will recheck readiness."
            ).classes("text-xs font-semibold text-amber-700")
            move_insert_trial_diagnostic = ui.label("").classes(
                "text-xs text-slate-500 break-all"
            )
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                supervised_move_insert_button = ui.button(
                    "Supervised Test move_insert",
                    on_click=_check_move_insert_trial_readiness,
                    icon="precision_manufacturing",
                ).props("outline dense")
                stop_move_insert_button = (
                    ui.button(
                        "Stop Supervised move_insert",
                        on_click=_stop_move_insert_trial,
                        icon="stop",
                    )
                    .props("outline dense color=red")
                    .classes("text-red-700")
                )
                ui.label(
                    "Stop cancels active force/search motion, waits for stationary "
                    "feedback, and keeps the selected part clamped. It does not withdraw "
                    "the selected part or run place_approach."
                ).classes("text-xs text-slate-500")
                confirm_move_insert_recovery_button = ui.button(
                    "Confirm Physical Recovery",
                    on_click=_open_move_insert_recovery_confirmation,
                    icon="verified_user",
                ).props("outline dense color=red")
                confirm_move_insert_recovery_button.set_visibility(False)
                confirm_move_insert_completion_button = ui.button(
                    "Confirm Completion",
                    on_click=lambda: move_insert_completion_confirm.open(),
                    icon="check_circle",
                ).props("outline dense")
                download_move_insert_diagnostic_button = ui.button(
                    "Download Diagnostic Bundle",
                    on_click=_download_move_insert_diagnostic_bundle,
                    icon="download",
                ).props("flat dense")

    with ui.dialog() as move_insert_trial_confirm, ui.card().classes("gap-3 max-w-xl"):
        move_insert_trial_confirm_title = ui.label("").classes("font-semibold")
        ui.label(
            "The UR5e will automatically attempt direct compliant insertion and use its "
            "bounded spiral search only if axial progress stalls. Recoverable binding first "
            "unloads force and may use at most three protected micro-backoff cycles before "
            "retrying insertion. The part remains clamped: this test never releases, lifts, "
            "or runs move_home."
        ).classes("text-sm")
        ui.label(
            "Clear the workcell, switch the pendant to Remote Control, and keep the emergency "
            "stop available before continuing."
        ).classes("text-xs text-red-700")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=move_insert_trial_confirm.close).props("flat")
            ui.button(
                "Confirm Supervised Test move_insert",
                on_click=_confirmed_execute_move_insert_trial,
                icon="play_arrow",
            ).props("color=red")

    with ui.dialog() as move_insert_recovery_confirm, ui.card().classes(
        "gap-3 max-w-xl"
    ):
        move_insert_recovery_confirm_title = ui.label("").classes("font-semibold")
        ui.label(
            "Confirm only after the operator physically moved the part clear using "
            "approved manual recovery and verified that the part is still clamped. "
            "This action commands no robot motion: it does not jog, run place_approach, "
            "release_part, lift, or run move_home."
        ).classes("text-sm text-red-700")
        ui.label(
            "The action uses your inspected physical-recovery confirmation and the exact "
            "terminal trial record. It does not require live RTDE feedback or a reconstructed "
            "RobotAgent. It clears only recovery_required; the insertion remains "
            "unsuccessful and unqualified."
        ).classes("text-xs text-slate-600")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=move_insert_recovery_confirm.close).props("flat")
            ui.button(
                "Confirm Physical Recovery",
                on_click=_confirmed_move_insert_recovery,
                icon="verified_user",
            ).props("color=red")

    with ui.dialog() as move_insert_completion_confirm, ui.card().classes(
        "gap-3 max-w-xl"
    ):
        ui.label("Confirm completed move_insert?").classes("font-semibold")
        ui.label(
            "Confirm only after visually checking that the selected part is correctly seated. "
            "The UR5e will release the part irreversibly and lift exactly once; it will not "
            "rerun move_insert. One successful release and lift confirms the selected exact "
            "part for the identical protected identities."
        ).classes("text-sm text-red-700")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=move_insert_completion_confirm.close).props("flat")
            ui.button(
                "Confirm Completion and Release",
                on_click=_confirmed_move_insert_completion,
                icon="lock_open",
            ).props("color=red")

    def _render_move_insert_trial() -> None:
        selected = _current_function() == "place_insert"
        move_insert_trial_container.set_visibility(
            _current_function() == "place_insert"
        )
        if not selected:
            return
        status = (
            dict(move_insert_trial.get("status") or {})
            if _move_insert_trial_is_current()
            else {}
        )
        state = _move_insert_trial_state() if status else "not_confirmed"
        state_labels = {
            "not_confirmed": "Not confirmed",
            "ready_to_test": "Ready to test",
            "testing": "Testing",
            "awaiting_visual_confirmation": "Awaiting visual confirmation",
            "confirmation_progress": "Confirmation progress",
            "confirmed": "Confirmed",
            "failure_recorded": "Not confirmed",
        }
        state_classes = {
            "not_confirmed": "text-xs font-semibold text-amber-700",
            "ready_to_test": "text-xs font-semibold text-blue-700",
            "testing": "text-xs font-semibold text-blue-700",
            "awaiting_visual_confirmation": "text-xs font-semibold text-amber-700",
            "confirmation_progress": "text-xs font-semibold text-blue-700",
            "confirmed": "text-xs font-semibold text-green-700",
            "failure_recorded": "text-xs font-semibold text-red-700",
        }
        move_insert_trial_selection_label.set_text(
            f"Selected: robot {_current_robot()}; destination "
            f"{_current_destination_location()}; part {_current_part_name()}."
        )
        if state == "confirmation_progress":
            confirmed_trial_count = int(
                status.get("confirmed_trial_count", 0) or 0
            )
            required_confirmed_trials = int(
                status.get("required_confirmed_trials", 1) or 1
            )
            move_insert_trial_state_label.set_text(
                f"Confirmed {confirmed_trial_count} of "
                f"{required_confirmed_trials}"
            )
        else:
            move_insert_trial_state_label.set_text(state_labels[state])
        move_insert_trial_state_label.classes(replace=state_classes[state])
        progress_message = (
            "Stopping supervised move_insert — waiting for force/search motion "
            f"to settle. {_current_part_name()} remains clamped."
            if move_insert_trial.get("stop_requested")
            else (
                _move_insert_trial_progress_message(status)
                if state == "testing"
                else str(status.get("message") or "")
            )
        )
        move_insert_trial_message.set_text(
            "Checking supervised move_insert readiness without motion..."
            if move_insert_trial.get("loading")
            else (
                progress_message
                or "Complete pick_grasp and place_approach before testing move_insert."
            )
        )
        missing_handoff = bool(
            _physical_function_execution_selected()
            and _current_robot() == "ur5e"
            and _current_destination_location() == "assembly_board-v1"
            and _current_part_name() in _MOVE_INSERT_SUPPORTED_PARTS
            and not bridge.digital_twin_function_held_part("ur5e")
        )
        move_insert_missing_handoff.set_text(
            "Function Execution has no retained "
            f"{_current_part_name()} handoff. Select place_approach, then Confirm Run "
            "place_approach once. That standalone run assumes "
            f"{_current_part_name()} is already physically clamped. Then return to "
            "place_insert; Supervised Test move_insert will recheck readiness."
        )
        move_insert_missing_handoff.set_visibility(missing_handoff)
        failure_id = str(status.get("failure_id") or "")
        bundle_path = str(
            status.get("download_path")
            or status.get("diagnostic_bundle_path")
            or ""
        )
        diagnostic_bits = []
        if failure_id:
            diagnostic_bits.append(f"Failure ID: {failure_id}.")
        if bundle_path:
            diagnostic_bits.append(f"Diagnostic bundle: {bundle_path}")
        expected_start_diagnostic = _move_insert_expected_start_diagnostic(status)
        if expected_start_diagnostic:
            diagnostic_bits.append(expected_start_diagnostic)
        insertion_depth_diagnostic = str(
            status.get("insertion_depth_diagnostic") or ""
        ).strip()
        if insertion_depth_diagnostic:
            diagnostic_bits.append(insertion_depth_diagnostic)
        hard_limit_reason = str(status.get("hard_limit_reason") or "").strip()
        if hard_limit_reason:
            diagnostic_bits.append(f"Hard limit: {hard_limit_reason}")
        soft_overload_reason = str(
            status.get("last_soft_overload_reason") or ""
        ).strip()
        if soft_overload_reason:
            diagnostic_bits.append(f"Last soft overload: {soft_overload_reason}")
        if status.get("hardware_stack_repair_required"):
            diagnostic_bits.append(
                "Normal Repair Hardware Stack is required before Cartesian motion."
            )
        move_insert_trial_diagnostic.set_text(" ".join(diagnostic_bits))
        move_insert_trial_diagnostic.set_visibility(bool(diagnostic_bits))
        download_move_insert_diagnostic_button.set_visibility(bool(bundle_path))
        download_move_insert_diagnostic_button.set_enabled(bool(bundle_path))

        local_active = bool(move_insert_trial.get("active"))
        move_insert_active = bool(local_active or status.get("active"))
        completion_motion_active = bool(status.get("completion_motion_active"))
        active = bool(move_insert_active or completion_motion_active)
        controls_idle = bool(
            not execution.get("busy")
            and not execution.get("checking")
            and not execution.get("preparing")
            and not move_insert_trial.get("loading")
        )
        supervised_move_insert_button.set_enabled(
            bool(
                controls_idle
                and not active
                and state != "confirmed"
                and state == "ready_to_test"
                and status.get("ready")
            )
        )
        stop_move_insert_button.set_visibility(True)
        stop_move_insert_button.set_enabled(
            bool(move_insert_active and not move_insert_trial.get("stop_requested"))
        )
        recovery_required = bool(status.get("recovery_required"))
        confirm_move_insert_recovery_button.set_visibility(recovery_required)
        confirm_move_insert_recovery_button.set_enabled(
            bool(
                recovery_required
                and controls_idle
                and not active
                and move_insert_trial.get("trial_id")
            )
        )
        confirm_move_insert_completion_button.set_enabled(
            bool(
                controls_idle
                and state == "awaiting_visual_confirmation"
                and status.get("completion_eligible")
                and move_insert_trial.get("trial_id")
            )
        )

    with assembly_container:  # noqa: SIM117 - preserve the Assembly card's UI slot.
        with ui.card().classes("w-full p-3"):
            ui.label("Assembly").classes("text-sm font-semibold")
            assembly_target_robot_status = ui.label("").classes("text-xs text-slate-600")
            ui.label(
                "Run Assembly is a manual commissioning action that executes these five "
                "functions in this exact order:"
            ).classes("text-xs text-slate-500")
            ui.label(
                "pick_approach → pick_grasp → place_approach → place_insert → move_home"
            ).classes("text-xs font-semibold text-blue-700")
            with ui.row().classes("items-center gap-2 w-full flex-wrap"):
                assembly_origin_select = (
                    ui.select([], label="origin_resource_location")
                    .props("dense")
                    .classes("w-60")
                )
                assembly_destination_select = (
                    ui.select([], label="destination_location")
                    .props("dense")
                    .classes("w-60")
                )
                assembly_part_select = (
                    ui.select([], label="part_name").props("dense").classes("w-36")
                )
            assembly_status = ui.label(
                "Assembly readiness is checked automatically without motion."
            ).classes("text-xs text-slate-500")
            assembly_progress_status = ui.label(
                "No Assembly is active. Completed functions: none."
            ).classes("text-xs text-slate-500")
            ui.label(
                "Assembly is blocked while the CAIS system is running, starting, or stopping. "
                "The Hardware Stack may remain running."
            ).classes("text-xs text-amber-700")
            ui.label(
                "The robot must begin in idle with an empty gripper. Missing optional robot "
                "corrections are allowed. Buffered or saved-unconfirmed robot corrections "
                "block Assembly until you use Save/Replace Pose or Clear Position."
            ).classes("text-xs text-slate-500")
            ui.label("Assembly Robot Corrections").classes("text-xs font-semibold")
            ui.label(
                "Shows pick_approach.descend, place_approach.move_above_destination, and "
                "place_approach.descend as active, missing, buffered, or unconfirmed."
            ).classes("text-xs text-slate-500")
            assembly_pick_approach_correction_status = ui.label("").classes(
                "text-xs text-slate-500"
            )
            assembly_place_approach_correction_status = ui.label("").classes(
                "text-xs text-slate-500"
            )

            async def _check_assembly_readiness() -> None:
                client = _current_client()
                if execution.get("busy") or execution.get("checking"):
                    _notify(
                        "A robot function check or execution is already active.",
                        type="warning",
                        client=client,
                    )
                    return
                _sync_assembly_options()
                robot = _current_robot()
                values = _assembly_kwargs()
                selection_signature = _assembly_selection_signature()
                selection_revision = int(execution["assembly_selection_revision"])
                execution.update({"checking": True, "active_function": "Assembly"})
                pending_assembly.clear()
                assembly_run_button.props("loading")
                _render_execution()
                _render_steps()
                assembly_status.set_text("Checking Assembly readiness without motion...")
                assembly_status.classes(replace="text-xs text-amber-700")
                try:
                    result = await bridge.digital_twin_assembly_readiness(
                        target,
                        robot,
                        origin_resource_location=values["origin_resource_location"],
                        destination_location=values["destination_location"],
                        part_name=values["part_name"],
                    )
                    if not _client_alive(assembly_status):
                        return
                    if (
                        selection_revision
                        != int(execution["assembly_selection_revision"])
                        or selection_signature != _assembly_selection_signature()
                    ):
                        _notify(
                            "Assembly selection changed during readiness; the old result was "
                            "discarded.",
                            type="warning",
                            client=client,
                        )
                        return
                    message = str(result.get("message") or "")
                    assembly_status.set_text(message)
                    assembly_status.classes(
                        replace=(
                            "text-xs text-green-700"
                            if result.get("success")
                            else "text-xs text-red-700"
                        )
                    )
                    if not result.get("success"):
                        _notify(message, type="warning", timeout=7000, client=client)
                        return
                    pending_assembly.update({"robot": robot, **values})
                    assembly_confirm_title.set_text(
                        f"Run Assembly on the physical {robot}?"
                    )
                    assembly_confirm_move_insert_status.set_text(
                        "Assembly readiness verified the protected move_insert recipe and "
                        f"qualification for exact part {values['part_name']}. No operator "
                        "tuning values are required."
                    )
                    assembly_confirm.open()
                except Exception as exc:
                    log.exception("Assembly readiness check failed")
                    message = f"Assembly readiness check failed: {exc}"
                    if _client_alive(assembly_status):
                        assembly_status.set_text(message)
                        assembly_status.classes(replace="text-xs text-red-700")
                    _notify(message, type="negative", timeout=7000, client=client)
                finally:
                    execution.update({"checking": False, "active_function": ""})
                    if _client_alive(assembly_run_button):
                        assembly_run_button.props(remove="loading")
                        _render_execution()
                        _render_steps()

            with ui.dialog() as assembly_confirm, ui.card().classes("gap-3 max-w-xl"):
                assembly_confirm_title = ui.label("").classes("font-semibold")
                ui.label("The Assembly runs all five functions in this exact order:").classes(
                    "text-sm"
                )
                ui.label(
                    "pick_approach → pick_grasp → place_approach → place_insert → move_home"
                ).classes("text-sm font-semibold")
                ui.label(
                    "place_insert.move_insert keeps the part gripped while it searches and "
                    "inserts. place_insert releases the part irreversibly "
                    "only after move_insert succeeds. Assembly stops immediately if any "
                    "function fails "
                    "and never retries or continues automatically."
                ).classes("text-xs text-red-700")
                assembly_confirm_move_insert_status = ui.label("").classes(
                    "text-xs text-slate-600 break-all"
                )
                ui.label(
                    "This commands physical robot motion. Clear the workcell and switch the "
                    "pendant to Remote Control before continuing."
                ).classes("text-xs text-red-700")
                with ui.row().classes("justify-end gap-2 w-full"):
                    ui.button("Cancel", on_click=assembly_confirm.close).props("flat")

                    async def _confirmed_assembly() -> None:
                        client = _current_client()
                        assembly_confirm.close()
                        if (
                            execution.get("busy")
                            or execution.get("checking")
                            or not pending_assembly
                        ):
                            _notify(
                                "Assembly readiness must complete before confirmation.",
                                type="warning",
                                client=client,
                            )
                            return
                        values = dict(pending_assembly)
                        robot = values.pop("robot")
                        execution.update({"busy": True, "active_function": "Assembly"})
                        assembly_run_button.props("loading")
                        _render_execution()
                        _render_steps()
                        assembly_status.set_text(
                            "Dispatching Assembly after fresh physical readiness checks."
                        )
                        assembly_status.classes(replace="text-xs text-amber-700")
                        assembly_progress_status.set_text(
                            "Assembly is starting. Completed functions: none."
                        )
                        assembly_progress_status.classes(replace="text-xs text-amber-700")
                        assembly_task: asyncio.Task | None = None
                        try:
                            assembly_task = asyncio.create_task(
                                bridge.digital_twin_execute_assembly(
                                    target,
                                    robot,
                                    origin_resource_location=values[
                                        "origin_resource_location"
                                    ],
                                    destination_location=values[
                                        "destination_location"
                                    ],
                                    part_name=values["part_name"],
                                    confirmed=True,
                                )
                            )
                            execution["assembly_task"] = assembly_task
                            result = await assembly_task
                            message = str(
                                result.get("message") or "Assembly completed."
                            )
                            completed_functions = [
                                str(function_name)
                                for function_name in result.get("completed_functions", []) or []
                            ]
                            failed_function = str(
                                result.get("failed_function") or ""
                            ).strip()
                            progress_text = (
                                "Completed functions: "
                                + (", ".join(completed_functions) or "none")
                                + "."
                            )
                            if failed_function:
                                progress_text += f" Failed function: {failed_function}."
                            if _client_alive(assembly_status):
                                assembly_status.set_text(message)
                                assembly_status.classes(
                                    replace=(
                                        "text-xs text-green-700"
                                        if result.get("success")
                                        else "text-xs text-red-700"
                                    )
                                )
                                assembly_progress_status.set_text(progress_text)
                                assembly_progress_status.classes(
                                    replace=(
                                        "text-xs text-green-700"
                                        if result.get("success")
                                        else "text-xs text-red-700"
                                    )
                                )
                            _notify(
                                message,
                                type=(
                                    "positive" if result.get("success") else "negative"
                                ),
                                timeout=7000,
                                client=client,
                            )
                        except Exception as exc:
                            log.exception("Assembly UI execution failed")
                            message = f"Assembly failed: {exc}"
                            if _client_alive(assembly_status):
                                assembly_status.set_text(message)
                                assembly_status.classes(replace="text-xs text-red-700")
                                assembly_progress_status.set_text(
                                    "Assembly stopped. See the failure above; no later function "
                                    "was requested by the UI."
                                )
                                assembly_progress_status.classes(
                                    replace="text-xs text-red-700"
                                )
                            _notify(
                                message,
                                type="negative",
                                timeout=7000,
                                client=client,
                            )
                        finally:
                            if execution.get("assembly_task") is assembly_task:
                                execution["assembly_task"] = None
                            pending_assembly.clear()
                            execution.update({"busy": False, "active_function": ""})
                            if _client_alive(assembly_run_button):
                                assembly_run_button.props(remove="loading")
                                _render_execution()
                                _render_steps()

                    ui.button(
                        "Confirm Run Assembly",
                        on_click=_confirmed_assembly,
                        icon="play_arrow",
                    ).props("color=red")

            assembly_run_button = (
                ui.button(
                    "Run Assembly",
                    on_click=_check_assembly_readiness,
                    icon="precision_manufacturing",
                )
                .props("outline dense")
                .classes("text-red-600")
            )

    def _render_execution() -> None:  # noqa: C901 - one render owns every motion gate.
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
            assembly_origin_select,
            assembly_destination_select,
            assembly_part_select,
        ):
            selector.set_enabled(bool(controls_enabled))
        assembly_target_robot_status.set_text(
            f"Assembly target: {target} | robot: {_current_robot()}"
        )
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
        hardware_target, hardware_statuses = _robot_function_hardware_snapshot()
        if hardware_target:
            stack = {
                "xarm only": "xarm6",
                "ur5e only": "ur5e",
                "dual robots": "dual robots",
            }[hardware_target]
            stack_status = dict(hardware_statuses.get(stack) or {})
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
        if bridge.teleop_cartesian_smooth_active(_current_robot()):
            blocker = "Release Cartesian Smooth Hold before Function Execution."
        demonstration_status = (
            dict(insertion_demonstration.get("status") or {})
            if insertion_demonstration.get("selection")
            == _insertion_demonstration_selection()
            else {}
        )
        demonstration_blocks_functions = bool(
            demonstration_status.get("active")
            or demonstration_status.get("recovery_required")
        )
        if demonstration_blocks_functions:
            blocker = str(
                demonstration_status.get("message")
                or "Insertion Demonstration blocks Robot Functions until it is stopped."
            )
        board_run_blocked = False
        place_approach_recording_blocked = False
        if (
            function_name == "place_approach"
            and _current_destination_location() == "assembly_board-v1"
        ):
            invalid_recordings = [
                str(step.get("invalid_reason") or "").strip()
                for step in bridge.digital_twin_list_function_file_steps(
                    target,
                    _current_robot(),
                    function_name,
                    _current_recording_name(),
                    part_name=_current_part_name(),
                )
                if str(step.get("invalid_reason") or "").strip()
            ]
            place_approach_recording_blocked = bool(invalid_recordings)
            if invalid_recordings:
                blocker = f"{blocker} {' '.join(invalid_recordings)}".strip()
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
        place_insert_run_blocked = False
        if (
            (
                str(getattr(bridge, "execution_mode", "") or "").strip().lower()
                == "physical"
                or str(getattr(bridge, "robot_env", "") or "").strip().lower()
                == "real"
                or bool(hardware_target)
            )
            and function_name == "place_insert"
            and _current_destination_location() == "assembly_board-v1"
        ):
            place_insert_run_blocked = True
            held_part = str(
                bridge.digital_twin_function_held_part(_current_robot()) or ""
            ).strip()
            if _current_robot() == "xarm6":
                place_insert_blocker = (
                    "Physical xarm6 move_insert remains blocked in this version. The "
                    "installed UFactory six-axis force/torque sensor still requires "
                    "verified CAIS wrench feedback, zeroing, force control, cancellation, "
                    "and a serialized xarm6 insertion action."
                )
            elif not held_part:
                if _current_part_name() in _MOVE_INSERT_SUPPORTED_PARTS:
                    place_insert_blocker = (
                        "Standalone Run place_insert is blocked because Function Execution "
                        "has no retained held_part context. Select place_approach and Confirm "
                        "Run place_approach once; that standalone run assumes exact part "
                        f"{_current_part_name()} is already "
                        "physically clamped. Then test move_insert."
                    )
                else:
                    place_insert_blocker = (
                        f"Standalone Run place_insert for exact part {_current_part_name()} "
                        "requires its retained pick_grasp and place_approach context."
                    )
            elif not _move_insert_trial_is_qualified():
                place_insert_blocker = (
                    f"Run place_insert is blocked until Supervised Test move_insert for exact "
                    f"part {_current_part_name()} finishes and you use Confirm Completion."
                )
            else:
                place_insert_run_blocked = False
                place_insert_blocker = ""
            blocker = f"{blocker} {place_insert_blocker}".strip()
        execution_blocker.set_text(blocker)
        execution_blocker.set_visibility(bool(blocker))
        enabled = bool(
            _current_robot() in {"xarm6", "ur5e"}
            and function_name
            and has_location
            and (not needs_part or _current_part_name())
            and controls_enabled
            and not board_run_blocked
            and not place_approach_recording_blocked
            and not place_insert_run_blocked
            and not demonstration_blocks_functions
        )
        run_button.set_enabled(enabled)
        assembly_run_button.set_enabled(
            bool(
                _current_robot() in {"xarm6", "ur5e"}
                and _current_assembly_origin_resource_location()
                and _current_assembly_destination_location()
                and _current_assembly_part_name()
                and controls_enabled
                and not demonstration_blocks_functions
            )
        )
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
        _render_insertion_demonstration()
        _render_move_insert_trial()

    async def _capture_position(step_name: str, primitive: str) -> None:
        client = _current_client()
        if execution.get("busy") or execution.get("checking"):
            _notify("A robot function check or execution is already active.", type="warning")
            return
        robot = _current_robot()
        function_name = _current_function()
        recording_name = _current_recording_name()
        part_name = _current_part_name()
        operator_confirmed_held_part = _operator_confirmed_held_part()
        operator_handoff_origin_resource_location = (
            _operator_held_part_origin_resource_location()
            if operator_confirmed_held_part
            else ""
        )
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
                operator_confirmed_held_part=operator_confirmed_held_part,
                operator_handoff_origin_resource_location=(
                    operator_handoff_origin_resource_location
                ),
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
        _refresh_assembly_correction_status()
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
                and saved_steps[str(step.get("step_name") or "")].get("confirmed")
            )
            recording_summary.set_text(
                f"{saved_count}/{len(recordable_steps)} optional robot corrections confirmed"
            )
        info = bridge.digital_twin_function_info(
            target,
            _current_robot(),
            function_name,
            name,
            _current_part_name(),
        )
        recording_info.set_text(
            (
                f"Saving as: {info.get('display_path')} — one {function_name} file for "
                f"{_current_robot()} across all part_name and location inputs"
            )
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
                        elif saved and saved.get("confirmed"):
                            ui.label("Pose saved").classes("text-xs text-green-700")
                        elif saved and saved.get("invalid_reason"):
                            ui.label("Recapture required").classes("text-xs text-red-700")
                        elif saved:
                            ui.label("Unconfirmed correction; ignored").classes(
                                "text-xs text-amber-700"
                            )
                        elif not position_required:
                            ui.label("Computed live; override optional").classes(
                                "text-xs text-slate-500"
                            )
                        else:
                            ui.label("Position required").classes("text-xs text-red-700")
                    pose = dict(position.get("pose") or {}) if position else {}
                    invalid_reason = str(
                        (saved or {}).get("invalid_reason") or ""
                    ).strip()
                    if invalid_reason:
                        ui.label(invalid_reason).classes("text-xs text-red-700")
                    if pose:
                        ui.label(
                            f"world → {pose.get('child_frame_id') or 'ee_link'}: "
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
                            ui.label(
                                "Preview, Test Position, and Function Run resolve current "
                                "geometry first, then apply this robot correction. Raw waypoint "
                                "Replay is disabled."
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
                        if controls_blocked or not saved or invalid_reason:
                            test_button.disable()

    def _reset_execution_status() -> None:
        message = "Run readiness is checked automatically without motion."
        classes = "text-xs text-slate-500"
        execution_status.set_text(message)
        execution_status.classes(replace=classes)

    def _invalidate_move_insert_trial() -> None:
        readiness_task = move_insert_trial.get("readiness_task")
        if isinstance(readiness_task, asyncio.Task) and not readiness_task.done():
            readiness_task.cancel()
        move_insert_trial.update(
            {
                "loading": False,
                "active": False,
                "stop_requested": False,
                "selection": (),
                "status": {},
                "trial_id": "",
                "readiness_revision": int(
                    move_insert_trial.get("readiness_revision") or 0
                )
                + 1,
                "readiness_task": None,
                "background_readiness_key": (),
            }
        )
        pending_move_insert_trial.clear()
        pending_move_insert_recovery.clear()
        move_insert_trial_confirm.close()
        move_insert_recovery_confirm.close()
        move_insert_completion_confirm.close()
        _render_move_insert_trial()

    def _schedule_selection_readiness_refresh() -> None:
        selection = _selection_signature()
        current_task = selection_update.get("readiness_task")
        if (
            isinstance(current_task, asyncio.Task)
            and not current_task.done()
            and selection_update.get("readiness_selection") == selection
        ):
            return
        if isinstance(current_task, asyncio.Task) and not current_task.done():
            current_task.cancel()

        selection_revision = int(execution["selection_revision"])

        async def _refresh_current_selection() -> None:
            await asyncio.sleep(0.05)
            if (
                selection_revision != int(execution["selection_revision"])
                or selection != _selection_signature()
            ):
                return
            await asyncio.gather(
                _refresh_assembly_board_v1_readiness(),
                _load_insertion_demonstration_readiness(),
                _load_move_insert_trial_readiness(),
            )

        task = asyncio.create_task(_refresh_current_selection())
        selection_update.update(
            {
                "readiness_task": task,
                "readiness_selection": selection,
            }
        )

        def _clear_selection_readiness_task(completed: asyncio.Task) -> None:
            if selection_update.get("readiness_task") is completed:
                selection_update["readiness_task"] = None
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("Robot Function selection readiness refresh failed")

        task.add_done_callback(_clear_selection_readiness_task)

    def _selection_changed(_e=None) -> None:
        if selection_update.get("active"):
            return
        execution["selection_revision"] = int(execution["selection_revision"]) + 1
        execution["assembly_selection_revision"] = (
            int(execution["assembly_selection_revision"]) + 1
        )
        pending_execution.clear()
        pending_assembly.clear()
        run_confirm.close()
        assembly_confirm.close()
        _invalidate_move_insert_trial()
        assembly_board_v1_state.update(
            {
                "last_failure": "",
                "accept_message": "",
                "accept_success": None,
            }
        )
        _sync_location()
        _sync_part_name()
        _sync_assembly_options()
        _reset_execution_status()
        _reset_assembly_status()
        _render_execution()
        _render_steps()
        _schedule_selection_readiness_refresh()

    def _part_name_changed(_e=None) -> None:
        if selection_update.get("active"):
            return
        execution["selection_revision"] = int(execution["selection_revision"]) + 1
        pending_execution.clear()
        run_confirm.close()
        _invalidate_move_insert_trial()
        _reset_execution_status()
        _render_execution()
        _render_steps()
        _schedule_selection_readiness_refresh()

    def _location_changed(_e=None) -> None:
        if selection_update.get("active"):
            return
        execution["selection_revision"] = int(execution["selection_revision"]) + 1
        pending_execution.clear()
        run_confirm.close()
        _invalidate_move_insert_trial()
        assembly_board_v1_state.update(
            {
                "last_failure": "",
                "accept_message": "",
                "accept_success": None,
            }
        )
        _reset_execution_status()
        _render_execution()
        _render_steps()
        _schedule_selection_readiness_refresh()

    def _assembly_selection_changed(_e=None) -> None:
        if selection_update.get("active"):
            return
        execution["assembly_selection_revision"] = (
            int(execution["assembly_selection_revision"]) + 1
        )
        pending_assembly.clear()
        assembly_confirm.close()
        _reset_assembly_status()
        _refresh_assembly_correction_status()
        _render_execution()

    robot_select.on_value_change(_selection_changed)
    function_select.on_value_change(_selection_changed)
    origin_select.on_value_change(_location_changed)
    destination_select.on_value_change(_location_changed)
    part_select.on_value_change(_part_name_changed)
    assembly_origin_select.on_value_change(_assembly_selection_changed)
    assembly_destination_select.on_value_change(_assembly_selection_changed)
    assembly_part_select.on_value_change(_assembly_selection_changed)
    _sync_location()
    _sync_part_name()
    _sync_assembly_options()
    _reset_execution_status()
    _reset_assembly_status()
    _render_execution()
    _render_steps()
    _schedule_selection_readiness_refresh()

    def _refresh_insertion_demonstration_status() -> None:
        if (
            _current_function() != "place_insert"
            or insertion_demonstration.get("selection")
            != _insertion_demonstration_selection()
            or not dict(insertion_demonstration.get("status") or {}).get("active")
        ):
            return
        selection = _insertion_demonstration_selection()
        try:
            result = bridge.digital_twin_insertion_recording_status(
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                recording_id=str(
                    insertion_demonstration.get("recording_id") or ""
                ),
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            log.exception("failed to refresh insertion demonstration status")
            return
        if selection != _insertion_demonstration_selection():
            return
        _apply_insertion_demonstration_status(dict(result), selection)
        _render_insertion_demonstration()
        _render_execution()

    def _refresh_move_insert_trial_status() -> None:
        if (
            _current_function() != "place_insert"
            or not _move_insert_trial_is_current()
            or not (
                move_insert_trial.get("active")
                or _move_insert_trial_state()
                in {"testing", "awaiting_visual_confirmation"}
            )
        ):
            return
        selection = _move_insert_trial_selection()
        try:
            result = bridge.digital_twin_move_insert_trial_status(
                target,
                _current_robot(),
                destination_location=_current_destination_location(),
                part_name=_current_part_name(),
                trial_id=str(move_insert_trial.get("trial_id") or ""),
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            log.exception("failed to refresh move_insert trial status")
            return
        if selection != _move_insert_trial_selection():
            return
        was_qualified = _move_insert_trial_is_qualified()
        _apply_move_insert_trial_status(dict(result), selection)
        _render_move_insert_trial()
        if was_qualified != _move_insert_trial_is_qualified():
            _render_execution()

    def _refresh_move_insert_trial_readiness_after_recording() -> None:
        if _current_function() != "place_insert":
            return
        if (
            insertion_demonstration.get("selection")
            != _insertion_demonstration_selection()
        ):
            return
        demonstration_status = dict(insertion_demonstration.get("status") or {})
        if (
            str(demonstration_status.get("state") or "")
            != "recording_saved_return_to_pre_insertion"
        ):
            return
        trial_status = (
            dict(move_insert_trial.get("status") or {})
            if _move_insert_trial_is_current()
            else {}
        )
        trial_state = str(
            trial_status.get("state")
            or trial_status.get("qualification_state")
            or ""
        )
        if (
            move_insert_trial.get("loading")
            or move_insert_trial.get("active")
            or trial_status.get("active")
            or trial_status.get("review_required")
            or trial_status.get("qualified")
            or execution.get("busy")
            or execution.get("checking")
            or execution.get("preparing")
            or trial_status.get("ready")
            or trial_state == "ready_to_test"
        ):
            return
        readiness_task = move_insert_trial.get("readiness_task")
        if isinstance(readiness_task, asyncio.Task) and not readiness_task.done():
            return
        background_key = (
            *_move_insert_trial_selection(),
            str(demonstration_status.get("recording_id") or ""),
            str(
                demonstration_status.get("demonstration_sha256")
                or demonstration_status.get("trace_sha256")
                or ""
            ),
        )
        move_insert_trial["background_readiness_key"] = background_key
        task = asyncio.create_task(
            _load_move_insert_trial_readiness(background=True)
        )
        move_insert_trial["readiness_task"] = task

        def _clear_readiness_task(completed: asyncio.Task) -> None:
            if move_insert_trial.get("readiness_task") is completed:
                move_insert_trial["readiness_task"] = None
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("automatic supervised move_insert readiness failed")

        task.add_done_callback(_clear_readiness_task)

    def _refresh_function_execution_progress() -> None:
        if not execution.get("busy"):
            return
        progress = bridge.digital_twin_robot_function_execution_progress()
        if not progress.get("active"):
            return
        message = str(progress.get("message") or "").strip()
        if execution.get("active_function") == "Assembly":
            assembly_step_index = int(progress.get("assembly_step_index", 0) or 0)
            assembly_step_count = int(
                progress.get("assembly_step_count", len(assembly_functions))
                or len(assembly_functions)
            )
            completed_functions = [
                str(function_name)
                for function_name in progress.get("completed_functions", []) or []
            ]
            failed_function = str(progress.get("failed_function") or "").strip()
            progress_bits = []
            if assembly_step_index:
                progress_bits.append(
                    f"Assembly step {assembly_step_index}/{assembly_step_count}."
                )
            active_step = str(progress.get("active_step") or "").strip()
            insert_phase = str(progress.get("insert_phase") or "").strip()
            if active_step:
                progress_bits.append(f"Active internal step: {active_step}.")
            if active_step == "place_insert.move_insert" and insert_phase:
                progress_bits.append(f"Insertion phase: {insert_phase}.")
            progress_bits.append(
                "Completed functions: "
                + (", ".join(completed_functions) or "none")
                + "."
            )
            if failed_function:
                progress_bits.append(f"Failed function: {failed_function}.")
            if _client_alive(assembly_progress_status):
                assembly_progress_status.set_text(" ".join(progress_bits))
                assembly_progress_status.classes(replace="text-xs text-amber-700")
            if message and _client_alive(assembly_status):
                assembly_status.set_text(message)
                assembly_status.classes(replace="text-xs text-amber-700")
            return
        if message and _client_alive(execution_status):
            execution_status.set_text(message)
            execution_status.classes(replace="text-xs text-amber-700")

    ui.timer(0.2, _refresh_function_execution_progress)
    ui.timer(0.2, _refresh_insertion_demonstration_status)
    ui.timer(0.2, _refresh_move_insert_trial_status)
    ui.timer(1.0, _refresh_move_insert_trial_readiness_after_recording)
    ui.timer(2.0, _refresh_assembly_board_v1_readiness, immediate=True)
    ui.label(
        "Capture checks read-only readiness automatically. Run and Test Position are separate "
        "physical motion actions requiring the selected robot's remote-control mode and "
        "explicit confirmation."
    ).classes("text-xs text-amber-700 mt-2")

    def _cancel_body_tasks(*_args: object) -> None:
        readiness_task = selection_update.get("readiness_task")
        if isinstance(readiness_task, asyncio.Task) and not readiness_task.done():
            readiness_task.cancel()
        task = execution.get("assembly_task")
        if isinstance(task, asyncio.Task) and not task.done():
            # The bridge shields only the in-flight function, then aborts the remainder.
            task.cancel()
        trial_task = move_insert_trial.get("trial_task")
        if isinstance(trial_task, asyncio.Task) and not trial_task.done():
            # The bridge keeps its motion lock until the in-flight trial settles.
            trial_task.cancel()
        pending_assembly.clear()
        pending_move_insert_trial.clear()
        pending_move_insert_recovery.clear()

    def _cancel_active_assembly(*_args: object) -> None:
        _cancel_body_tasks()
        recording_status = dict(insertion_demonstration.get("status") or {})
        if bool(recording_status.get("active")):
            threading.Thread(
                target=bridge.digital_twin_cancel_insertion_recording,
                kwargs={
                    "target": str(recording_status.get("target") or target),
                    "robot": str(recording_status.get("robot") or "ur5e"),
                    "destination_location": str(
                        recording_status.get("destination_location")
                        or "assembly_board-v1"
                    ),
                    "part_name": str(recording_status.get("part_name") or ""),
                    "recording_id": str(
                        recording_status.get("recording_id") or ""
                    ),
                    "note": "Control client disconnected during recording.",
                },
                daemon=True,
            ).start()

    client = _current_client()
    if client is not None:
        client.on_disconnect(_cancel_active_assembly)
        client.on_delete(_cancel_active_assembly)
    return _cancel_body_tasks


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
        initial_robot = "ur5e"
        selected_robot_state = {"robot": initial_robot, "changing": False}
        with ui.row().classes("items-center gap-4 mb-4"):
            ui.label("Robot:").classes("font-semibold text-sm")
            robot_select = ui.toggle(["xarm6", "ur5e"], value=initial_robot).classes(
                "text-sm"
            )

        mode_state = {"mode": "cartesian"}  # cartesian | gripper | joint
        cartesian_jog_mode = {
            "mode": "smooth",
            "preparing": False,
            "updating_toggle": False,
        }  # step | smooth
        cartesian_command_state = {"pending": False}
        conflicting_motion_buttons = []
        smooth_hold = {
            "pressed": False,
            "active": False,
            "stopping": False,
            "stop_requested": False,
            "robot": "",
            "axis": "",
            "speed_mm_s": 0.0,
            "generation": 0,
            "task": None,
        }
        cartesian_jog_buttons = []
        joint_jog_buttons = []
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
        cartesian_readiness_refresh = {
            "busy": False,
            "revision": 0,
            "robot": "",
            "readiness": None,
            "target": None,
        }

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
                    _schedule_cartesian_readiness_refresh(invalidate=True)
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
            if (
                key == "cartesian"
                and requested == 0.0
                and robot_key == str(robot_select.value or "xarm6")
            ):
                _request_smooth_stop(
                    reason="Cartesian speed set to 0",
                    expected=True,
                )
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
            _refresh_cartesian_controls()

        def _motion_speed_slider_release_value(event) -> object:
            values = event.args
            if isinstance(values, (list, tuple)) and len(values) == 1:
                return values[0]
            return values

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
                        control._props["min"] = minimum
                        control._props["max"] = maximum
                        control._props["step"] = step
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
            if speed_mm_s == 0.0:
                smooth_hold["pressed"] = False
                if _client_alive(teleop_warning_label):
                    ui.notify(
                        f"{robot}: Cartesian speed is 0; no motion was commanded.",
                        type="warning",
                        position="bottom-right",
                        timeout=2500,
                    )
                return
            if robot == "xarm6":
                readiness = cartesian_readiness_refresh["readiness"]
                if not (
                    isinstance(readiness, dict)
                    and str(readiness.get("cartesian_mode") or "off") == "smooth"
                    and readiness.get("cartesian_mode_ready") is True
                ):
                    prepared = await _apply_cartesian_mode("Smooth Hold")
                    if not prepared:
                        smooth_hold["pressed"] = False
                        return
            if (
                not smooth_hold["pressed"]
                or int(smooth_hold["generation"]) != generation
                or cartesian_jog_mode["mode"] != "smooth"
                or str(robot_select.value or "") != robot
            ):
                return
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
            smooth_hold["stop_requested"] = False
            smooth_hold["generation"] = int(smooth_hold["generation"]) + 1
            generation = int(smooth_hold["generation"])
            robot = str(robot_select.value or "xarm6")
            label = cartesian_status["label"]
            if label is not None:
                label.set_text(
                    f"Starting World {axis.upper()}{'+' if direction > 0 else '-'} "
                    "Smooth Hold..."
                )
                label.classes(replace="text-xs text-blue-700 mb-2")
            smooth_hold["task"] = asyncio.create_task(
                _run_smooth_hold(robot, axis, direction, generation)
            )

        def _request_smooth_stop(
            _event=None,
            *,
            reason: str = "browser pointer or focus event",
            expected: bool = False,
        ) -> None:
            was_holding = bool(smooth_hold["pressed"] or smooth_hold["active"])
            already_requested = bool(smooth_hold["stop_requested"])
            smooth_hold["pressed"] = False
            smooth_hold["stop_requested"] = True
            if (
                was_holding
                and not already_requested
                and not expected
                and _client_alive(teleop_warning_label)
            ):
                ui.notify(
                    f"Smooth Hold stopped by {reason}.",
                    type="warning",
                    position="bottom-right",
                    timeout=5000,
                )
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
                readiness = await asyncio.to_thread(
                    bridge.teleop_cartesian_readiness,
                    robot,
                )
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
                _schedule_cartesian_readiness_refresh(invalidate=True)

        def _refresh_cartesian_controls() -> None:
            label = cartesian_jog_controls["readiness"]
            toggle = cartesian_jog_controls["toggle"]
            if label is None or toggle is None or not _client_alive(label):
                return
            if smooth_hold["pressed"] or smooth_hold["active"] or smooth_hold["stopping"]:
                # Disabling the pointer-capturing button can make the browser emit
                # pointercancel and silently end an otherwise healthy hold. The
                # guarded start handler already rejects every concurrent axis.
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
            readiness = cartesian_readiness_refresh["readiness"]
            target = cartesian_readiness_refresh["target"]
            if (
                str(cartesian_readiness_refresh["robot"] or "") != robot
                or not isinstance(readiness, dict)
            ):
                label.set_text("Cartesian frame validation: checking...")
                label.classes(replace="text-xs text-slate-500 mb-2")
                for button in cartesian_jog_buttons:
                    button.disable()
                for button in conflicting_motion_buttons:
                    button.enable()
                joint_speed_ready = bool(
                    float(_joint_speed_deg_s(robot) or 0.0) > 0.0
                )
                for button in joint_jog_buttons:
                    button.set_enabled(joint_speed_ready)
                toggle.enable()
                robot_select.enable()
                return
            environment = str(readiness.get("environment") or "")
            selected_mode = str(cartesian_jog_mode["mode"] or "step")
            if environment == "real":
                base_ready = bool(readiness.get("cartesian_jog_ready"))
                cartesian_speed_ready = bool(
                    float(_cartesian_speed_mm_s(robot) or 0.0) > 0.0
                )
                message = str(readiness.get("message") or "")
                reported_mode = str(readiness.get("cartesian_mode") or "off")
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
                elif not cartesian_speed_ready:
                    label.set_text(
                        "Cartesian jog disabled: set Cartesian speed above 0 mm/s."
                    )
                    label.classes(replace="text-xs text-amber-700 mb-2")
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
                elif base_ready and selected_mode == "smooth":
                    preparation = (
                        "Mode 5"
                        if robot == "xarm6"
                        else "UR5e Cartesian Smooth Hold"
                    )
                    label.set_text(
                        f"Smooth Hold selected. Press and hold a World arrow; the first "
                        f"press prepares {preparation}."
                    )
                    label.classes(replace="text-xs text-amber-700 mb-2")
                else:
                    label.set_text(f"Cartesian frame validation failed: {message}")
                    label.classes(replace="text-xs text-red-700 mb-2")
                smooth_available = True
            else:
                if not isinstance(target, dict):
                    target = {}
                base_ready = bool(target.get("ready"))
                cartesian_speed_ready = True
                smooth_available = False
                ready = bool(base_ready and selected_mode == "step")
                label.set_text("Simulation uses Step Cartesian jog with velocity scaling.")
                label.classes(replace="text-xs text-slate-500 mb-2")
            if not smooth_available and selected_mode == "smooth":
                cartesian_jog_mode["mode"] = "step"
                _set_cartesian_toggle_value("Step")
                _request_smooth_stop(reason="Cartesian mode change", expected=True)
                selected_mode = "step"
                ready = bool(base_ready)
            controls_enabled = bool(
                (ready or (base_ready and selected_mode in {"step", "smooth"}))
                and cartesian_speed_ready
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
            joint_speed_ready = bool(
                environment != "real"
                or float(_joint_speed_deg_s(robot) or 0.0) > 0.0
            )
            for button in joint_jog_buttons:
                button.set_enabled(joint_speed_ready)
            toggle.set_enabled(
                not cartesian_jog_mode["preparing"]
                and not cartesian_command_state["pending"]
            )
            robot_select.enable()

        async def _refresh_cartesian_readiness_async(
            robot: str,
            revision: int,
        ) -> None:
            try:
                def _load() -> tuple[dict, dict | None]:
                    readiness = dict(bridge.teleop_cartesian_readiness(robot))
                    target = None
                    if str(readiness.get("environment") or "") != "real":
                        target = dict(bridge.teleop_target(robot, "cartesian"))
                    return readiness, target

                readiness, target = await asyncio.to_thread(_load)
            except Exception as exc:
                readiness = {
                    "environment": "",
                    "cartesian_jog_ready": False,
                    "message": f"readiness check failed: {type(exc).__name__}: {exc}",
                }
                target = None
            finally:
                cartesian_readiness_refresh["busy"] = False

            if not _client_alive(cartesian_jog_controls["readiness"]):
                return
            if (
                int(cartesian_readiness_refresh["revision"]) != revision
                or str(robot_select.value or "xarm6") != robot
            ):
                _schedule_cartesian_readiness_refresh()
                return
            cartesian_readiness_refresh["robot"] = robot
            cartesian_readiness_refresh["readiness"] = readiness
            cartesian_readiness_refresh["target"] = target
            _refresh_cartesian_controls()

        def _schedule_cartesian_readiness_refresh(
            *,
            invalidate: bool = False,
        ) -> None:
            if invalidate:
                cartesian_readiness_refresh["revision"] = (
                    int(cartesian_readiness_refresh["revision"]) + 1
                )
                cartesian_readiness_refresh["robot"] = ""
                cartesian_readiness_refresh["readiness"] = None
                cartesian_readiness_refresh["target"] = None
                _refresh_cartesian_controls()
            if (
                cartesian_readiness_refresh["busy"]
                or smooth_hold["pressed"]
                or smooth_hold["active"]
                or smooth_hold["stopping"]
                or cartesian_jog_mode["preparing"]
                or cartesian_command_state["pending"]
            ):
                return
            robot = str(robot_select.value or "xarm6")
            revision = int(cartesian_readiness_refresh["revision"])
            cartesian_readiness_refresh["busy"] = True
            asyncio.create_task(
                _refresh_cartesian_readiness_async(robot, revision)
            )

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
                    _request_smooth_stop(reason="Teleop Profile change", expected=True)
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
            ui.label(
                "Set a jog speed to 0 to disable that jog type. Existing robot-specific "
                "maximums remain unchanged."
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
                        cartesian_speed_slider.LOOPBACK = False
                        cartesian_speed_slider._props["loopback"] = False
                        cartesian_speed_slider.on(
                            "change",
                            lambda event: _set_motion_speed(
                                str(robot_select.value or "xarm6"),
                                "cartesian",
                                _motion_speed_slider_release_value(event),
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
                        joint_speed_slider.LOOPBACK = False
                        joint_speed_slider._props["loopback"] = False
                        joint_speed_slider.on(
                            "change",
                            lambda event: _set_motion_speed(
                                str(robot_select.value or "xarm6"),
                                "joint",
                                _motion_speed_slider_release_value(event),
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
                ui.label(
                    "Smooth Hold is the Hardware Stack default: press and keep holding an "
                    "arrow for continuous motion, then release to stop. Step moves one "
                    "finite Step (mm) distance per click."
                ).classes("text-xs text-slate-500 mb-2")

                def _set_cartesian_jog_mode(value: str) -> None:
                    if cartesian_jog_mode["updating_toggle"]:
                        return
                    asyncio.create_task(_apply_cartesian_mode(str(value or "Step")))

                cartesian_jog_controls["toggle"] = ui.toggle(
                    ["Step", "Smooth Hold"],
                    value="Smooth Hold",
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
                    readiness = cartesian_readiness_refresh["readiness"]
                    if not (
                        isinstance(readiness, dict)
                        and str(readiness.get("cartesian_mode") or "off") == "step"
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
                        _schedule_cartesian_readiness_refresh(invalidate=True)

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

                    joint_minus_button = ui.button(
                        "-", on_click=_joint_minus, icon="remove"
                    ).props("outline")
                    joint_plus_button = ui.button(
                        "+", on_click=_joint_plus, icon="add"
                    ).props("outline")
                    joint_jog_buttons.extend(
                        (joint_minus_button, joint_plus_button)
                    )
                    conflicting_motion_buttons.extend(
                        (joint_minus_button, joint_plus_button)
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
                cartesian_jog_mode["mode"] = "smooth"
                _set_cartesian_toggle_value("Smooth Hold")
                _configure_motion_speed_controls(selected_robot)
                _refresh_motion_speed_labels()
                _refresh_effective_labels()
                _schedule_cartesian_readiness_refresh(invalidate=True)

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
        _schedule_cartesian_readiness_refresh(invalidate=True)
        ui.timer(1.0, _refresh_teleop_status)
        ui.timer(0.5, _schedule_cartesian_readiness_refresh)
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
                    _request_smooth_stop(reason="keyboard release", expected=True)
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
                lambda _event=None, event_name=event_name: smooth_stop(
                    reason=event_name,
                    expected=event_name == "pointerup",
                ),
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
    if speed_mm_s is not None and float(speed_mm_s) == 0.0:
        msg = f"{robot} Cartesian speed is 0; no motion was commanded"
        ui.notify(msg, type="warning", position="bottom-right", timeout=2500)
        return False, msg
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
    if speed_deg_s is not None and float(speed_deg_s) == 0.0:
        msg = f"{robot} joint jog speed is 0; no motion was commanded"
        ui.notify(msg, type="warning", position="bottom-right", timeout=2500)
        return False, msg
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
