"""Control page: Gazebo/hardware launch, interactive teleop with arrow buttons + keyboard."""

from __future__ import annotations

import asyncio
import math

from nicegui import ui
from nicegui.client import Client
from nicegui.events import KeyEventArguments

from cais_spade_llm.ui.bridge import SystemBridge


# Gazebo launch variants with friendly labels.
_GAZEBO_VARIANTS = {
    "gazebo_dual": ("Dual Robots (xArm6 + UR5e)", "Full dual-robot Gazebo + MoveIt + RViz"),
    "gazebo_xarm6": ("xArm6 Only", "Single xArm6 Gazebo + MoveIt + RViz"),
    "gazebo_ur5e": ("UR5e Only", "Single UR5e + RG2 Gazebo + MoveIt + RViz"),
}

_HARDWARE_STACKS = {
    "xarm6": ("xArm6 Hardware Stack", "Start xArm6 MoveIt realmove stack (includes embedded driver)"),
    "ur5e": ("UR5e Hardware Stack", "Auto sequence: start driver, wait for ready, then start MoveIt"),
}
_HARDWARE_PROC_NAMES = (
    "hardware_xarm6_driver",
    "hardware_xarm6_moveit",
    "hardware_ur5e_driver",
    "hardware_ur5e_moveit",
)

_SUPPORT_PROCS = {
    "perception": ("Perception", "Part detection via Gazebo ground-truth camera"),
}


def _client_alive(element) -> bool:
    try:
        client = getattr(element, "client", None)
        if client is None:
            return False
        return (client.id in Client.instances) and (not getattr(client, "_deleted", False))
    except RuntimeError:
        return False


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
            "Hardware stacks are combined per arm. UR5e starts Driver then MoveIt; xArm6 MoveIt realmove includes its driver."
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

        def _refresh():
            if not _client_alive(launch_container):
                return
            launch_container.clear()
            statuses = bridge.ros2_all_statuses()
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
                    def _stop_all():
                        bridge.ros2_stop_all()
                        ui.notify("Stopped all tracked processes", type="info")
                        _refresh()

                    def _cleanup():
                        bridge.ros2_cleanup_processes()
                        ui.notify("Cleanup complete: removed stale ROS2/MoveIt/driver processes", type="info")
                        _refresh()

                    ui.button("Stop All", on_click=_stop_all, icon="stop_circle").props("flat dense").classes("text-red-600")
                    ui.button("Cleanup", on_click=_cleanup, icon="cleaning_services").props("flat dense").classes("text-amber-700")

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

        def _start(n=name):
            if blocked_reason:
                ui.notify(blocked_reason, type="warning")
                return
            err = bridge.ros2_start(n)
            if err:
                ui.notify(err, type="warning")
            else:
                ui.notify(f"Started {label}", type="positive")

        def _stop(n=name):
            bridge.ros2_stop(n)
            ui.notify(f"Stopped {label}", type="info")

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
    driver_state = str(status.get("driver", "stopped"))
    moveit_state = str(status.get("moveit", "stopped"))
    color = "green" if overall == "running" else ("orange" if overall == "partial" else "grey")

    with ui.row().classes("items-center gap-4 w-full"):
        ui.icon("circle", color=color).classes("text-xs")

        with ui.column().classes("gap-0 flex-1"):
            ui.label(label).classes("font-semibold text-sm")
            ui.label(desc).classes("text-xs text-slate-400")
            ui.label(f"Driver: {driver_state} | MoveIt: {moveit_state}").classes("text-xs text-slate-500")

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
            status = bridge.teleop_connection_status()
            connected = bool(status.get("connected", False))
            env = str(status.get("environment", "gazebo")).strip().lower()

            teleop_backend_icon.props(f"color={'green' if connected else 'red'}")
            if connected:
                teleop_backend_label.set_text("Teleop backend: connected")
            else:
                teleop_backend_label.set_text(
                    "Teleop backend: disconnected (starts on first teleop command)"
                )

            if env == "real":
                teleop_env_icon.props("color=green")
                teleop_env_label.set_text("Teleop environment: hardware (real)")
            elif env == "gazebo":
                teleop_env_icon.props("color=blue")
                teleop_env_label.set_text("Teleop environment: simulation (gazebo)")
            else:
                teleop_env_icon.props("color=grey")
                teleop_env_label.set_text(f"Teleop environment: {env or 'unknown'}")

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

            # Avoid spinning up teleop backend when no environment is running.
            statuses = bridge.ros2_all_statuses()
            env_running = any(statuses.get(name) == "running" for name in _GAZEBO_VARIANTS) or any(
                statuses.get(name) == "running" for name in _HARDWARE_PROC_NAMES
            )
            if not env_running:
                _set_state_unavailable("no environment running")
                return

            teleop_status = bridge.teleop_connection_status()
            if not bool(teleop_status.get("connected", False)):
                _set_state_unavailable("teleop backend disconnected")
                return

            state_refresh["busy"] = True
            robot = str(robot_select.value or "xarm6")
            try:
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
