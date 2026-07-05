"""Resources page: simulation controls, robot phase pipeline, state cards, config editor."""

from __future__ import annotations

import json

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.status_badge import status_badge

_PHASES = ["idle", "at_pick", "picked", "positioned", "placed"]

# Friendly labels for ROS2 processes.
_ROS2_LABELS = {
    "gazebo_moveit": (
        "Gazebo + MoveIt",
        "Launches dual-robot Gazebo simulation with MoveIt motion planning and RViz",
    ),
    "perception": ("Perception", "Part detection via Gazebo ground-truth camera"),
    "teleop_xarm6": ("Teleop xArm6", "Keyboard teleoperation for xArm6"),
    "teleop_ur5e": ("Teleop UR5e", "Keyboard teleoperation for UR5e"),
}


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Resources").classes("text-2xl font-bold")

        # ── Simulation Controls ──────────────────────────────────────
        _simulation_controls(bridge)

        # ── Robot Status Cards ───────────────────────────────────────
        robot_container = ui.column().classes("w-full gap-6")

        def _refresh():
            robot_container.clear()
            states = bridge.get_robot_states()

            if not states:
                with robot_container:
                    ui.label("No robots available — start the system first").classes(
                        "text-slate-400 italic"
                    )
                return

            with robot_container:
                for name, state in states.items():
                    _robot_card(bridge, name, state)

        ui.timer(2.0, _refresh)


def _simulation_controls(bridge: SystemBridge) -> None:
    """ROS2/Gazebo process launch controls."""
    with ui.card().classes("w-full"):
        ui.label("Simulation Controls").classes("text-lg font-semibold mb-1")
        ui.label(
            "Launch and manage ROS2 processes. Gazebo + MoveIt must be running before starting the system in Simulation mode."
        ).classes("text-xs text-slate-500 mb-3")

        proc_container = ui.column().classes("w-full gap-3")

        def _refresh_procs():
            proc_container.clear()
            statuses = bridge.ros2_all_statuses()

            with proc_container:
                for name, status in statuses.items():
                    label, description = _ROS2_LABELS.get(name, (name, ""))
                    with ui.row().classes("items-center gap-4 w-full"):
                        # Status dot.
                        color = "green" if status == "running" else "grey"
                        ui.icon("circle", color=color).classes("text-xs")

                        # Label + description.
                        with ui.column().classes("gap-0 flex-1"):
                            ui.label(label).classes("font-semibold text-sm")
                            if description:
                                ui.label(description).classes("text-xs text-slate-400")

                        # Start / Stop buttons.
                        is_running = status == "running"

                        def _start(n=name):
                            err = bridge.ros2_start(n)
                            if err:
                                ui.notify(err, type="warning")
                            else:
                                ui.notify(f"Started {n}", type="positive")

                        def _stop(n=name):
                            bridge.ros2_stop(n)
                            ui.notify(f"Stopped {n}", type="info")

                        ui.button("Start", on_click=_start, icon="play_arrow").props(
                            "flat dense" + (" disable" if is_running else "")
                        ).classes("text-green-600")
                        ui.button("Stop", on_click=_stop, icon="stop").props(
                            "flat dense" + (" disable" if not is_running else "")
                        ).classes("text-red-600")

                # Utility buttons.
                ui.separator()
                with ui.row().classes("gap-4"):

                    def _stop_all():
                        bridge.ros2_stop_all()
                        ui.notify("All ROS2 processes stopped", type="info")

                    def _kill_gazebo():
                        bridge.ros2_kill_gazebo()
                        ui.notify("Killed orphan Gazebo processes", type="info")

                    ui.button("Stop All", on_click=_stop_all, icon="stop_circle").props(
                        "flat dense"
                    ).classes("text-red-600")
                    ui.button(
                        "Kill Orphan Gazebo", on_click=_kill_gazebo, icon="delete_sweep"
                    ).props("flat dense").classes("text-orange-600")

        ui.timer(3.0, _refresh_procs)


def _robot_card(bridge: SystemBridge, name: str, state: dict) -> None:
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-4"):
            ui.label(name).classes("text-lg font-bold")
            current = state.get("current_state", "idle")
            status_badge(current)

        # Phase pipeline visualization.
        with ui.row().classes("gap-1 mt-2"):
            current_phase = state.get("current_state", "idle")
            for phase in _PHASES:
                is_current = phase == current_phase
                color = "bg-blue-500 text-white" if is_current else "bg-slate-200 text-slate-600"
                ui.label(phase).classes(f"px-3 py-1 rounded text-xs font-mono {color}")

        # State details.
        with ui.row().classes("gap-8 mt-3 flex-wrap"):
            _detail("Execution Mode", state.get("execution_mode", "?"))
            _detail("Controller Ready", str(state.get("controller_ready", "?")))
            _detail("Held Part", state.get("held_part") or "None")
            _detail("Gripper", state.get("gripper_state", "?"))

        pos = state.get("position", {})
        if pos:
            with ui.row().classes("gap-4 mt-2"):
                for axis in ("x", "y", "z"):
                    val = pos.get(axis, 0)
                    ui.label(
                        f"{axis}: {val:.3f}" if isinstance(val, float) else f"{axis}: {val}"
                    ).classes("text-xs font-mono text-slate-500")

        # Config editor (expandable).
        with ui.expansion("Configuration", icon="settings").classes("w-full mt-2"):
            res_files = bridge.list_resource_files()
            matching = [f for f in res_files if name in f]
            if matching:
                config_path = matching[0]
                try:
                    config_data = bridge.load_config(config_path)
                except Exception:
                    config_data = {}

                editor = (
                    ui.textarea(
                        value=json.dumps(config_data, indent=2),
                        label=f"Config: {config_path.split('/')[-1]}",
                    )
                    .classes("w-full font-mono text-xs")
                    .props("rows=15")
                )

                def _save(p=config_path, e=editor):
                    try:
                        data = json.loads(e.value)
                        bridge.save_config(p, data)
                        ui.notify("Configuration saved", type="positive")
                    except json.JSONDecodeError as exc:
                        ui.notify(f"Invalid JSON: {exc}", type="negative")

                ui.button("Save Config", on_click=_save, icon="save").classes("mt-2")
            else:
                ui.label("No config file found").classes("text-slate-400")


def _detail(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-xs text-slate-500")
        ui.label(value).classes("text-sm font-semibold")
