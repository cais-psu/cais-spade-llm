"""NiceGUI page for the ICRA 2027 Spec2Skill case study."""

from __future__ import annotations

import asyncio

from nicegui import ui
from nicegui.elements.badge import Badge
from nicegui.elements.button import Button
from nicegui.elements.label import Label

from cais_spade_llm.spec2skill.adapters.dual_gazebo import (
    DualGazeboRuntime,
    DualGazeboStatus,
    read_dual_gazebo_status,
    start_dual_gazebo,
    stop_dual_gazebo,
)

_FLOW_STEPS = (
    "product requirement: assemble MCP",
    "PA retrieves manual/specification/CAD",
    "PA grounds target_feature, target pose, insertion axis, tolerances",
    "RA retrieves fresh resource state and resource-owned primitive catalog",
    "RA authors primitive_steps",
    "state checks + CCA + IK/collision/trajectory validation",
)
_PLACEHOLDER_REPLY = (
    "Placeholder only — the Spec2Skill PA/RA pipeline is not connected yet, "
    "so no plan or robot action was executed."
)


def _placeholder_reply(_message: str) -> str:
    """Return the fixed non-executing chat response."""
    return _PLACEHOLDER_REPLY


def _render_flow_step(number: int, text: str) -> None:
    """Render one static workflow step."""
    with (
        ui.card().classes("w-full border border-slate-200 shadow-sm"),
        ui.row().classes("items-center gap-4 w-full"),
    ):
        ui.badge(str(number)).props("color=indigo rounded")
        ui.label(text).classes("text-base text-slate-800")


def _status_color(state: str) -> str:
    if state == "running":
        return "green"
    if state == "stopped":
        return "grey"
    return "amber"


def _set_enabled(element: Button, enabled: bool) -> None:
    if enabled:
        element.props(remove="disable")
    else:
        element.props("disable")


def _apply_dual_gazebo_status(
    status: DualGazeboStatus,
    *,
    busy: bool,
    status_badge: Badge,
    status_message: Label,
    start_button: Button,
    stop_button: Button,
    refresh_button: Button,
) -> None:
    """Apply a fresh runtime snapshot to the launcher controls."""
    status_badge.set_text(status.state)
    status_badge.props(f"color={_status_color(status.state)}")
    if status.blocked_reason:
        status_message.set_text(status.blocked_reason)
        status_message.classes(replace="text-sm text-amber-700")
    elif status.state == "running":
        status_message.set_text("Dual Gazebo is running under existing UI ownership.")
        status_message.classes(replace="text-sm text-green-700")
    elif status.state == "stopped":
        status_message.set_text("Dual Gazebo is stopped and ready to launch.")
        status_message.classes(replace="text-sm text-slate-600")
    else:
        status_message.set_text(f"Dual Gazebo status: {status.state}")
        status_message.classes(replace="text-sm text-amber-700")

    _set_enabled(
        start_button,
        not busy and status.state == "stopped" and not status.blocked_reason,
    )
    _set_enabled(stop_button, not busy and status.state == "running")
    _set_enabled(refresh_button, not busy)


def _render_dual_gazebo(runtime: DualGazeboRuntime) -> None:
    """Render the isolated `gazebo_dual_spec2skill` launcher and controls."""
    with ui.card().classes("flex-1 min-w-80 border border-slate-200 shadow-sm"):
        with ui.row().classes("w-full items-center justify-between gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Dual Gazebo Environment").classes(
                    "text-lg font-semibold text-slate-900"
                )
                ui.label(
                    "xArm6 + UR5e · NIST CAD · Gazebo + MoveIt/RViz · No hardware"
                ).classes(
                    "text-xs text-slate-500"
                )
            status_badge = ui.badge("checking").props("color=grey outline")

        status_message = ui.label("Reading fresh runtime status...").classes(
            "text-sm text-slate-600"
        )
        action_state = {
            "busy": False,
            "refreshing": False,
            "status": DualGazeboStatus(state="checking"),
        }

        with ui.row().classes("items-center gap-2 flex-wrap"):
            start_button = ui.button("Start", icon="play_arrow").props("disable")
            stop_button = ui.button("Stop", icon="stop").props("outline disable")
            refresh_button = ui.button("Refresh", icon="refresh").props("flat")

        def _apply_status(status: DualGazeboStatus) -> None:
            action_state["status"] = status
            _apply_dual_gazebo_status(
                status,
                busy=bool(action_state["busy"]),
                status_badge=status_badge,
                status_message=status_message,
                start_button=start_button,
                stop_button=stop_button,
                refresh_button=refresh_button,
            )

        async def _refresh_status() -> None:
            if action_state["refreshing"]:
                return
            action_state["refreshing"] = True
            try:
                status = await asyncio.to_thread(read_dual_gazebo_status, runtime)
            except (OSError, RuntimeError, ValueError) as exc:
                status_badge.set_text("unavailable")
                status_badge.props("color=red")
                status_message.set_text(f"Unable to read dual Gazebo status: {exc}")
                status_message.classes(replace="text-sm text-red-700")
                _set_enabled(start_button, False)
                _set_enabled(stop_button, False)
            else:
                _apply_status(status)
            finally:
                action_state["refreshing"] = False

        async def _start() -> None:
            if action_state["busy"]:
                return
            action_state["busy"] = True
            _apply_status(action_state["status"])
            try:
                error = await asyncio.to_thread(start_dual_gazebo, runtime)
                if error:
                    ui.notify(error, type="warning", timeout=5000)
                else:
                    ui.notify("Started Dual Robots (xArm6 + UR5e)", type="positive")
            except (OSError, RuntimeError, ValueError) as exc:
                ui.notify(f"Dual Gazebo start failed: {exc}", type="negative", timeout=5000)
            finally:
                action_state["busy"] = False
                await _refresh_status()

        async def _stop() -> None:
            if action_state["busy"]:
                return
            action_state["busy"] = True
            _apply_status(action_state["status"])
            try:
                await asyncio.to_thread(stop_dual_gazebo, runtime)
                ui.notify("Stopped Dual Robots (xArm6 + UR5e)", type="info")
            except (OSError, RuntimeError, ValueError) as exc:
                ui.notify(f"Dual Gazebo stop failed: {exc}", type="negative", timeout=5000)
            finally:
                action_state["busy"] = False
                await _refresh_status()

        start_button.on_click(_start)
        stop_button.on_click(_stop)
        refresh_button.on_click(_refresh_status)
        ui.timer(0.1, _refresh_status, once=True)
        ui.timer(3.0, _refresh_status)


def _render_placeholder_chat() -> None:
    """Render a browser-local chat that cannot plan or execute requests."""
    with ui.card().classes("flex-1 min-w-80 border border-slate-200 shadow-sm"):
        ui.label("User Interaction").classes("text-lg font-semibold text-slate-900")
        ui.label(
            "Placeholder only. Messages are not sent to agents and clear on reload."
        ).classes("text-xs text-slate-500")

        messages = ui.column().classes(
            "w-full min-h-40 max-h-72 overflow-auto rounded bg-slate-50 p-3 gap-2"
        )
        with ui.row().classes("w-full items-end gap-2"):
            message_input = ui.input(
                placeholder="assembly the medium gear",
            ).classes("flex-1")
            send_button = ui.button("Send", icon="send")

        def _send() -> None:
            message = str(message_input.value or "").strip()
            if not message:
                return
            message_input.set_value("")
            with messages:
                ui.chat_message(text=message, name="You", sent=True)
                ui.chat_message(
                    text=_placeholder_reply(message),
                    name="Spec2Skill",
                    sent=False,
                )

        send_button.on_click(_send)
        message_input.on("keydown.enter", _send)


def render(runtime: DualGazeboRuntime) -> None:
    """Render the Spec2Skill Phase 0.1 operator shell and research workflow."""
    with ui.column().classes("w-full max-w-6xl mx-auto gap-6 p-6"):
        with ui.row().classes("w-full items-start justify-between gap-4"):
            with ui.column().classes("gap-1"):
                ui.label("Spec2Skill").classes("text-3xl font-bold text-slate-900")
                ui.label(
                    "A Multi-Agent Framework for Primitive Composition in Robotic Assembly"
                ).classes("text-lg text-slate-600")
            ui.badge("ICRA 2027 Case Study").props("color=indigo outline").classes(
                "text-sm px-3 py-2"
            )

        with ui.card().classes("w-full bg-indigo-50 border border-indigo-100 shadow-none"):
            ui.label("Phase 0.1: Operator Shell").classes(
                "text-base font-semibold text-indigo-900"
            )
            ui.label(
                "The environment launcher reuses existing UI runtime ownership. The chat "
                "is local and does not call agents or execute robot actions."
            ).classes("text-sm text-indigo-800")

        with ui.row().classes("w-full gap-4 items-stretch flex-wrap"):
            _render_dual_gazebo(runtime)
            _render_placeholder_chat()

        ui.label("Proposed Workflow").classes("text-xl font-semibold text-slate-900")

        with ui.column().classes("w-full gap-2"):
            for number, text in enumerate(_FLOW_STEPS, start=1):
                _render_flow_step(number, text)
                if number < len(_FLOW_STEPS):
                    ui.icon("south").classes("self-center text-slate-400")

        with ui.row().classes("w-full gap-4 items-stretch flex-wrap"):
            with ui.card().classes(
                "flex-1 min-w-72 bg-amber-50 border border-amber-200 shadow-none"
            ):
                ui.badge("rejected").props("color=amber outline")
                ui.label("concrete feedback → RA revision").classes(
                    "text-base font-semibold text-amber-900"
                )
            with ui.card().classes(
                "flex-1 min-w-72 bg-emerald-50 border border-emerald-200 shadow-none"
            ):
                ui.badge("accepted").props("color=green outline")
                ui.label("RobotAgent execution").classes(
                    "text-base font-semibold text-emerald-900"
                )

        ui.label(
            "ProductAgent, ResourceAgent, CCA, and RobotAgent remain shared runtime "
            "authorities outside the Spec2Skill package."
        ).classes("text-sm text-slate-500")
