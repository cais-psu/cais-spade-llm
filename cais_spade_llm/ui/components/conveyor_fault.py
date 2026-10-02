"""Checkpoint-guarded controls for the selected simulation failure."""

from __future__ import annotations

import json

from nicegui import ui


def render_conveyor_fault(bridge, timer) -> None:
    """Read and control the simulation fault through SystemBridge only."""
    with ui.card().classes("w-full") as card:
        title = ui.label("Failure injection").classes("text-lg font-semibold")
        status = ui.label().classes("whitespace-pre-line")
        ui.label("Trigger requires the configured observed checkpoint. Stop preserves the failure; Reset Gazebo or Reset All clears it.").classes("text-sm")
        with ui.row():
            arm = ui.button("Arm failure")
            disarm = ui.button("Disarm failure")
            trigger = ui.button("Trigger failure", color="red")
        evidence = ui.code(language="json").classes("w-full")

    def refresh():
        fault = bridge.get_conveyor_fault()
        state = fault["status"]
        title.text = fault.get("scenario", "Failure injection")
        status.text = (
            f"{title.text}: execution stopped. Injection: {fault.get('evidence', {}).get('injection_status', 'pending')}. "
            f"Gazebo marker: {fault.get('visual', {}).get('status', 'pending')}."
            if state == "triggered" else f"Fault: {state}. Checkpoint: {fault.get('checkpoint', 'not configured')}."
        )
        status.text += (
            f"\nAffected resources: {', '.join(fault.get('affected_resources', [fault.get('resource_id', '')]))}. "
            f"Checkpoint: {'ready' if fault.get('ready') else 'not ready'}."
        )
        card.classes(
            add="border-2 border-red-600" if state == "triggered" else "",
            remove="border-2 border-red-600" if state != "triggered" else "",
        )
        active = (
            bridge.system_running
            and bridge.execution_mode == "simulation"
            and fault.get("run_active", False)
        )
        arm.set_enabled(active and state == "disarmed")
        disarm.set_enabled(active and state == "armed")
        trigger.set_enabled(active and state == "armed" and fault.get("ready", False))
        evidence.content = json.dumps(fault, indent=2)

    async def change(action):
        try:
            if action == "trigger":
                await bridge.trigger_conveyor_fault()
            else:
                await bridge.arm_conveyor_fault(action == "arm")
        except (ValueError, RuntimeError) as exc:
            ui.notify(str(exc), type="warning")
        refresh()

    arm.on_click(lambda: change("arm"))
    disarm.on_click(lambda: change("disarm"))
    trigger.on_click(lambda: change("trigger"))
    timer(1.0, refresh)
    refresh()
