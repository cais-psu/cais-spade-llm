"""Simulation performance status and explicit viewer and scene controls."""

from __future__ import annotations

import asyncio
import json
import time

from nicegui import ui


def render_simulation_controls(bridge, timer, starting) -> None:
    """Render measured performance and viewer controls through existing bridge methods."""
    busy = asyncio.Lock()
    with ui.column().classes('w-full gap-2'):
        status = ui.label('Simulation is stopped.').classes('text-sm whitespace-pre-line')
        with ui.row().classes('gap-3'):
            open_viewer = ui.button('Open RViz', icon='visibility')
            close_viewer = ui.button('Close RViz', icon='visibility_off')
            stop_scene = ui.button('Stop Simulation', icon='stop_circle')

    async def stop_simulation():
        if starting() or bridge.system_running:
            ui.notify('Stop System before stopping the simulation.', type='warning')
            return
        for name in ('recovery_rviz', 'gazebo_dual'):
            error = await asyncio.to_thread(bridge.ros2_stop, name)
            if error:
                ui.notify(error, type='warning')

    async def viewer(opened: bool):
        if opened and not await asyncio.to_thread(bridge.simulation_environment_running):
            ui.notify('Start Simulation before opening RViz.', type='warning')
            return
        error = await asyncio.to_thread(bridge.ros2_start if opened else bridge.ros2_stop,
                                        'recovery_rviz')
        if error:
            ui.notify(error, type='warning')

    async def refresh():
        if busy.locked():
            return
        async with busy:
            running = await asyncio.to_thread(bridge.simulation_environment_running)
            open_viewer.set_enabled(running)
            stop_scene.set_enabled(running and not starting() and not bridge.system_running)
            if not running:
                status.text = 'Simulation is stopped.'
                return
            ok, output = await asyncio.to_thread(
                bridge.ros2_exec, 'cat /tmp/cais_simulation_performance_${ROS_DOMAIN_ID:-0}.json', 3.0)
            try:
                live = json.loads(output) if ok else {}
                if time.time() - live.get('observed_at_unix', 0) > 5:
                    raise ValueError('stale observation')
                applied = live['settings']
                rate = live['real_time_factor']
                measured = f'{rate:.2f}×' if rate is not None else 'warming up'
                status.text = (
                    f"Measured simulation speed: {measured}\n"
                    f"UR controllers {applied['ur_controller_rate_hz']} Hz · KMR 225 Hz · "
                    f"physics 0.001 s · cameras {'on' if applied['enable_camera_streams'] else 'off'} · "
                    f"shadows {'on' if applied['dynamic_shadows'] else 'off'} · "
                    f"Gazebo viewer {applied['gazebo_gui_rate_hz']} FPS"
                )
            except (ValueError, TypeError, KeyError):
                status.text = 'Simulation performance observation unavailable; waiting for the running scene.'

    open_viewer.on_click(lambda: viewer(True))
    close_viewer.on_click(lambda: viewer(False))
    stop_scene.on_click(stop_simulation)
    timer(2.0, refresh)
