"""Log viewer page: dual-pane live log streaming."""

from __future__ import annotations

import asyncio
import re

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.polling import LogTailer


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Log Viewer").classes("text-2xl font-bold")

        log_paths = bridge.get_log_paths()

        if not log_paths:
            ui.label("No log files found.").classes("text-slate-400 italic")
            return

        with ui.row().classes("w-full gap-4"):
            for name, path in log_paths.items():
                with ui.card().classes("flex-1 min-w-[400px]"):
                    ui.label(f"{name}").classes("text-lg font-semibold mb-1")
                    ui.label(path).classes("text-xs text-slate-400 mb-2 truncate")

                    filter_input = ui.input(placeholder="Regex filter...").classes("w-full mb-2")
                    log_widget = ui.log(max_lines=500).classes("w-full h-96")

                    _start_tail(log_widget, filter_input, path)


def _start_tail(log_widget: ui.log, filter_input: ui.input, path: str) -> None:
    """Start an async task to tail a log file into a NiceGUI log widget."""
    tailer = LogTailer(path, initial_lines=200, poll_interval=0.5)

    async def _tail():
        async for line in tailer:
            pattern = filter_input.value
            if pattern:
                try:
                    if not re.search(pattern, line, re.IGNORECASE):
                        continue
                except re.error:
                    pass
            log_widget.push(line)

    task = asyncio.create_task(_tail())

    # Clean up when the client disconnects.
    async def _cleanup():
        tailer.stop()
        task.cancel()

    ui.on("disconnect", _cleanup)
