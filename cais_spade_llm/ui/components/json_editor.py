"""Reusable JSON editor component."""

from __future__ import annotations

import json

from nicegui import ui


def json_editor(data: dict, label: str = "JSON", rows: int = 15) -> ui.textarea:
    """Create a textarea pre-filled with pretty-printed JSON."""
    return (
        ui.textarea(
            value=json.dumps(data, indent=2),
            label=label,
        )
        .classes("w-full font-mono text-xs")
        .props(f"rows={rows}")
    )
