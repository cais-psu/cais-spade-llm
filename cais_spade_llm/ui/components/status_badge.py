"""Reusable status badge component."""

from __future__ import annotations

from nicegui import ui

_COLORS = {
    "idle": "grey",
    "at_pick": "blue",
    "picked": "cyan",
    "positioned": "teal",
    "placed": "green",
    "recovery_required": "red",
    "running": "orange",
    "completed": "green",
    "failed": "red",
    "pending": "grey",
    "dispatched": "blue",
    "blocked": "deep-orange",
    "alive": "green",
    "dead": "red",
}


def status_badge(status: str) -> ui.badge:
    color = _COLORS.get(status.lower(), "grey")
    return ui.badge(status, color=color)
