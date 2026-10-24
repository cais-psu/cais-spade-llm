"""Reusable status badge component."""

from __future__ import annotations

from nicegui import ui

from cais_spade_llm.resources.nominal_des import RUNNING_STATES

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
    # A resource between an activity's start and its observed finish is shown as running.
    color = _COLORS.get(status.lower(), "orange" if status in RUNNING_STATES else "grey")
    return ui.badge(status, color=color)
