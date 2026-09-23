"""Reusable resource status cards using each resource's declared state fields."""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

from cais_spade_llm.ui.components.status_badge import status_badge
from cais_spade_llm.ui.resource_status import SNAPSHOT_EVIDENCE, TELEMETRY_FIELDS

_UNAVAILABLE = object()


def _text(value: Any) -> str:
    if value is _UNAVAILABLE:
        return "Unavailable"
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _fields(state: dict, variables: dict) -> list[str]:
    fields = []
    for field in variables:
        if "{part_name}" not in field:
            fields.append(field)
            continue
        prefix, suffix = field.split("{part_name}")
        fields.extend(key for key in state if key.startswith(prefix) and key.endswith(suffix))
    return list(dict.fromkeys(fields))


def _table(label: str, rows: list[dict], columns: tuple[str, ...], row_key: str) -> None:
    ui.label(label).classes("text-sm font-semibold mt-2")
    if not rows:
        ui.label("Unavailable").classes("text-sm text-slate-500")
        return
    ui.table(
        columns=[{"name": key, "label": key, "field": key, "align": "left"} for key in columns],
        rows=rows,
        row_key=row_key,
    ).classes("w-full").props("dense flat bordered wrap-cells")


def _group_fields(fields: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    groups: dict[str, list[str]] = {
        "inventory": [],
        "output": [],
        "Conveyor occupancy": [],
        "Buffer occupancy": [],
    }
    scalar_fields = []
    for field in fields:
        if field in {"current_state", "resource_state", *TELEMETRY_FIELDS}:
            continue
        if field.startswith("inventory."):
            groups["inventory"].append(field)
        elif field.startswith("output."):
            groups["output"].append(field)
        elif field.startswith(("part_location.", "part_order.")):
            groups["Conveyor occupancy"].append(field)
        elif field.startswith("zone_") and field.endswith("_part"):
            groups["Buffer occupancy"].append(field)
        else:
            scalar_fields.append(field)
    return scalar_fields, groups


def render_robot_status_card(
    name: str,
    state: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
    evidence: str = SNAPSHOT_EVIDENCE,
) -> None:
    """Render current values without supplying assumed phases or observations.

    Args:
        name: Exact resource ID.
        state: Current runtime valuation or supplied agent snapshot.
        model: Descriptor supplying field declarations, never fallback values.
        evidence: Source of the displayed values.
    """
    variables = model["state_variables"] if model is not None else {}
    fields = _fields(state, variables) if model is not None else list(state)
    current = state.get("resource_state", state.get("current_state", _UNAVAILABLE))
    has_control_state = model is None or "resource_state" in variables
    phases = variables.get("resource_state", {}).get("domain", [])
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-4"):
            ui.label(name).classes("text-lg font-bold")
            if has_control_state:
                status_badge(_text(current))
        ui.label(evidence).classes("text-xs text-slate-500")

        if phases:
            with ui.row().classes("gap-1 mt-2"):
                for phase in phases:
                    color = (
                        "bg-blue-500 text-white"
                        if phase == current
                        else "bg-slate-200 text-slate-600"
                    )
                    ui.label(_text(phase)).classes(f"px-3 py-1 rounded text-xs font-mono {color}")

        scalar_fields, groups = _group_fields(fields)
        with ui.row().classes("gap-8 mt-3 flex-wrap"):
            for field in scalar_fields:
                _detail(field, _text(state.get(field, _UNAVAILABLE)))
            for field in TELEMETRY_FIELDS:
                if field in state:
                    _detail(field, _text(state[field]))

        for label, grouped in groups.items():
            if label == "Conveyor occupancy":
                parts = list(dict.fromkeys(field.split(".", 1)[1] for field in grouped))
                rows = [
                    {
                        "part_name": part,
                        "part_location": _text(state.get(f"part_location.{part}", _UNAVAILABLE)),
                        "part_order": _text(state.get(f"part_order.{part}", _UNAVAILABLE)),
                    }
                    for part in parts
                ]
                if grouped or "part_location.{part_name}" in variables:
                    _table(label, rows, ("part_name", "part_location", "part_order"), "part_name")
            elif grouped or label + ".{part_name}" in variables:
                rows = [
                    {"field": field, "value": _text(state.get(field, _UNAVAILABLE))}
                    for field in grouped
                ]
                _table(label, rows, ("field", "value"), "field")


def render_environment_outcome(outcome: dict[str, Any]) -> None:
    """Display execution outcome independently of agent startup success.

    Args:
        outcome: The active environmental runtime's outcome and selected tasks.
    """
    if not outcome:
        return
    status = outcome.get("status", "Unavailable")
    with ui.column().classes("w-full gap-2"):
        status_badge(str(status))
        if outcome.get("reason"):
            ui.label(outcome["reason"]).classes("text-sm text-amber-800")
        unavailable = outcome.get("execution_unavailable", [])
        tasks = unavailable or outcome.get("tasks", [])
        if tasks:
            rows_by_action = {}
            for task in tasks:
                key = (task["resource_id"], task["event_name"])
                rows_by_action[key] = {
                    "resource_id": task["resource_id"],
                    "event_name": task["event_name"],
                    "key": json.dumps(key, ensure_ascii=False),
                }
            rows = list(rows_by_action.values())
            _table(
                "Unavailable actions" if unavailable else "Selected actions",
                rows,
                ("resource_id", "event_name"),
                "key",
            )
        if outcome.get("details"):
            with ui.expansion("Execution details", icon="info"):
                ui.code(
                    json.dumps(outcome["details"], indent=2, ensure_ascii=False), language="json"
                )


def _detail(label: str, value: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-xs text-slate-500")
        ui.label(value).classes("text-sm font-semibold")
