"""Perception page: detected parts table + 2D workspace scatter plot."""

from __future__ import annotations

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge

# Workspace boundaries (mm) from robot configs.
_WORKSPACES = {
    "xarm6": {"x_min": -150, "x_max": 650, "y_min": -800, "y_max": -400, "color": "rgba(33, 150, 243, 0.15)"},
    "ur5e": {"x_min": 100, "x_max": 900, "y_min": -800, "y_max": -400, "color": "rgba(76, 175, 80, 0.15)"},
}


def render(bridge: SystemBridge) -> None:
    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-6"):
        ui.label("Perception").classes("text-2xl font-bold")

        # ── Detected Parts Table ─────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Detected Parts").classes("text-lg font-semibold mb-2")

            parts_table = ui.table(
                columns=[
                    {"name": "part", "label": "Part", "field": "part", "sortable": True},
                    {"name": "x", "label": "X (mm)", "field": "x"},
                    {"name": "y", "label": "Y (mm)", "field": "y"},
                    {"name": "z", "label": "Z (mm)", "field": "z"},
                ],
                rows=[],
            ).classes("w-full")

            async def _refresh_parts():
                # Try to get observations from camera via bridge.
                observations = _get_observations(bridge)
                rows = [
                    {"part": name, "x": round(p.get("x", 0), 1), "y": round(p.get("y", 0), 1), "z": round(p.get("z", 0), 1)}
                    for name, p in observations.items()
                ]
                parts_table.rows = rows
                _update_plot(plot, observations)

            ui.button("Refresh", on_click=_refresh_parts, icon="refresh").classes("mb-2")

        # ── 2D Workspace Plot ────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Workspace View (top-down)").classes("text-lg font-semibold mb-2")
            plot = ui.plotly({
                "data": [],
                "layout": _base_layout(),
            }).classes("w-full h-96")

            # Initial load.
            observations = _get_observations(bridge)
            _update_plot(plot, observations)

            ui.timer(5.0, _refresh_parts)


def _get_observations(bridge: SystemBridge) -> dict:
    """Get part observations from product agents' camera or part tracker."""
    # Try camera on product agents.
    for pa in bridge.product_agents:
        cam = getattr(pa, "camera", None)
        if cam and hasattr(cam, "observe_all"):
            try:
                return cam.observe_all()
            except Exception:
                pass
    # Fallback: use part tracker positions.
    pt = bridge.get_part_tracker()
    return {k: v.get("position", {}) for k, v in pt.items() if v.get("position")}


def _base_layout() -> dict:
    return {
        "title": "Part Positions (Top-Down)",
        "xaxis": {"title": "X (mm)", "range": [-200, 1000]},
        "yaxis": {"title": "Y (mm)", "range": [-900, 100], "scaleanchor": "x"},
        "showlegend": True,
        "shapes": [
            {
                "type": "rect",
                "x0": ws["x_min"], "x1": ws["x_max"],
                "y0": ws["y_min"], "y1": ws["y_max"],
                "fillcolor": ws["color"],
                "line": {"dash": "dash", "width": 1},
            }
            for ws in _WORKSPACES.values()
        ],
        "annotations": [
            {"x": (ws["x_min"] + ws["x_max"]) / 2, "y": ws["y_max"] + 20,
             "text": name, "showarrow": False, "font": {"size": 12}}
            for name, ws in _WORKSPACES.items()
        ],
    }


def _update_plot(plot, observations: dict) -> None:
    if not observations:
        plot.update_figure(data=[], layout=_base_layout())
        return

    xs = [p.get("x", 0) for p in observations.values()]
    ys = [p.get("y", 0) for p in observations.values()]
    names = list(observations.keys())

    plot.update_figure(
        data=[{
            "type": "scatter",
            "x": xs,
            "y": ys,
            "mode": "markers+text",
            "text": names,
            "textposition": "top center",
            "marker": {"size": 12, "color": "#f44336"},
            "name": "Parts",
        }],
        layout=_base_layout(),
    )
