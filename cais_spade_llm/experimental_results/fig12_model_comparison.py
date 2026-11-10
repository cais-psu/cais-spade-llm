"""Fig. 12: model comparison, as the results page draws it.

Attempts passed per setting, pooled over the four scenarios, with each model's 95% Wilson interval and a grey bar
for every model's attempts pooled; then the share of checked candidate events each check rejected in the
event-level framework setting. Panel (a), panel (b) and the legend are each one file, so they can be arranged and
lettered in PowerPoint.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from cais_spade_llm.experimental_results.page_data import model_cell
from cais_spade_llm.experimental_results.style import (
    BAR_HEIGHT,
    COLUMN_WIDTH_IN,
    GROUP_STEP,
    HAIR,
    HEADER_OFFSET,
    INK,
    INK_2,
    MARKERS,
    MODEL_COLORS,
    OUTCOME_LABELS,
    ROW_OFFSETS,
    SETTING_GROUPS,
    apply_style,
    level_label,
    save,
)

NAME = "fig12"
FW_EVENT_LEVEL = 5
# The checks run on each candidate event, by the audit outcome each one sets (Fig. 11's name, in words only: no
# outcome color beside it, user decision 2026-11-09), with the key of each one's rejection count. Blocking (CCA) runs once per outline, when the LLM declares it complete, so it
# has no share of checked events and is not drawn.
CHECKS = (("infeasible", "ra"), ("physically_unsupported", "physical"), ("unsafe", "cca"))
# The total row: the share of checked events that the checks rejected together (each event counts once).
ALL_CHECKS = "All checks"
POOLED_LABEL = "All models pooled"
POOLED_ALPHA = 0.12
# One value column per model right of the plot: each column's center in points from the plot's right end.
COLUMN_CENTERS_PT = (24, 52, 80)
# Marker sizes in points; a diamond reads larger than a square or circle of the same size.
MARK_SIZES = {"s": 3.6, "o": 3.9, "D": 3.1}
# Vertical spread of the models' marks within a row, in data units, so every mark stays visible.
MARK_STEP = 0.24
# Rows of the rejection panel, in data units: a check's name, then its marks; the sum of the checks comes last.
CHECK_STEP = 1.5
CHECK_ROW_OFFSET = 0.85
ALL_CHECKS_Y = len(CHECKS) * CHECK_STEP + HEADER_OFFSET
# Each panel's y limits in data units, bottom first because rows run downward.
PASSED_YLIM = (len(SETTING_GROUPS) * GROUP_STEP - 0.6, -0.35)
REJECTED_YLIM = (ALL_CHECKS_Y + 0.6, -0.9)
# Layout in inches: both panels share the margins and the scale of a data unit, so they line up when stacked; each
# file holds a panel's plot and tick labels, without a title (the user adds titles in PowerPoint).
LEFT_IN, RIGHT_IN = 0.3, (COLUMN_CENTERS_PT[-1] + 16) / 72
UNIT_IN = 0.2
TOP_IN, TICKS_IN = 0.05, 0.22


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval of k successes in n trials, as the results page computes it."""
    if not n:
        return 0.0, 0.0
    z2 = z * z
    center = (k + z2 / 2) / (n + z2)
    half = (z / (n + z2)) * math.sqrt(k * (n - k) / n + z2 / 4)
    return max(0.0, center - half), min(1.0, center + half)


def _pct(share: float) -> str:
    """Return a share as a whole percentage, rounding halves up as the page does."""
    return f"{math.floor(share * 100 + 0.5)}%"


def rounded_parts(values: list[float]) -> tuple[list[int], int]:
    """Return whole percentages for shares that add up to a total, and the total, as the results page prints them.

    The total rounds as usual and the parts take the largest remainders, so the parts add up to the total.
    """
    total = math.floor(sum(values) * 100 + 0.5)
    parts = [math.floor(value * 100 + 1e-9) for value in values]
    order = sorted(range(len(values)), key=lambda index: values[index] * 100 - parts[index], reverse=True)
    for index in order[: total - sum(parts)]:
        parts[index] += 1
    return parts, total


def _passed(data: dict[str, Any], model_id: str, setting: int) -> tuple[int, int]:
    """Return a model's passed and started attempts in one setting, over the four scenarios."""
    cells = [model_cell(data, model_id, setting, scenario) for scenario in data["scenarios"]]
    return sum(cell["classes"][0] for cell in cells), sum(cell["n"] for cell in cells)


def _rejection_shares(data: dict[str, Any], model_id: str) -> list[float | None]:
    """Return the share of a model's checked candidate events each check rejected in the event-level framework."""
    cells = [cell for scenario in data["scenarios"]
             if (cell := model_cell(data, model_id, FW_EVENT_LEVEL, scenario))["n"] and cell["rejections"]
             and cell["proposals"]]
    checked = sum(cell["proposals"] for cell in cells)
    return [sum(cell["rejections"][key] for cell in cells) / checked if checked else None for _label, key in CHECKS]


def _panel_height(ylim: tuple[float, float]) -> float:
    return (ylim[0] - ylim[1]) * UNIT_IN


def _panel_figure(ylim: tuple[float, float]) -> tuple[Figure, Axes]:
    """Return a column-wide figure holding one panel, with the panel's tick labels below it."""
    height = TOP_IN + _panel_height(ylim) + TICKS_IN
    fig = Figure(figsize=(COLUMN_WIDTH_IN, height))
    ax = fig.add_axes((LEFT_IN / COLUMN_WIDTH_IN, TICKS_IN / height, 1 - (LEFT_IN + RIGHT_IN) / COLUMN_WIDTH_IN,
                       _panel_height(ylim) / height))
    ax.set_ylim(*ylim)
    return fig, ax


def _column_transform(ax: Axes, column: int) -> Any:
    return offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=COLUMN_CENTERS_PT[column], units="points")


def _column_marks(ax: Axes, y: float, models: list[dict[str, Any]]) -> None:
    """Head each model's value column with the model's mark."""
    for column, model in enumerate(models):
        marker = MARKERS[model["mark"]]
        ax.plot([1.0], [y], linestyle="none", marker=marker, markersize=MARK_SIZES[marker],
                color=MODEL_COLORS[model["color"]],
                transform=_column_transform(ax, column), clip_on=False)


def _column_texts(ax: Axes, y: float, texts: list[str], **style: Any) -> None:
    for column, text in enumerate(texts):
        ax.text(1.0, y, text, transform=_column_transform(ax, column), ha="center", va="center", **style)


def _model_y(y: float, index: int, count: int) -> float:
    return y + (index - (count - 1) / 2) * MARK_STEP


def _finish_axes(ax: Axes, ticks: list[float]) -> None:
    ax.set_xticks(ticks, [_pct(tick) for tick in ticks])
    ax.set_xlim(0, ticks[-1])
    ax.tick_params(axis="x", labelsize=6.5, length=2, colors=INK_2)
    ax.grid(axis="x", color=HAIR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _draw_passed(ax: Axes, data: dict[str, Any]) -> None:
    """Draw each setting's grey pooled bar and every model's pass share with its Wilson interval."""
    models = data["models"]
    labels = {}
    for group_index, (method, settings) in enumerate(SETTING_GROUPS):
        top = group_index * GROUP_STEP
        if group_index:
            ax.axhline(top - 0.25, color=HAIR, linewidth=0.6, clip_on=False, xmin=-0.02, xmax=1.0)
        ax.text(0, top + HEADER_OFFSET, method, fontweight="bold", fontsize=7, va="center", ha="left")
        for setting, offset in zip(settings, ROW_OFFSETS):
            y = top + offset
            labels[y] = level_label(data["settings"][setting]["n"])
            passed = [_passed(data, model["id"], setting) for model in models]
            pooled_k, pooled_n = sum(k for k, _n in passed), sum(n for _k, n in passed)
            if pooled_n:
                ax.barh(y, pooled_k / pooled_n, height=BAR_HEIGHT, color=to_rgba(INK_2, POOLED_ALPHA), linewidth=0)
            texts = []
            for index, (model, (k, n)) in enumerate(zip(models, passed)):
                if not n:
                    texts.append("–")
                    continue
                share, (low, high) = k / n, wilson(k, n)
                marker = MARKERS[model["mark"]]
                bars = ax.errorbar(share, _model_y(y, index, len(models)), xerr=[[share - low], [high - share]],
                                   fmt=marker, markersize=MARK_SIZES[marker], color=MODEL_COLORS[model["color"]],
                                   ecolor=to_rgba(MODEL_COLORS[model["color"]], 0.55), elinewidth=0.9, capsize=1.5,
                                   capthick=0.9)
                for artist in bars.get_children():
                    artist.set_clip_on(False)
                texts.append(_pct(share))
            _column_texts(ax, y, texts, fontsize=7, color=INK)
    _column_marks(ax, HEADER_OFFSET, models)
    ax.set_yticks(list(labels), list(labels.values()))
    ax.tick_params(axis="y", labelsize=7, pad=3)
    _finish_axes(ax, [0, 0.25, 0.5, 0.75, 1])


def _draw_rejected(ax: Axes, data: dict[str, Any]) -> None:
    """Draw the share of each model's checked events that each check rejected, then their sum."""
    models = data["models"]
    shares = [_rejection_shares(data, model["id"]) for model in models]
    largest = max((value for values in shares for value in values if value is not None), default=0.0)
    # Scale in 10% steps, so the gridlines fall on round shares, as the page does.
    limit = max(0.1, math.ceil(round(largest / 0.1, 9)) * 0.1)
    rounded = [None if None in values else rounded_parts(values) for values in shares]
    for row, (outcome, _key) in enumerate(CHECKS):
        top = row * CHECK_STEP
        y = top + CHECK_ROW_OFFSET
        ax.text(0, top + HEADER_OFFSET, OUTCOME_LABELS[outcome], fontsize=7, va="center", ha="left")
        ax.axhline(y, color=HAIR, linewidth=0.6, zorder=0)
        texts = []
        for index, (model, values) in enumerate(zip(models, shares)):
            if values[row] is None:
                texts.append("–")
                continue
            marker = MARKERS[model["mark"]]
            ax.plot(values[row], _model_y(y, index, len(models)), linestyle="none", marker=marker,
                    markersize=MARK_SIZES[marker], color=MODEL_COLORS[model["color"]], clip_on=False)
            texts.append(f"{rounded[index][0][row]}%")
        _column_texts(ax, y, texts, fontsize=7, color=INK)
    ax.text(0, ALL_CHECKS_Y, ALL_CHECKS, fontsize=7, fontweight="bold", va="center", ha="left")
    _column_texts(ax, ALL_CHECKS_Y, ["–" if parts is None else f"{parts[1]}%" for parts in rounded], fontsize=7,
                  fontweight="bold", color=INK)
    _column_marks(ax, -0.45, models)
    ax.set_yticks([])
    _finish_axes(ax, [0, limit / 2, limit])


def build(data: dict[str, Any], out_dir: Path) -> list[Path]:
    """Draw Fig. 12 from the page data: panel (a), panel (b) and the legend, each in its own file.

    Returns:
        The written paths.
    """
    apply_style()
    fig, ax = _panel_figure(PASSED_YLIM)
    _draw_passed(ax, data)
    paths = save(fig, out_dir, f"{NAME}a-outlines-passed")

    fig, ax = _panel_figure(REJECTED_YLIM)
    _draw_rejected(ax, data)
    paths += save(fig, out_dir, f"{NAME}b-checked-events-rejected")

    handles = [Line2D([], [], color=MODEL_COLORS[model["color"]], marker=MARKERS[model["mark"]],
                      markersize=MARK_SIZES[MARKERS[model["mark"]]], linewidth=0.9,
                      label=model["name"].removesuffix(" medium"))
               for model in data["models"]]
    handles.append(Patch(facecolor=to_rgba(INK_2, POOLED_ALPHA), label=POOLED_LABEL))
    legend = Figure(figsize=(COLUMN_WIDTH_IN * 2, 0.3))
    legend.legend(handles=handles, loc="center", ncol=len(handles), fontsize=7, handlelength=1.6,
                  columnspacing=1.6, handletextpad=0.5)
    return paths + save(legend, out_dir, f"{NAME}-legend")
