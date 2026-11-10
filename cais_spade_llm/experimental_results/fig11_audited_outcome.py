"""Fig. 11: audited outcome of each started attempt, by setting and failure scenario, all models pooled.

One file per failure scenario and one for the legend, so the panels can be arranged and lettered in PowerPoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

from cais_spade_llm.experimental_results.page_data import pooled_cell
from cais_spade_llm.experimental_results.style import (
    BAR_HEIGHT,
    FULL_WIDTH_IN,
    GROUP_STEP,
    HAIR,
    HEADER_OFFSET,
    INK,
    INK_2,
    MUTED,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    ROW_OFFSETS,
    SETTING_GROUPS,
    apply_style,
    level_label,
    save,
)

NAME = "fig11"
# Half the text width, so two panels side by side fill the page at full size.
PANEL_SIZE_IN = (FULL_WIDTH_IN / 2, 2.15)
# Fig. 11 shows these two audit outcomes as one segment (user decision, 2026-11-08): neither leaves an
# established outline.
JOINED = ("unresolved", "did_not_finish")
JOINED_LABEL = "No outline established"
NOT_AUDITED = "not audited"
# Segments narrower than this share of the bar get no count label.
MIN_LABELLED_SHARE = 0.08
# Median effort per attempt as three columns right of the bars: header, cell key, and the column's center in
# points from the bars' right end. Units go on a second header line to keep the columns narrow. The caption says
# these are medians.
EFFORT_COLUMNS = (("Tokens\n(k)", "tokens_k", 17), ("Time\n(min)", "minutes", 43), ("Lookups", "lookups", 72))


def _segments(classes: list[str], counts: list[int]) -> list[tuple[str, int, str]]:
    """Return (label, count, color) in the audit's order, with the joined pair at the first of the two."""
    by_name = dict(zip(classes, counts))
    segments = []
    for name in classes:
        if name == JOINED[1]:
            continue
        if name == JOINED[0]:
            segments.append((JOINED_LABEL, by_name[JOINED[0]] + by_name[JOINED[1]], OUTCOME_COLORS[JOINED[0]]))
        else:
            segments.append((OUTCOME_LABELS[name], by_name[name], OUTCOME_COLORS[name]))
    return segments


def _draw_effort(ax: Axes, y: float, texts: list[str], **style: Any) -> None:
    """Write one row of the effort columns, each centered on its column."""
    for (_header, _key, center), text in zip(EFFORT_COLUMNS, texts):
        transform = offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=center, units="points")
        ax.text(1.0, y, text, transform=transform, ha="center", **style)


def _effort_texts(cell: dict[str, Any]) -> list[str]:
    """Return a cell's median tokens, time and lookups, in the order of ``EFFORT_COLUMNS``."""
    return ["–" if cell[key] is None else str(cell[key]) for _header, key, _center in EFFORT_COLUMNS]


def _draw_scenario(ax: Axes, data: dict[str, Any], scenario: str) -> None:
    """Draw one scenario's stacked bars, one per setting, with each row's median effort at the right."""
    labels = {}
    for group_index, (method, settings) in enumerate(SETTING_GROUPS):
        top = group_index * GROUP_STEP
        if group_index:
            ax.axhline(top - 0.25, color=HAIR, linewidth=0.6, clip_on=False, xmin=-0.02, xmax=1.0)
        ax.text(0, top + HEADER_OFFSET, method, fontweight="bold", fontsize=7, va="center", ha="left")
        for setting, offset in zip(settings, ROW_OFFSETS):
            y = top + offset
            labels[y] = level_label(data["settings"][setting]["n"])
            cell = pooled_cell(data, setting, scenario)
            left = 0.0
            for _label, count, color in _segments(data["classes"], cell["classes"]):
                if not count:
                    continue
                share = count / cell["n"]
                ax.barh(y, share, left=left, height=BAR_HEIGHT, color=color, edgecolor="white", linewidth=0.6)
                if share >= MIN_LABELLED_SHARE:
                    ax.text(left + share / 2, y, str(count), ha="center", va="center", fontsize=6.5, color=INK)
                left += share
            _draw_effort(ax, y, _effort_texts(cell), va="center", fontsize=7, color=INK)
    # Headers end just above the first bar, so a one-line header sits on the second line of the others.
    _draw_effort(ax, ROW_OFFSETS[0] - BAR_HEIGHT / 2 - 0.05, [header for header, *_rest in EFFORT_COLUMNS],
                 va="bottom", fontsize=6.5, color=MUTED, fontweight="bold", linespacing=1.1)
    ax.set_yticks(list(labels), list(labels.values()))
    ax.tick_params(axis="y", labelsize=7, pad=3)
    ax.tick_params(axis="x", labelsize=6.5, length=2, colors=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(SETTING_GROUPS) * GROUP_STEP - 0.6, -0.35)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.grid(axis="x", color=HAIR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _slug(scenario: str) -> str:
    return scenario.lower().replace(" ", "-")


def build(data: dict[str, Any], out_dir: Path) -> list[Path]:
    """Draw Fig. 11 from the page data: one file per scenario, then the legend.

    Returns:
        The written paths.
    """
    apply_style()
    paths = []
    for scenario in data["scenarios"]:
        fig = Figure(figsize=PANEL_SIZE_IN)
        ax = fig.add_axes((0.08, 0.08, 0.58, 0.88))
        _draw_scenario(ax, data, scenario)
        paths += save(fig, out_dir, f"{NAME}-{_slug(scenario)}")

    any_not_audited = any(cell[scenario]["classes"][data["classes"].index(NOT_AUDITED)]
                          for cell in data["cells"]["all"] for scenario in data["scenarios"])
    handles = [Patch(facecolor=color, edgecolor=HAIR, linewidth=0.5, label=label)
               for label, _count, color in _segments(data["classes"], [0] * len(data["classes"]))
               if label != OUTCOME_LABELS[NOT_AUDITED] or any_not_audited]
    legend = Figure(figsize=(FULL_WIDTH_IN, 0.3))
    legend.legend(handles=handles, loc="center", ncol=len(handles), fontsize=7, handlelength=1.1, handleheight=0.9,
                  columnspacing=1.2, handletextpad=0.4)
    return paths + save(legend, out_dir, f"{NAME}-legend")
