"""Generate S1-S3 evaluation graphs from the experiment results CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cais_spade_llm_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENTS_DIR = SCRIPT_PATH.parents[1]
RESULTS_PATH = EXPERIMENTS_DIR / "results" / "s1_s2_s3_evaluation_results.csv"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"

SCENARIO_FIELD = "Scen."
PURE_LLM_RULE_SAT_FIELD = "Pure LLM Rule Sat. (%)"
PROPOSED_FRAMEWORK_RULE_SAT_FIELD = "Proposed Framework Rule Sat. (%)"
REPAIR_ATTEMPTS_FIELD = "Repair Attempts"
VERIF_TIME_FIELD = "Verif. Time (s)"

NAVY = "#062b5f"
BLUE = "#0057b8"
ORANGE = "#f58200"
GRID = "#d7dce2"
TEXT = "#111827"
BORDER = "#a8b2bd"


def load_rows() -> list[dict[str, str]]:
    with RESULTS_PATH.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def style_axes(ax: plt.Axes, x_grid: bool = False, y_grid: bool = False) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=TEXT, labelsize=9, length=0)
    if x_grid:
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.85, color=GRID)
    if y_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.85, color=GRID)
    ax.set_axisbelow(True)


def add_header(
    ax: plt.Axes,
    title: str,
    icon_text: str,
    height: float = 0.16,
    title_size: int = 12,
) -> None:
    ax.add_patch(
        Rectangle(
            (0, 1.02),
            1,
            height,
            transform=ax.transAxes,
            color=NAVY,
            clip_on=False,
            zorder=10,
        )
    )
    ax.scatter(
        [0.06],
        [1.02 + height / 2],
        s=420,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="white",
        linewidth=1.0,
        clip_on=False,
        zorder=11,
    )
    ax.text(
        0.06,
        1.02 + height / 2,
        icon_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=title_size - 2,
        fontweight="bold",
        clip_on=False,
        zorder=12,
    )
    ax.text(
        0.11,
        1.02 + height / 2,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="white",
        fontsize=title_size,
        fontweight="bold",
        clip_on=False,
        zorder=12,
    )


def draw_header_axis(ax: plt.Axes, title: str, icon_text: str, title_size: int) -> None:
    ax.set_axis_off()
    ax.add_patch(
        Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=NAVY, clip_on=False)
    )
    ax.scatter(
        [0.075],
        [0.5],
        s=680,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="white",
        linewidth=1.0,
        clip_on=False,
    )
    ax.text(
        0.075,
        0.5,
        icon_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=title_size - 1,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.135,
        0.5,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="white",
        fontsize=title_size,
        fontweight="bold",
        clip_on=False,
    )


def label_section(
    ax: plt.Axes,
    title: str,
    icon_text: str,
    y: float = 1.09,
) -> None:
    ax.scatter(
        [-0.08],
        [y],
        s=260,
        transform=ax.transAxes,
        facecolor=BLUE,
        edgecolor=BLUE,
        clip_on=False,
        zorder=10,
    )
    ax.text(
        -0.08,
        y,
        icon_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
        zorder=11,
    )
    ax.text(
        0.0,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
        clip_on=False,
        zorder=11,
    )


def add_value_labels(ax: plt.Axes, bars, suffix: str = "", decimals: int = 2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{decimals}f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=TEXT,
        )


def add_horizontal_value_labels(
    ax: plt.Axes,
    bars,
    decimals: int = 2,
    offset: float = 0.08,
) -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.{decimals}f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=TEXT,
        )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)


def plot_rule_satisfaction_rate(rows: list[dict[str, str]]) -> None:
    scenarios = [row[SCENARIO_FIELD] for row in rows]
    pure_llm = [as_float(row, PURE_LLM_RULE_SAT_FIELD) for row in rows]
    proposed_framework = [
        as_float(row, PROPOSED_FRAMEWORK_RULE_SAT_FIELD) for row in rows
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    fig.patch.set_facecolor("white")
    x_positions = range(len(scenarios))
    width = 0.34
    pure_bars = ax.bar(
        [x - width / 2 for x in x_positions],
        pure_llm,
        width,
        label="Pure LLM",
        color=BLUE,
        edgecolor=BLUE,
    )
    proposed_bars = ax.bar(
        [x + width / 2 for x in x_positions],
        proposed_framework,
        width,
        label="Proposed Framework",
        color=ORANGE,
        edgecolor=ORANGE,
    )

    add_header(ax, "Safety Rule Satisfaction", "S")
    ax.set_ylabel("Rule Satisfaction (%)", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.55, len(scenarios) - 0.45)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=0.9,
    )
    style_axes(ax, y_grid=True)
    add_value_labels(ax, pure_bars)
    add_value_labels(ax, proposed_bars)
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.9))
    save_figure(fig, "rule_satisfaction_rate")


def plot_horizontal_series(
    rows: list[dict[str, str]],
    field: str,
    title: str,
    output_name: str,
    xmax: float,
    xticks: list[float],
    section_icon: str,
    header: str | None = None,
) -> None:
    scenarios = [row[SCENARIO_FIELD] for row in rows]
    values = [as_float(row, field) for row in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    fig.patch.set_facecolor("white")
    bars = ax.barh(scenarios, values, height=0.48, color=BLUE, edgecolor=BLUE)
    ax.invert_yaxis()
    if header is not None:
        add_header(ax, header, "C")
    section_y = 0.98 if header is not None else 1.09
    label_section(ax, title, section_icon, y=section_y)
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_axes(ax, x_grid=True)
    add_horizontal_value_labels(ax, bars, offset=xmax * 0.025)
    fig.tight_layout(rect=(0.12, 0.07, 0.98, 0.84))
    save_figure(fig, output_name)


def plot_cost_axis(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    field: str,
    title: str,
    xmax: float,
    xticks: list[float],
    section_icon: str,
    section_y: float = 1.09,
) -> None:
    scenarios = [row[SCENARIO_FIELD] for row in rows]
    values = [as_float(row, field) for row in rows]
    bars = ax.barh(scenarios, values, height=0.42, color=BLUE, edgecolor=BLUE)
    ax.invert_yaxis()
    label_section(ax, title, section_icon, y=section_y)
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.set_xlabel("")
    ax.set_ylabel("")
    style_axes(ax, x_grid=True)
    add_horizontal_value_labels(ax, bars, offset=xmax * 0.025)


def plot_evaluation_dashboard(rows: list[dict[str, str]]) -> None:
    scenarios = [row[SCENARIO_FIELD] for row in rows]
    pure_llm = [as_float(row, PURE_LLM_RULE_SAT_FIELD) for row in rows]
    proposed_framework = [
        as_float(row, PROPOSED_FRAMEWORK_RULE_SAT_FIELD) for row in rows
    ]

    fig = plt.figure(figsize=(10.4, 5.0), facecolor="white")
    left_header_ax = fig.add_axes((0.055, 0.79, 0.57, 0.13))
    rule_ax = fig.add_axes((0.055, 0.13, 0.57, 0.61))
    repair_ax = fig.add_axes((0.72, 0.61, 0.255, 0.22))
    time_ax = fig.add_axes((0.72, 0.29, 0.255, 0.22))

    x_positions = range(len(scenarios))
    width = 0.34
    pure_bars = rule_ax.bar(
        [x - width / 2 for x in x_positions],
        pure_llm,
        width,
        label="Pure LLM",
        color=BLUE,
        edgecolor=BLUE,
    )
    proposed_bars = rule_ax.bar(
        [x + width / 2 for x in x_positions],
        proposed_framework,
        width,
        label="Proposed Framework",
        color=ORANGE,
        edgecolor=ORANGE,
    )
    draw_header_axis(left_header_ax, "Safety Rule Satisfaction", "S", 13)
    rule_ax.set_ylabel(
        "Rule Satisfaction (%)", fontsize=11, fontweight="bold", color=TEXT
    )
    rule_ax.set_xticks(list(x_positions))
    rule_ax.set_xticklabels(scenarios, fontsize=12, fontweight="bold")
    rule_ax.set_ylim(0, 110)
    rule_ax.set_xlim(-0.55, len(scenarios) - 0.45)
    rule_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=0.9,
    )
    style_axes(rule_ax, y_grid=True)
    add_value_labels(rule_ax, pure_bars)
    add_value_labels(rule_ax, proposed_bars)

    plot_cost_axis(
        repair_ax,
        rows,
        REPAIR_ATTEMPTS_FIELD,
        "Repair Attempts",
        5,
        [0, 1, 2, 3, 4, 5],
        "R",
        section_y=1.13,
    )
    plot_cost_axis(
        time_ax,
        rows,
        VERIF_TIME_FIELD,
        "Verification Time (s)",
        30,
        [0, 5, 10, 15, 20, 25, 30],
        "T",
        section_y=1.13,
    )

    fig.text(
        0.79,
        0.12,
        "Cost increases with scale.",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
    )
    save_figure(fig, "s1_s2_s3_evaluation_dashboard")


def main() -> None:
    rows = load_rows()
    plot_rule_satisfaction_rate(rows)
    plot_horizontal_series(
        rows,
        REPAIR_ATTEMPTS_FIELD,
        "Repair Attempts",
        "repair_attempts",
        5,
        [0, 1, 2, 3, 4, 5],
        "R",
    )
    plot_horizontal_series(
        rows,
        VERIF_TIME_FIELD,
        "Verification Time (s)",
        "verification_time",
        30,
        [0, 5, 10, 15, 20, 25, 30],
        "T",
    )
    plot_evaluation_dashboard(rows)


if __name__ == "__main__":
    main()
