"""Draw the case-study cell as a top-view schematic for Journal Paper 2.

Positions are in metres and follow the configured Gazebo layout in
``cais_spade_llm/initialization/recovery_framework_gazebo.json`` and
``MACHINING_STATION_LAYOUT.md``. Robot, tray, and label sizes are drawn
slightly larger than life where the true size would not be legible.

Run from any directory:

    python case_study_setup.py

It writes ``case_study_setup.{pdf,svg,png}`` (layout and part flow) and
``case_study_setup_failures.{pdf,svg,png}`` (same layout with the four
failure scenarios marked) next to this script.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_WIDTH_MM = 180.0
X_LIMITS = (-7.10, 1.72)
Y_LIMITS = (-1.42, 3.42)

INK = "#2b2f36"
MUTED = "#5d6673"
FLOW = "#1f5fa8"
KMR_ROUTE = "#b8661d"
KMR_BODY = "#f0a648"
ROBOT = "#2f3b4c"
REACH = "#5b7ba6"
MACHINE = "#dde1e7"
STORAGE = "#dce8f5"
STORAGE_EDGE = "#3d5a80"
TABLE = "#f6e9d4"
TABLE_EDGE = "#b08850"
BELT = "#6b7280"
LOADING = "#3bb39b"
STAGING = "#bfe3dc"
PEG = "#4a78b5"
GEAR = "#e0843c"
FAILURE = "#d62728"

LABEL_SIZE = 8.0
SMALL_SIZE = 6.5

FAILURE_SCENARIOS = (
    "Conveyor breakdown",
    "ur5e-1 breakdown",
    "Machining breakdown during part processing",
    "Part slippage",
)


def _rect(ax: Axes, center: tuple[float, float], size: tuple[float, float], **style) -> Rectangle:
    x, y = center
    w, h = size
    patch = Rectangle((x - w / 2, y - h / 2), w, h, **style)
    ax.add_patch(patch)
    return patch


def _label(ax: Axes, x: float, y: float, text: str, *, bold: bool = True, size: float = LABEL_SIZE,
           ha: str = "center", va: str = "center", color: str = INK, zorder: float = 20) -> None:
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight="bold" if bold else "normal", zorder=zorder)


def _square_peg(ax: Axes, x: float, y: float, size: float = 0.085, zorder: float = 6) -> None:
    _rect(ax, (x, y), (size, size), facecolor=PEG, edgecolor="white", linewidth=0.4, zorder=zorder)


def _round_peg(ax: Axes, x: float, y: float, radius: float = 0.048, zorder: float = 6) -> None:
    ax.add_patch(Circle((x, y), radius, facecolor=PEG, edgecolor="white", linewidth=0.4, zorder=zorder))


def _gear(ax: Axes, x: float, y: float, radius: float, *, teeth: int = 9, zorder: float = 7,
          color: str = GEAR) -> None:
    points = []
    for index in range(teeth * 2):
        angle = math.pi * index / teeth
        r = radius if index % 2 == 0 else radius * 0.72
        points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="none", zorder=zorder))
    ax.add_patch(Circle((x, y), radius * 0.28, facecolor="white", edgecolor="none", zorder=zorder + 0.1))


def _robot(ax: Axes, x: float, y: float, heading_deg: float, *, reach: bool = True) -> None:
    """Draw a UR5e as a base disc with a folded two-link arm pointing at its work."""
    if reach:
        ax.add_patch(Circle((x, y), 0.85, facecolor=REACH, alpha=0.07, edgecolor="none", zorder=0.4))
        ax.add_patch(Circle((x, y), 0.85, facecolor="none", edgecolor=REACH, linewidth=0.5,
                            linestyle=(0, (3, 2)), alpha=0.8, zorder=0.5))
    heading = math.radians(heading_deg)
    elbow = (x + 0.22 * math.cos(heading + 0.55), y + 0.22 * math.sin(heading + 0.55))
    wrist = (elbow[0] + 0.20 * math.cos(heading - 0.55), elbow[1] + 0.20 * math.sin(heading - 0.55))
    ax.plot([x, elbow[0], wrist[0]], [y, elbow[1], wrist[1]], color=ROBOT, linewidth=2.4,
            solid_capstyle="round", solid_joinstyle="round", zorder=11)
    finger_dx, finger_dy = -math.sin(heading) * 0.045, math.cos(heading) * 0.045
    reach_dx, reach_dy = math.cos(heading) * 0.06, math.sin(heading) * 0.06
    for side in (-1, 1):
        start = (wrist[0] + side * finger_dx, wrist[1] + side * finger_dy)
        ax.plot([start[0], start[0] + reach_dx], [start[1], start[1] + reach_dy], color=ROBOT,
                linewidth=1.1, solid_capstyle="round", zorder=11)
    ax.plot([wrist[0] - finger_dx, wrist[0] + finger_dx], [wrist[1] - finger_dy, wrist[1] + finger_dy],
            color=ROBOT, linewidth=1.1, solid_capstyle="round", zorder=11)
    ax.add_patch(Circle((x, y), 0.12, facecolor=ROBOT, edgecolor="white", linewidth=0.6, zorder=12))
    ax.add_patch(Circle((x, y), 0.045, facecolor="white", edgecolor="none", zorder=12.5))


def _flow(ax: Axes, start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>,head_length=3.2,head_width=1.7", color=FLOW,
                                 linewidth=1.1, shrinkA=0, shrinkB=0, zorder=15))


def _dashed_route(ax: Axes, points: list[tuple[float, float]]) -> None:
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=KMR_ROUTE, linewidth=1.1, linestyle=(0, (4, 2.2)), zorder=3,
            solid_capstyle="butt")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    tail = (x1 - (x1 - x0) / length * 0.12, y1 - (y1 - y0) / length * 0.12)
    ax.add_patch(FancyArrowPatch(tail, (x1, y1), arrowstyle="-|>,head_length=3.2,head_width=1.7",
                                 color=KMR_ROUTE, linewidth=1.1, shrinkA=0, shrinkB=0, zorder=3))


def _failure_badge(ax: Axes, x: float, y: float, number: int, radius: float = 0.105) -> None:
    ax.add_patch(Circle((x, y), radius, facecolor=FAILURE, edgecolor="white", linewidth=0.8, zorder=30))
    ax.text(x, y - 0.004, str(number), ha="center", va="center", fontsize=6.5, fontweight="bold",
            color="white", zorder=31)


def _draw_storage_and_kmr(ax: Axes) -> None:
    _rect(ax, (-6.45, 2.30), (1.00, 1.25), facecolor=STORAGE, edgecolor=STORAGE_EDGE, linewidth=0.9, zorder=2)
    for tray_x in (-6.62, -6.28):
        _rect(ax, (tray_x, 2.30), (0.22, 1.08), facecolor="white", edgecolor=STORAGE_EDGE,
              linewidth=0.5, zorder=3)
    for slot_y in (1.92, 2.17, 2.42, 2.67):
        _square_peg(ax, -6.62, slot_y)
        _round_peg(ax, -6.28, slot_y)
    _label(ax, -6.45, 1.53, "Storage", va="top")

    # KMR parked on Storage_KMR_docking_pose; the platform stands sideways (yaw +90 deg).
    for dock_x in (-4.70, -2.70):
        ax.add_patch(FancyBboxPatch((dock_x - 0.315, 2.20 - 0.54), 0.63, 1.08,
                                    boxstyle="round,pad=0,rounding_size=0.06", facecolor="none",
                                    edgecolor=KMR_ROUTE, linewidth=0.7, linestyle=(0, (2, 1.6)), zorder=2))
    _dashed_route(ax, [(-5.54, 2.13), (-4.70, 2.20)])
    _dashed_route(ax, [(-5.54, 2.13), (-5.54, 3.10), (-2.70, 3.10), (-2.70, 2.20)])

    ax.add_patch(FancyBboxPatch((-5.54 - 0.315, 2.13 - 0.54), 0.63, 1.08,
                                boxstyle="round,pad=0,rounding_size=0.06", facecolor=KMR_BODY,
                                edgecolor="#8a5a14", linewidth=0.9, zorder=8))
    for wheel_x in (-5.80, -5.28):
        for wheel_y in (1.74, 2.52):
            _rect(ax, (wheel_x, wheel_y), (0.07, 0.22), facecolor=INK, edgecolor="none", zorder=8.5)
    _robot(ax, -5.54, 1.88, 165.0, reach=False)
    _label(ax, -5.54, 1.53, "KUKA KMR", va="top")


def _draw_machine_cell(ax: Axes, name: str, x: float, robot_label: str, peg_shape: str) -> None:
    _rect(ax, (x, 1.90), (0.531, 0.503), facecolor=MACHINE, edgecolor=INK, linewidth=0.9, zorder=5)
    _label(ax, x, 1.97, name, size=8.5)
    if peg_shape == "square":
        _square_peg(ax, x, 1.79, size=0.075)
    else:
        _round_peg(ax, x, 1.79, radius=0.042)
    _rect(ax, (x - 0.55, 1.10), (0.16, 0.16), facecolor=STAGING, edgecolor="#2a8c7a", linewidth=0.6, zorder=5)
    _robot(ax, x, 1.10, 90.0)
    _label(ax, x + 0.19, 1.06, robot_label, ha="left")
    # Handling robot carries the trimmed peg from the machine front to its Conveyor loading position.
    _flow(ax, (x - 0.08, 1.635), (x - 0.08, 0.63), rad=0.32)


def _draw_conveyor(ax: Axes) -> None:
    _rect(ax, (-2.85, 0.50), (4.20, 0.30), facecolor=BELT, edgecolor=INK, linewidth=0.9, zorder=4)
    roller_x = -4.85
    while roller_x < -0.80:
        ax.plot([roller_x, roller_x], [0.36, 0.64], color="#8b929c", linewidth=0.4, zorder=4.2)
        roller_x += 0.15
    for chevron_x in (-4.55, -3.55, -2.55, -1.55, -1.05):
        ax.plot([chevron_x - 0.04, chevron_x + 0.04, chevron_x - 0.04], [0.565, 0.50, 0.435],
                color="white", linewidth=0.9, solid_joinstyle="miter", zorder=4.5)
    for loading_x in (-4.05, -2.05):
        _rect(ax, (loading_x, 0.50), (0.20, 0.22), facecolor=LOADING, edgecolor="white", linewidth=0.6, zorder=4.6)
    _label(ax, -3.05, 0.27, "Conveyor", va="top")


def _draw_assembly_station(ax: Axes) -> None:
    _rect(ax, (0.0, 0.0), (1.50, 1.60), facecolor=TABLE, edgecolor=TABLE_EDGE, linewidth=0.9, zorder=1)

    zone_edges = (-0.74, -0.62, -0.50, -0.38, -0.26)
    for index, (left, right) in enumerate(zip(zone_edges, zone_edges[1:])):
        _rect(ax, ((left + right) / 2, 0.50), (right - left, 0.24),
              facecolor="#9fbcdc" if index == 3 else "#cfe0f1", edgecolor=STORAGE_EDGE, linewidth=0.6, zorder=5)
    _label(ax, -0.50, 0.98, "Buffer")

    _rect(ax, (0.0, 0.0), (0.384, 0.384), facecolor="white", edgecolor=INK, linewidth=0.9, zorder=5)
    for column, hole_x in enumerate((-0.135, -0.045, 0.045, 0.135)):
        _square_peg(ax, hole_x, 0.11, size=0.05)
        _round_peg(ax, hole_x, 0.025, radius=0.028)
        if column < 3:
            _gear(ax, -0.09 + column * 0.09, -0.10, (0.028, 0.035, 0.042)[column])

    _rect(ax, (0.50, 0.58), (0.24, 0.18), facecolor=STORAGE, edgecolor=STORAGE_EDGE, linewidth=0.7, zorder=5)
    _label(ax, 0.84, 0.58, "Exit", ha="left")

    _rect(ax, (0.50, -0.50), (0.36, 0.46), facecolor="#4b5563", edgecolor=INK, linewidth=0.9, zorder=5)
    _rect(ax, (0.45, -0.50), (0.18, 0.34), facecolor="#a9b0ba", edgecolor="none", zorder=5.5)
    for gear_y, radius in ((-0.58, 0.028), (-0.50, 0.035), (-0.42, 0.042)):
        _gear(ax, 0.44, gear_y, radius)
    _label(ax, 0.84, -0.50, "Printer", ha="left")

    ax.plot([0.20, 0.80], [0.0, 0.0], color=MUTED, linewidth=0.5, zorder=9)
    _label(ax, 0.84, 0.0, "Assembly\nBoard", ha="left")

    _robot(ax, 0.0, 0.50, 180.0)
    _robot(ax, 0.0, -0.50, 0.0)
    _label(ax, 0.10, 0.98, "UR5e-3", ha="left")
    _label(ax, 0.0, -0.86, "UR5e-4", va="top")

    _flow(ax, (-0.30, 0.37), (-0.15, 0.21), rad=0.25)
    _flow(ax, (0.31, -0.40), (0.20, -0.14), rad=-0.25)
    _flow(ax, (0.20, 0.15), (0.37, 0.50), rad=0.30)


def _legend_row(ax: Axes, x: float, y: float, text: str) -> None:
    ax.text(x + 0.34, y, text, ha="left", va="center", fontsize=SMALL_SIZE, color=INK, zorder=20)


def _draw_legend(ax: Axes) -> None:
    left, top, step = -7.02, 1.12, 0.215
    rows = iter(range(10))

    y = top - next(rows) * step
    _flow(ax, (left, y), (left + 0.26, y))
    _legend_row(ax, left, y, "Part transfer")

    y = top - next(rows) * step
    _dashed_route(ax, [(left, y), (left + 0.26, y)])
    _legend_row(ax, left, y, "KMR route")

    y = top - next(rows) * step
    ax.add_patch(FancyBboxPatch((left + 0.06, y - 0.07), 0.14, 0.14, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="none", edgecolor=KMR_ROUTE, linewidth=0.7, linestyle=(0, (2, 1.6))))
    _legend_row(ax, left, y, "KMR docking pose")

    y = top - next(rows) * step
    _rect(ax, (left + 0.13, y), (0.20, 0.13), facecolor=LOADING, edgecolor="none")
    _legend_row(ax, left, y, "Conveyor loading position")

    y = top - next(rows) * step
    _rect(ax, (left + 0.13, y), (0.13, 0.13), facecolor=STAGING, edgecolor="#2a8c7a", linewidth=0.6)
    _legend_row(ax, left, y, "Staging tray")

    y = top - next(rows) * step
    ax.add_patch(Circle((left + 0.13, y), 0.09, facecolor=REACH, alpha=0.12, edgecolor="none"))
    ax.add_patch(Circle((left + 0.13, y), 0.09, facecolor="none", edgecolor=REACH, linewidth=0.5,
                        linestyle=(0, (3, 2))))
    _legend_row(ax, left, y, "UR5e reach (0.85 m)")

    y = top - next(rows) * step
    _square_peg(ax, left + 0.13, y, zorder=20)
    _legend_row(ax, left, y, "KET square peg (×4)")

    y = top - next(rows) * step
    _round_peg(ax, left + 0.13, y, zorder=20)
    _legend_row(ax, left, y, "RGOCG round peg (×4)")

    y = top - next(rows) * step
    _gear(ax, left + 0.13, y, 0.065, zorder=20)
    _legend_row(ax, left, y, "Printed gear (×3)")


def _draw_scale_bar(ax: Axes) -> None:
    x0, y0 = -4.90, -0.80
    ax.plot([x0, x0 + 1.0], [y0, y0], color=INK, linewidth=1.0, solid_capstyle="butt")
    for tick_x in (x0, x0 + 0.5, x0 + 1.0):
        ax.plot([tick_x, tick_x], [y0 - 0.04, y0 + 0.04], color=INK, linewidth=0.7)
    _label(ax, x0 + 0.5, y0 + 0.08, "1 m", bold=False, size=SMALL_SIZE, va="bottom")


def _draw_failures(ax: Axes) -> None:
    _failure_badge(ax, -3.05, 0.50, 1)
    _failure_badge(ax, -3.80, 1.38, 2)
    _failure_badge(ax, -3.79, 2.15, 3)
    # Part slippage: ur5e-4 releases gear_small early; it rests upright in ur5e-3's region.
    _failure_badge(ax, 0.0, 0.287, 4, radius=0.092)

    left, top, step = -1.50, 3.28, 0.235
    _label(ax, left, top, "Failure scenarios", ha="left", size=7.0)
    for number, scenario in enumerate(FAILURE_SCENARIOS, start=1):
        y = top - number * step
        _failure_badge(ax, left + 0.09, y, number, radius=0.085)
        ax.text(left + 0.26, y, scenario, ha="left", va="center", fontsize=SMALL_SIZE, color=INK, zorder=20)


def draw_case_study_setup(show_failures: bool) -> plt.Figure:
    """Draw the case-study cell.

    Args:
        show_failures: Mark the four failure scenarios and list them.

    Returns:
        The finished figure, sized for a double-column journal page.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    width_m = X_LIMITS[1] - X_LIMITS[0]
    height_m = Y_LIMITS[1] - Y_LIMITS[0]
    width_in = FIGURE_WIDTH_MM / 25.4
    figure = plt.figure(figsize=(width_in, width_in * height_m / width_m))
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_storage_and_kmr(ax)
    _draw_machine_cell(ax, "M1", -4.05, "UR5e-1", "square")
    _draw_machine_cell(ax, "M2", -2.05, "UR5e-2", "round")
    _draw_conveyor(ax)
    _draw_assembly_station(ax)
    _draw_legend(ax)
    _draw_scale_bar(ax)
    if show_failures:
        _draw_failures(ax)
    return figure


def main() -> None:
    """Write both figure variants as PDF, SVG, and 600-dpi PNG."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for stem, show_failures in (("case_study_setup", False), ("case_study_setup_failures", True)):
        figure = draw_case_study_setup(show_failures)
        for suffix in ("pdf", "svg", "png"):
            path = OUTPUT_DIR / f"{stem}.{suffix}"
            figure.savefig(path, dpi=600, facecolor="white")
            logger.info("Wrote %s", path)
        plt.close(figure)


if __name__ == "__main__":
    main()
