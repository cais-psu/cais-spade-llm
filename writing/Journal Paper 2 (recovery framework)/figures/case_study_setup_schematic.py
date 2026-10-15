"""Draw the case-study setup and export an editable PowerPoint figure.

Read equipment positions from the configured Gazebo scene. Symbol sizes and
horizontal and vertical spacing are schematic, rather than a scale drawing.
Run with the project's Poetry environment to write SVG, PDF, 600-dpi PNG,
and native PowerPoint objects beside this script.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent
SCENE_FILE = OUTPUT_DIR.parents[2] / "cais_spade_llm/initialization/recovery_framework_gazebo.json"
PPTX_WRITER = OUTPUT_DIR / "cca_editable_v18/pptx_native.py"
STEM = "case_study_setup_schematic"
CAPTION = "Schematic top view of the manufacturing setup used in the case study (not to scale)."
WIDTH_MM = 180.0
HEIGHT_MM = 80.0
INK = "#303943"
GRAY = "#DEE3E8"
BLUE = "#DCE9F3"
BLUE_EDGE = "#66859C"
ORANGE = "#B8783D"
TABLE = "#FCF2E5"
BELT = "#687580"
TEAL = "#7AAEA9"


def _position(pose: Sequence[float]) -> tuple[float, float]:
    return 150.45 + 21.0 * pose[0], 26.5 + 15.0 * pose[1]


def _rectangle(
    ax: Axes, name: str, x: float, y: float, w: float, h: float,
    *, fill: str = GRAY, edge: str = INK, width: float = 0.7, zorder: float = 3,
) -> None:
    ax.add_patch(Rectangle(
        (x - w / 2, y - h / 2), w, h, facecolor=fill, edgecolor=edge,
        linewidth=width, zorder=zorder, label=name,
    ))


def _circle(
    ax: Axes, name: str, x: float, y: float, radius: float,
    *, fill: str = GRAY, edge: str = INK, width: float = 0.6, zorder: float = 5,
) -> None:
    ax.add_patch(Circle(
        (x, y), radius, facecolor=fill, edgecolor=edge, linewidth=width,
        zorder=zorder, label=name,
    ))


def _line(
    ax: Axes, name: str, points: Sequence[tuple[float, float]],
    *, color: str = INK, width: float = 0.65, zorder: float = 4,
) -> None:
    xs, ys = zip(*points)
    ax.plot(
        xs, ys, color=color, linewidth=width, solid_capstyle="round",
        solid_joinstyle="round", zorder=zorder, label=name,
    )


def _label(
    ax: Axes, name: str, x: float, y: float, *, text: str | None = None,
    size: float = 9.0, align: str = "center",
) -> None:
    label = ax.text(
        x, y, text if text is not None else name, fontsize=size,
        fontweight="bold", ha=align, va="center", color=INK,
        linespacing=1.05, zorder=20,
    )
    label.set_label(name)


def _robot(
    ax: Axes, name: str, points: Sequence[tuple[float, float]],
    *, joint_fill: str = BLUE,
) -> None:
    for start, end in zip(points[:-1], points[1:]):
        _line(ax, name, (start, end), width=4.8)
        _line(ax, name, (start, end), color=GRAY, width=3.4, zorder=4.1)
    for index, (x, y) in enumerate(points[:-1]):
        _circle(ax, name, x, y, 2.5 if index == 0 else 1.1, fill=joint_fill)
        _circle(ax, name, x, y, 0.7 if index == 0 else 0.3, fill=INK, edge=INK, width=0)
    wrist, tip = points[-2:]
    angle = math.atan2(tip[1] - wrist[1], tip[0] - wrist[0])
    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy * 0.85, dx * 0.85
    x, y = tip
    _line(ax, name, ((x - nx, y - ny), (x + nx, y + ny)), width=0.9, zorder=6)
    for side in (-1, 1):
        start = x + side * nx, y + side * ny
        end = start[0] + dx * 1.3, start[1] + dy * 1.3
        _line(ax, name, (start, end), width=0.9, zorder=6)


def _draw_storage_and_kmr(ax: Axes, scene: dict[str, Any]) -> None:
    x, y = _position(scene["Storage"]["world_pose"])
    _rectangle(ax, "Storage", x, y, 18, 21, fill=BLUE, edge=BLUE_EDGE)
    for slot_x in (x - 4, x + 4):
        _rectangle(ax, "Storage", slot_x, y, 5, 17.5, fill="white", edge=BLUE_EDGE, width=0.4)
        for slot_y in (y - 6, y - 2, y + 2, y + 6):
            _rectangle(ax, "Storage", slot_x, slot_y, 2.2, 1.4, fill=GRAY, width=0.35)
    _label(ax, "Storage", x, y - 15)

    x, y = _position(scene["KMR"]["initial_pose"])
    _rectangle(ax, "KUKA KMR", x, y, 12, 20, fill=GRAY)
    _rectangle(ax, "KUKA KMR", x, y + 4.1, 8.5, 6.2, fill=BLUE, edge=BLUE_EDGE, width=0.45)
    for wheel_x in (x - 5.5, x + 5.5):
        for wheel_y in (y - 7, y + 7):
            _rectangle(ax, "KUKA KMR", wheel_x, wheel_y, 1.25, 3.2, fill=INK, width=0)
    _line(ax, "KUKA KMR", ((x - 5.1, y - 9), (x + 5.1, y - 9)), color=ORANGE, width=1.4)
    _robot(ax, "KUKA KMR", (
        (x, y - 3), (x - 3.6, y - 5.1), (x - 6.5, y - 2.8), (x - 8.3, y - 2.8),
    ), joint_fill="#E6BB92")
    _label(ax, "KUKA KMR", x, y + 14)


def _draw_machine(ax: Axes, machine: dict[str, Any], robot: dict[str, Any], label: str) -> None:
    name = machine["resource_id"]
    x, y = _position(machine["world_pose"])
    _rectangle(ax, name, x, y, 13.5, 13, fill=GRAY)
    _rectangle(ax, name, x, y + 0.7, 9.3, 8.6, fill="white", width=0.4)
    _rectangle(ax, name, x, y + 3.3, 2.5, 2.5, fill=GRAY, width=0.4)
    _rectangle(ax, name, x, y - 1, 5.1, 2.6, fill=GRAY, width=0.4)
    _line(ax, name, ((x - 3.9, y - 5.5), (x + 3.9, y - 5.5)), color=BLUE_EDGE, width=0.75)
    _label(ax, name, x, y + 11.2, size=9.5)

    x, y = _position(robot["base_xyz"])
    _rectangle(ax, label, x, y, 7.5, 5.7, fill=BLUE, edge=BLUE_EDGE, width=0.45)
    _robot(ax, label, ((x, y), (x - 3.6, y + 2.7), (x - 4.9, y - 0.2), (x - 6.7, y - 1.2)))
    _label(ax, label, x + 6.1, y + 0.2, size=8.7, align="left")


def _draw_conveyor_and_buffer(ax: Axes, scene: dict[str, Any]) -> None:
    x, y = _position(scene["Conveyor"]["world_pose"])
    length = scene["Conveyor"]["length"] * 21
    left, right = x - length / 2, x + length / 2
    _rectangle(ax, "Conveyor", x, y, length, 5.4, fill=BELT)
    for index in range(1, 26):
        roller_x = left + length * index / 26
        _line(ax, "Conveyor", ((roller_x, y - 2.2), (roller_x, y + 2.2)), color="#94A0A9", width=0.35)
    for rail_y in (y - 2.65, y + 2.65):
        _line(ax, "Conveyor", ((left, rail_y), (right, rail_y)), color=ORANGE, width=1.0)
    for machine in scene["machines"]:
        loading_x, loading_y = _position(machine["conveyor_loading_pose"])
        _rectangle(ax, "Conveyor", loading_x, loading_y, 4.1, 3.7, fill=TEAL, edge="white", width=0.45, zorder=6)
    _label(ax, "Conveyor", x, y - 8.8, size=9.5)

    buffer_x, buffer_y = _position(scene["Buffer For Machined parts"]["world_pose"])
    buffer_right = buffer_x + 0.24 * 21
    buffer_width = buffer_right - right
    # Join the equipment symbols so a drawing gap cannot suggest a broken belt.
    _rectangle(ax, "Buffer", (right + buffer_right) / 2, buffer_y, buffer_width, 5.4, fill=BLUE)
    for slot in range(1, 4):
        slot_x = right + slot * buffer_width / 4
        _line(ax, "Buffer", ((slot_x, buffer_y - 2.5), (slot_x, buffer_y + 2.5)), color=BLUE_EDGE, width=0.5)
    _line(ax, "Buffer", ((buffer_x, buffer_y + 3.5), (buffer_x, buffer_y + 13.8)), color=BLUE_EDGE, width=0.5)
    _label(ax, "Buffer", buffer_x, buffer_y + 17)


def _draw_assembly(ax: Axes, scene: dict[str, Any]) -> None:
    robots = {robot["resource_id"]: robot for robot in scene["robots"]}
    upper_x, upper_y = _position(robots["ur5e-3"]["base_xyz"])
    lower_x, lower_y = _position(robots["ur5e-4"]["base_xyz"])
    board_x, board_y = (upper_x + lower_x) / 2, (upper_y + lower_y) / 2
    _rectangle(ax, "", board_x + 0.75, board_y, 33, 28, fill=TABLE, edge=ORANGE, zorder=0)

    _robot(ax, "UR5e-3", (
        (upper_x, upper_y), (upper_x + 3.1, upper_y - 2.2),
        (upper_x + 4.9, upper_y - 0.5), (upper_x + 4.9, upper_y + 1),
    ))
    _label(ax, "UR5e-3", upper_x, upper_y + 9.6, size=8.7)
    _robot(ax, "UR5e-4", (
        (lower_x, lower_y), (lower_x + 2.8, lower_y - 2.1),
        (lower_x + 5.1, lower_y - 0.8), (lower_x + 5.1, lower_y + 0.9),
    ))
    _label(ax, "UR5e-4", lower_x, lower_y - 9.6, size=8.7)

    _rectangle(ax, "Assembly Board", board_x, board_y, 10.5, 8.5, fill="white")
    for offset in (-3.5, -1.2, 1.2, 3.5):
        _rectangle(ax, "Assembly Board", board_x + offset, board_y + 2.4, 1.1, 1.1, fill="white", width=0.35)
        _circle(ax, "Assembly Board", board_x + offset, board_y + 0.2, 0.55, fill="white", width=0.35)
    for offset, radius in ((-2.8, 0.5), (0, 0.7), (2.8, 0.9)):
        _circle(ax, "Assembly Board", board_x + offset, board_y - 2.3, radius, fill="white", width=0.35)
    _line(ax, "Assembly Board", (
        (board_x - 13.6, board_y - 1), (board_x - 9.7, board_y), (board_x - 5.9, board_y),
    ), color=BLUE_EDGE, width=0.5)
    _label(ax, "Assembly Board", board_x - 22, board_y - 3.4, text="Assembly\nBoard", size=8.7)

    exit_x, exit_y = _position(scene["Exit"]["world_pose"])
    _rectangle(ax, "Exit", exit_x + 1.2, exit_y, 7.4, 7.2, fill=BLUE, edge=BLUE_EDGE)
    _rectangle(ax, "Exit", exit_x + 1.2, exit_y, 5.7, 5.5, fill="white", edge=BLUE_EDGE, width=0.35)
    _label(ax, "Exit", exit_x + 2.3, upper_y + 9.6, size=8.7)

    printer_x, printer_y = _position(scene["3D Printing Station"]["world_pose"])
    printer_x += 1.5
    _rectangle(ax, "Printer", printer_x, printer_y, 8.7, 10.5, fill=GRAY)
    _rectangle(ax, "Printer", printer_x, printer_y - 0.5, 6.5, 7.3, fill="white", width=0.4)
    _line(ax, "Printer", ((printer_x - 3.1, printer_y + 2), (printer_x + 3.1, printer_y + 2)), width=0.8)
    _rectangle(ax, "Printer", printer_x - 0.4, printer_y + 1.4, 1.9, 1.7, fill=GRAY, width=0.4)
    _rectangle(ax, "Printer", printer_x, printer_y - 2.2, 4.5, 2.1, fill=BLUE, edge=BLUE_EDGE, width=0.35)
    _label(ax, "Printer", printer_x + 0.8, lower_y - 9.6, size=8.7)


def draw_case_study_setup_schematic() -> Figure:
    """Draw all thirteen supplied labels and their equipment symbols.

    Returns:
        A 180 by 80 mm figure with equipment centers grounded in the scene.
    """
    scene = json.loads(SCENE_FILE.read_text(encoding="utf-8"))
    figure = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=144)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, HEIGHT_MM)
    ax.set_aspect("equal")
    ax.axis("off")
    robots = {robot["resource_id"]: robot for robot in scene["robots"]}
    _draw_storage_and_kmr(ax, scene)
    for machine, label in zip(scene["machines"], ("UR5e-1", "UR5e-2"), strict=True):
        _draw_machine(ax, machine, robots[machine["handling_robot"]], label)
    _draw_conveyor_and_buffer(ax, scene)
    _draw_assembly(ax, scene)
    return figure


def _powerpoint_scene(figure: Figure) -> dict[str, Any]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    factor = 100 / 25.4
    groups: dict[str, list[dict[str, Any]]] = {}
    for artist in sorted(figure.axes[0].get_children(), key=lambda item: item.get_zorder()):
        if not isinstance(artist, (Rectangle, Circle, Line2D, Text)):
            continue
        if isinstance(artist, Text) and not artist.get_text():
            continue
        name = artist.get_label()
        if name.startswith("_") or artist is figure.axes[0].patch:
            continue
        shape: dict[str, Any]
        if isinstance(artist, (Circle, Rectangle)):
            if isinstance(artist, Circle):
                x, y = artist.center
                w = h = 2 * artist.radius
                x, y = x - w / 2, y - h / 2
                kind = "ellipse"
            else:
                x, y = artist.get_xy()
                w, h = artist.get_width(), artist.get_height()
                kind = "rect"
            shape = {
                "kind": kind, "x": x * factor, "y": (HEIGHT_MM - y - h) * factor,
                "w": w * factor, "h": h * factor,
                "fill": to_hex(artist.get_facecolor()), "stroke": to_hex(artist.get_edgecolor()),
                "stroke_width": artist.get_linewidth() * 100 / 72,
            }
        elif isinstance(artist, Line2D):
            points = [(float(x) * factor, (HEIGHT_MM - float(y)) * factor)
                      for x, y in zip(artist.get_xdata(), artist.get_ydata(), strict=True)]
            shape = {
                "kind": "path", "commands": [["M", *points[0]], *[["L", *point] for point in points[1:]]],
                "fill": None, "stroke": to_hex(artist.get_color()),
                "stroke_width": artist.get_linewidth() * 100 / 72,
            }
        else:
            bounds = artist.get_window_extent(renderer=renderer)
            scale = 100 / figure.dpi
            font_size = artist.get_fontsize() * 100 / 72
            # Office paragraph metrics need more room than Matplotlib glyph bounds.
            frame_height = max(bounds.height * scale + 4, font_size * 1.4 * len(artist.get_text().splitlines()))
            center_y = (figure.bbox.height - (bounds.y0 + bounds.y1) / 2) * scale
            shape = {
                "kind": "text", "x": bounds.x0 * scale - 2,
                "y": center_y - frame_height / 2,
                "w": bounds.width * scale + 4, "h": frame_height,
                "text": artist.get_text(), "size": font_size,
                "font": "Arial", "bold": True, "color": INK,
                "align": "center", "valign": "center", "line_spacing": 1.05,
                "fill": None, "stroke": None,
            }
        groups.setdefault(name, []).append(shape)
    return {
        "width": WIDTH_MM * factor, "height": HEIGHT_MM * factor,
        "title": "Case-study setup schematic",
        "groups": [{"name": name, "shapes": shapes} for name, shapes in groups.items()],
    }


def main() -> None:
    """Write matching paper figure exports and an editable PowerPoint slide."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    with matplotlib.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": STEM,
    }):
        figure = draw_case_study_setup_schematic()
        for suffix in ("svg", "pdf", "png"):
            metadata = None
            if suffix == "svg":
                metadata = {"Title": "Case-study setup schematic", "Description": CAPTION, "Date": None}
            elif suffix == "pdf":
                metadata = {"Title": "Case-study setup schematic", "Subject": CAPTION, "CreationDate": None, "ModDate": None}
            destination = OUTPUT_DIR / f"{STEM}.{suffix}"
            figure.savefig(destination, dpi=600, facecolor="white", metadata=metadata)
            logger.info("Wrote %s", destination)
        scene = _powerpoint_scene(figure)
        spec = importlib.util.spec_from_file_location("_case_study_pptx_native", PPTX_WRITER)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the native PowerPoint writer: {PPTX_WRITER}")
        writer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(writer)
        destination = OUTPUT_DIR / f"{STEM}.pptx"
        logger.info("Wrote %s: %s", destination, writer.write_pptx(scene, destination))
        (OUTPUT_DIR / f"{STEM}_scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
        (OUTPUT_DIR / f"{STEM}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
        plt.close(figure)


if __name__ == "__main__":
    main()
