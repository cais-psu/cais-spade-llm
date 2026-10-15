"""Draw the case-study material flow as editable blocks and handling arrows.

Use the screenshot's exact equipment and robot names. Export matching PDF,
SVG, 600-dpi PNG, and a PowerPoint slide with native connected shapes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent
STEM = "case_study_flow_block_diagram"
CAPTION = (
    "Flow block diagram of the manufacturing setup used in the case study. "
    "Arrows indicate material flow. UR5e-3 also handles the transfer from "
    "Assembly Board to Exit."
)
WIDTH_MM = 180.0
HEIGHT_MM = 80.0
FACTOR = 100 / 25.4
INK = "#334350"
ARROW = "#47657D"
BLUE = "#E6EFF6"
GRAY = "#F0F2F4"
ORANGE = "#FBF0E2"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def _block(name: str, x: float, y: float, w: float, h: float, fill: str) -> dict[str, Any]:
    bounds = {
        "x": (x - w / 2) * FACTOR, "y": (HEIGHT_MM - y - h / 2) * FACTOR,
        "w": w * FACTOR, "h": h * FACTOR,
    }
    return {
        "name": name,
        "shapes": [
            {"kind": "rect", **bounds, "fill": fill, "stroke": INK, "stroke_width": 0.85 * 100 / 72},
            {
                "kind": "text", **bounds, "text": "Assembly\nBoard" if name == "Assembly Board" else name,
                "size": 9.0 * 100 / 72, "font": "Arial", "bold": True, "color": INK,
                "align": "center", "valign": "center", "line_spacing": 1.05,
                "fill": None, "stroke": None,
            },
        ],
    }


def _handling_label(name: str, x: float, y: float, w: float) -> dict[str, Any]:
    return {
        "name": name,
        "shapes": [{
            "kind": "text", "x": (x - w / 2) * FACTOR,
            "y": (HEIGHT_MM - y - 2.5) * FACTOR, "w": w * FACTOR, "h": 5 * FACTOR,
            "text": name, "size": 8.5 * 100 / 72, "font": "Arial", "bold": True,
            "color": ARROW, "align": "center", "valign": "center",
            "fill": None, "stroke": None,
        }],
    }


def _connector(
    source: str, target: str, start: tuple[float, float], end: tuple[float, float],
    *, start_site: int = 3, end_site: int = 1, middle_x: float | None = None,
) -> dict[str, Any]:
    result = {
        "kind": "line", "name": f"{source} → {target}",
        "x1": start[0] * FACTOR, "y1": (HEIGHT_MM - start[1]) * FACTOR,
        "x2": end[0] * FACTOR, "y2": (HEIGHT_MM - end[1]) * FACTOR,
        "stroke": ARROW, "stroke_width": 0.9 * 100 / 72, "arrow": True,
        "source": source, "target": target, "start_site": start_site, "end_site": end_site,
    }
    if middle_x is not None:
        result["middle_x"] = middle_x * FACTOR
    return result


def build_scene() -> dict[str, Any]:
    """Build the parallel M1/M2 flow and the Printer input to Assembly Board.

    Returns:
        A scene containing eight equipment blocks, five handling labels,
        and eight material-flow connectors.
    """
    links = [
        _connector("Storage", "M1", (22, 46), (58, 62), middle_x=51),
        _connector("Storage", "M2", (22, 46), (58, 30), middle_x=51),
        _connector("M1", "Conveyor", (78, 62), (103, 46), middle_x=99),
        _connector("M2", "Conveyor", (78, 30), (103, 46), middle_x=99),
        _connector("Conveyor", "Buffer", (125, 46), (131, 46)),
        _connector("Buffer", "Assembly Board", (141, 40.5), (141, 22), start_site=2, end_site=0),
        _connector("Printer", "Assembly Board", (102.5, 15), (129, 15)),
        _connector("Assembly Board", "Exit", (153, 15), (163, 15)),
    ]
    return {
        "width": WIDTH_MM * FACTOR, "height": HEIGHT_MM * FACTOR,
        "title": "Case-study flow block diagram",
        "groups": [
            {"name": "", "shapes": links},
            _block("Storage", 12, 46, 20, 11, BLUE),
            _block("M1", 68, 62, 20, 11, GRAY),
            _block("M2", 68, 30, 20, 11, GRAY),
            _block("Conveyor", 114, 46, 22, 11, BLUE),
            _block("Buffer", 141, 46, 20, 11, BLUE),
            _block("Assembly Board", 141, 15, 24, 14, ORANGE),
            _block("Printer", 92, 15, 21, 11, GRAY),
            _block("Exit", 170, 15, 14, 11, BLUE),
            _handling_label("KUKA KMR", 36, 51.5, 25),
            _handling_label("UR5e-1", 88, 67.5, 19),
            _handling_label("UR5e-2", 88, 24.5, 19),
            _handling_label("UR5e-3", 154, 31.5, 19),
            _handling_label("UR5e-4", 116, 20.8, 19),
        ],
    }


def draw_figure(scene: dict[str, Any]) -> Figure:
    """Render the editable scene using the paper figure's physical size.

    Args:
        scene: Equipment blocks, handling labels, and material-flow links.

    Returns:
        The finished 180 by 80 mm Matplotlib figure.
    """
    figure = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=144)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, scene["width"])
    ax.set_ylim(scene["height"], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for group in scene["groups"]:
        for shape in group["shapes"]:
            if shape["kind"] == "rect":
                ax.add_patch(Rectangle(
                    (shape["x"], shape["y"]), shape["w"], shape["h"],
                    facecolor=shape["fill"], edgecolor=shape["stroke"],
                    linewidth=shape["stroke_width"] * 72 / 100, zorder=3,
                ))
            elif shape["kind"] == "text":
                label = ax.text(
                    shape["x"] + shape["w"] / 2, shape["y"] + shape["h"] / 2,
                    shape["text"], fontsize=shape["size"] * 72 / 100,
                    fontweight="bold", color=shape["color"],
                    ha="center", va="center", linespacing=1.05, zorder=4,
                )
                label.set_label(group["name"])
            else:
                start = (shape["x1"], shape["y1"])
                end = (shape["x2"], shape["y2"])
                points = [start, end]
                if "middle_x" in shape:
                    points = [start, (shape["middle_x"], start[1]), (shape["middle_x"], end[1]), end]
                    ax.plot(
                        [point[0] for point in points[:-1]], [point[1] for point in points[:-1]],
                        color=ARROW, linewidth=0.9, solid_capstyle="butt", zorder=1,
                    )
                ax.add_patch(FancyArrowPatch(
                    points[-2], points[-1], arrowstyle="-|>", mutation_scale=8,
                    linewidth=0.9, color=ARROW, shrinkA=0, shrinkB=0, zorder=2,
                ))
    return figure


def _connect_powerpoint_shapes(path: Path, scene: dict[str, Any]) -> None:
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    slide = ET.fromstring(parts["ppt/slides/slide1.xml"])
    tree = slide.find("p:cSld/p:spTree", NS)
    if tree is None:
        raise ValueError("PowerPoint slide has no shape tree")
    block_ids = {}
    connectors = {}
    for group in list(tree.findall("p:grpSp", NS)):
        name = group.find("p:nvGrpSpPr/p:cNvPr", NS).get("name")
        shapes = group.findall("p:sp", NS)
        if len(shapes) == 2:
            block, label = shapes
            block.append(copy.deepcopy(label.find("p:txBody", NS)))
            properties = block.find("p:nvSpPr/p:cNvPr", NS)
            properties.set("name", name)
            block_ids[name] = properties.get("id")
            tree.append(block)
        else:
            for shape in shapes:
                shape.find("p:nvSpPr/p:cNvPr", NS).set("name", name)
                tree.append(shape)
        for connector in group.findall("p:cxnSp", NS):
            name = connector.find("p:nvCxnSpPr/p:cNvPr", NS).get("name")
            connectors[name] = connector
            tree.insert(2, connector)
        tree.remove(group)
    for link in scene["groups"][0]["shapes"]:
        connector = connectors[link["name"]]
        properties = connector.find("p:nvCxnSpPr/p:cNvCxnSpPr", NS)
        ET.SubElement(properties, f"{{{NS['a']}}}stCxn", id=block_ids[link["source"]], idx=str(link["start_site"]))
        ET.SubElement(properties, f"{{{NS['a']}}}endCxn", id=block_ids[link["target"]], idx=str(link["end_site"]))
        if "middle_x" in link:
            geometry = connector.find("p:spPr/a:prstGeom", NS)
            geometry.set("prst", "bentConnector3")
            adjustments = geometry.find("a:avLst", NS)
            fraction = (link["middle_x"] - link["x1"]) / (link["x2"] - link["x1"])
            ET.SubElement(adjustments, f"{{{NS['a']}}}gd", name="adj1", fmla=f"val {round(fraction * 100000)}")
    parts["ppt/slides/slide1.xml"] = ET.tostring(slide, encoding="utf-8", xml_declaration=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


def main() -> None:
    """Export the flow block diagram with editable connected PowerPoint blocks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    scene = build_scene()
    with matplotlib.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": STEM,
    }):
        figure = draw_figure(scene)
        for suffix in ("svg", "pdf", "png"):
            metadata = None
            if suffix == "svg":
                metadata = {"Title": scene["title"], "Description": CAPTION, "Date": None}
            elif suffix == "pdf":
                metadata = {"Title": scene["title"], "Subject": CAPTION, "CreationDate": None, "ModDate": None}
            destination = OUTPUT_DIR / f"{STEM}.{suffix}"
            figure.savefig(destination, dpi=600, facecolor="white", metadata=metadata)
            logger.info("Wrote %s", destination)
        plt.close(figure)
    spec = importlib.util.spec_from_file_location("_case_study_flow_pptx_native", OUTPUT_DIR / "cca_editable_v18/pptx_native.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the native PowerPoint writer")
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    destination = OUTPUT_DIR / f"{STEM}.pptx"
    writer.write_pptx(scene, destination)
    _connect_powerpoint_shapes(destination, scene)
    logger.info("Wrote %s: %s", destination, writer.validate_pptx(destination))
    (OUTPUT_DIR / f"{STEM}_scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / f"{STEM}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
