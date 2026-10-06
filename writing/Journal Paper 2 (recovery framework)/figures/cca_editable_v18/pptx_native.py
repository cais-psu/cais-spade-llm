"""Export an editable native PowerPoint slide and Cairo preview from a scene.

The scene uses 100 logical pixels per inch. All PowerPoint objects are native
shapes, connectors, groups, or text; no raster images are embedded.
"""

from __future__ import annotations

import argparse
import json
import logging
from math import atan2, cos, sin, pi
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


logger = logging.getLogger(__name__)
EMU = 9144
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def _el(tag: str, parent: ET.Element | None = None, **attrs: Any) -> ET.Element:
    prefix, name = tag.split(":")
    element = ET.Element(f"{{{NS[prefix]}}}{name}", {key: str(value) for key, value in attrs.items()})
    if parent is not None:
        parent.append(element)
    return element


def _xml(element: ET.Element) -> bytes:
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def _emu(value: float) -> int:
    return round(value * EMU)


def _color(value: str | None, default: str = "000000") -> str:
    return (value or default).lstrip("#").upper()


def _fill(parent: ET.Element, value: str | None, opacity: float = 1) -> None:
    if value is None:
        _el("a:noFill", parent)
        return
    solid = _el("a:solidFill", parent)
    rgb = _el("a:srgbClr", solid, val=_color(value))
    if opacity < 1:
        _el("a:alpha", rgb, val=round(max(0, opacity) * 100000))


def _line_style(parent: ET.Element, shape: dict) -> None:
    line = _el("a:ln", parent, w=_emu(shape.get("stroke_width", 1)))
    _fill(line, shape.get("stroke"))
    if shape.get("dash"):
        _el("a:prstDash", line, val="dash")
    _el("a:round", line)
    if shape.get("arrow"):
        _el("a:tailEnd", line, type="triangle", w="med", len="med")


def _bounds(shape: dict) -> tuple[float, float, float, float]:
    if shape.get("kind", shape.get("type")) == "line":
        return (min(shape["x1"], shape["x2"]), min(shape["y1"], shape["y2"]),
                abs(shape["x2"] - shape["x1"]), abs(shape["y2"] - shape["y1"]))
    if shape.get("kind", shape.get("type")) == "path":
        points = [(row[index], row[index + 1]) for row in shape["commands"]
                  for index in range(1, len(row), 2)]
        if not points:
            raise ValueError("A path must contain coordinates")
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return shape["x"], shape["y"], shape["w"], shape["h"]


def _transform(parent: ET.Element, shape: dict) -> None:
    x, y, w, h = _bounds(shape)
    attrs = {}
    if shape.get("kind", shape.get("type")) == "line":
        if shape["x2"] < shape["x1"]:
            attrs["flipH"] = "1"
        if shape["y2"] < shape["y1"]:
            attrs["flipV"] = "1"
    transform = _el("a:xfrm", parent, **attrs)
    _el("a:off", transform, x=_emu(x), y=_emu(y))
    _el("a:ext", transform, cx=max(1, _emu(w)), cy=max(1, _emu(h)))


def _custom_geometry(parent: ET.Element, shape: dict) -> None:
    x, y, w, h = _bounds(shape)
    geometry = _el("a:custGeom", parent)
    for name in ("avLst", "gdLst", "ahLst", "cxnLst"):
        _el(f"a:{name}", geometry)
    _el("a:rect", geometry, l="0", t="0", r="r", b="b")
    path = _el("a:path", _el("a:pathLst", geometry), w=max(1, _emu(w)), h=max(1, _emu(h)))
    for row in shape["commands"]:
        op = row[0]
        if op == "Z":
            _el("a:close", path)
            continue
        tags = {"M": "moveTo", "L": "lnTo", "C": "cubicBezTo"}
        if op not in tags:
            raise ValueError(f"Unsupported path command {op!r}")
        command = _el("a:" + tags[op], path)
        for index in range(1, len(row), 2):
            _el("a:pt", command, x=_emu(row[index] - x), y=_emu(row[index + 1] - y))


def _text_lines(shape: dict) -> list[list[dict]]:
    if "lines" in shape:
        return [[({"text": run} if isinstance(run, str) else run) for run in line]
                for line in shape["lines"]]
    return [[{"text": line}] for line in shape.get("text", "").split("\n")]


def _text_body(parent: ET.Element, shape: dict) -> None:
    body = _el("p:txBody", parent)
    properties = _el("a:bodyPr", body, wrap="square", lIns="0", tIns="0", rIns="0", bIns="0",
                     anchor="ctr" if shape.get("valign", "center") == "center" else "t")
    _el("a:noAutofit", properties)
    _el("a:lstStyle", body)
    for runs in _text_lines(shape):
        paragraph = _el("a:p", body)
        properties = _el("a:pPr", paragraph, algn={"left": "l", "center": "ctr", "right": "r"}.get(shape.get("align", "left"), "l"))
        spacing = _el("a:lnSpc", properties)
        _el("a:spcPct", spacing, val=round(shape.get("line_spacing", 1.1) * 100000))
        for run in runs:
            merged = {**shape, **run}
            row = _el("a:r", paragraph)
            attrs = {"lang": "en-US", "sz": round(merged.get("size", 20) * 72),
                     "b": int(merged.get("bold", False)), "i": int(merged.get("italic", False))}
            if "baseline" in merged:
                attrs["baseline"] = round(merged["baseline"] * 100000)
            properties = _el("a:rPr", row, **attrs)
            _fill(properties, merged.get("color", "20242B"))
            for kind in ("latin", "ea", "cs"):
                _el(f"a:{kind}", properties, typeface=merged.get("font", "Arial"))
            _el("a:t", row).text = str(run.get("text", ""))
        _el("a:endParaRPr", paragraph, lang="en-US", sz=round(shape.get("size", 20) * 72))


def _shape(parent: ET.Element, shape: dict, identifier: int) -> None:
    kind = shape.get("kind", shape.get("type"))
    connector = kind == "line"
    root = _el("p:cxnSp" if connector else "p:sp", parent)
    nonvisual = _el("p:nvCxnSpPr" if connector else "p:nvSpPr", root)
    _el("p:cNvPr", nonvisual, id=identifier, name=shape.get("name", f"{kind} {identifier}"))
    _el("p:cNvCxnSpPr" if connector else "p:cNvSpPr", nonvisual, **({"txBox": "1"} if kind == "text" else {}))
    _el("p:nvPr", nonvisual)
    properties = _el("p:spPr", root)
    _transform(properties, shape)
    if kind == "path":
        _custom_geometry(properties, shape)
    else:
        geometry = _el("a:prstGeom", properties, prst="rect" if kind == "text" else kind)
        adjustments = _el("a:avLst", geometry)
        if kind == "roundRect":
            _, _, width, height = _bounds(shape)
            radius = max(0, min(shape.get("radius", 8), min(width, height) / 2))
            adjustment = round(radius / max(1, min(width, height)) * 100000)
            _el("a:gd", adjustments, name="adj", fmla=f"val {adjustment}")
    _fill(properties, shape.get("fill"), shape.get("opacity", 1))
    _line_style(properties, shape)
    if kind == "text":
        _text_body(root, shape)


def _group(parent: ET.Element, name: str, identifier: int, shapes: list[dict]) -> ET.Element:
    bounds = [_bounds(shape) for shape in shapes]
    allowance = max((shape.get("stroke_width", 0) / 2 for shape in shapes), default=0) + 1
    x = min((row[0] for row in bounds), default=0) - allowance
    y = min((row[1] for row in bounds), default=0) - allowance
    width = max((row[0] + row[2] for row in bounds), default=1) + allowance - x
    height = max((row[1] + row[3] for row in bounds), default=1) + allowance - y
    group = _el("p:grpSp", parent)
    nonvisual = _el("p:nvGrpSpPr", group)
    _el("p:cNvPr", nonvisual, id=identifier, name=name)
    _el("p:cNvGrpSpPr", nonvisual)
    _el("p:nvPr", nonvisual)
    transform = _el("a:xfrm", _el("p:grpSpPr", group))
    _el("a:off", transform, x=_emu(x), y=_emu(y))
    _el("a:ext", transform, cx=_emu(width), cy=_emu(height))
    _el("a:chOff", transform, x=_emu(x), y=_emu(y))
    _el("a:chExt", transform, cx=_emu(width), cy=_emu(height))
    return group


def _tree(parent: ET.Element) -> ET.Element:
    tree = _el("p:spTree", parent)
    nonvisual = _el("p:nvGrpSpPr", tree)
    _el("p:cNvPr", nonvisual, id="1", name="")
    _el("p:cNvGrpSpPr", nonvisual)
    _el("p:nvPr", nonvisual)
    _el("p:grpSpPr", tree)
    return tree


def _relationships(items: list[tuple[str, str, str]]) -> bytes:
    root = ET.Element("Relationships", xmlns="http://schemas.openxmlformats.org/package/2006/relationships")
    for identifier, kind, target in items:
        ET.SubElement(root, "Relationship", Id=identifier, Type=NS["r"] + "/" + kind, Target=target)
    return _xml(root)


def _theme() -> bytes:
    root = _el("a:theme", name="CCA native editable figure")
    elements = _el("a:themeElements", root)
    colors = _el("a:clrScheme", elements, name="CCA")
    palette = {"dk1": "20242B", "lt1": "FFFFFF", "dk2": "394452", "lt2": "F4F6F8",
               "accent1": "6B21D9", "accent2": "2775C9", "accent3": "6B7280", "accent4": "DB3038",
               "accent5": "159447", "accent6": "EF9B22", "hlink": "0563C1", "folHlink": "954F72"}
    for name, value in palette.items():
        _el("a:srgbClr", _el("a:" + name, colors), val=value)
    fonts = _el("a:fontScheme", elements, name="Arial")
    for name in ("majorFont", "minorFont"):
        font = _el("a:" + name, fonts)
        for kind in ("latin", "ea", "cs"):
            _el("a:" + kind, font, typeface="Arial")
    styles = _el("a:fmtScheme", elements, name="CCA")
    fills = _el("a:fillStyleLst", styles)
    for _ in range(3):
        _el("a:schemeClr", _el("a:solidFill", fills), val="phClr")
    lines = _el("a:lnStyleLst", styles)
    for width in (9525, 19050, 28575):
        line = _el("a:ln", lines, w=width, cap="flat", cmpd="sng", algn="ctr")
        _el("a:schemeClr", _el("a:solidFill", line), val="phClr")
        _el("a:prstDash", line, val="solid")
    effects = _el("a:effectStyleLst", styles)
    for _ in range(3):
        _el("a:effectLst", _el("a:effectStyle", effects))
    backgrounds = _el("a:bgFillStyleLst", styles)
    for _ in range(3):
        _el("a:schemeClr", _el("a:solidFill", backgrounds), val="phClr")
    return _xml(root)


def write_pptx(scene: dict, output: Path) -> dict:
    """Write one portrait slide with editable groups, shapes, and text.

    Args:
        scene: Scene containing width, height, title, and named shape groups.
        output: Destination PowerPoint path.

    Returns:
        Validation counts for the generated native objects.
    """
    width, height = scene["width"], scene["height"]
    slide = _el("p:sld")
    common = _el("p:cSld", slide, name=scene.get("title", "CCA safety monitor"))
    tree = _tree(common)
    identifier = 2
    for record in scene["groups"]:
        group = _group(tree, record["name"], identifier, record["shapes"])
        identifier += 1
        for shape in record["shapes"]:
            _shape(group, shape, identifier)
            identifier += 1
    _el("a:masterClrMapping", _el("p:clrMapOvr", slide))
    presentation = _el("p:presentation", saveSubsetFonts="1")
    _el("p:sldMasterId", _el("p:sldMasterIdLst", presentation), id="2147483648", **{f"{{{NS['r']}}}id": "rId1"})
    _el("p:sldId", _el("p:sldIdLst", presentation), id="256", **{f"{{{NS['r']}}}id": "rId2"})
    _el("p:sldSz", presentation, cx=_emu(width), cy=_emu(height), type="custom")
    _el("p:notesSz", presentation, cx="6858000", cy="9144000")
    master = _el("p:sldMaster")
    _tree(_el("p:cSld", master))
    _el("p:clrMap", master, **{key: value for key, value in {
        "accent1": "accent1", "accent2": "accent2", "accent3": "accent3", "accent4": "accent4",
        "accent5": "accent5", "accent6": "accent6", "bg1": "lt1", "bg2": "lt2", "folHlink": "folHlink",
        "hlink": "hlink", "tx1": "dk1", "tx2": "dk2"}.items()})
    _el("p:sldLayoutId", _el("p:sldLayoutIdLst", master), id="2147483649", **{f"{{{NS['r']}}}id": "rId1"})
    styles = _el("p:txStyles", master)
    for kind in ("titleStyle", "bodyStyle", "otherStyle"):
        _el("p:" + kind, styles)
    layout = _el("p:sldLayout", type="blank", preserve="1")
    _tree(_el("p:cSld", layout, name="Blank"))
    _el("a:masterClrMapping", _el("p:clrMapOvr", layout))
    parts = {
        "ppt/presentation.xml": _xml(presentation), "ppt/slides/slide1.xml": _xml(slide),
        "ppt/slideMasters/slideMaster1.xml": _xml(master), "ppt/slideLayouts/slideLayout1.xml": _xml(layout),
        "ppt/theme/theme1.xml": _theme(),
        "_rels/.rels": _relationships([("rId1", "officeDocument", "ppt/presentation.xml")]),
        "ppt/_rels/presentation.xml.rels": _relationships([
            ("rId1", "slideMaster", "slideMasters/slideMaster1.xml"), ("rId2", "slide", "slides/slide1.xml")]),
        "ppt/slides/_rels/slide1.xml.rels": _relationships([("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml")]),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": _relationships([
            ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", "theme", "../theme/theme1.xml")]),
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": _relationships([("rId1", "slideMaster", "../slideMasters/slideMaster1.xml")]),
    }
    types = ET.Element("Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types")
    ET.SubElement(types, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(types, "Default", Extension="xml", ContentType="application/xml")
    for path, kind in [("ppt/presentation.xml", "presentation.main"), ("ppt/slides/slide1.xml", "slide"),
                       ("ppt/slideMasters/slideMaster1.xml", "slideMaster"), ("ppt/slideLayouts/slideLayout1.xml", "slideLayout")]:
        ET.SubElement(types, "Override", PartName="/" + path, ContentType=f"application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml")
    ET.SubElement(types, "Override", PartName="/ppt/theme/theme1.xml", ContentType="application/vnd.openxmlformats-officedocument.theme+xml")
    parts["[Content_Types].xml"] = _xml(types)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return validate_pptx(output)


def validate_pptx(path: Path) -> dict:
    """Check package XML, internal relationships, object IDs, and native content."""
    import posixpath

    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("PowerPoint ZIP integrity check failed")
        names = set(archive.namelist())
        for name in names:
            element = ET.fromstring(archive.read(name))
            if name.endswith(".rels"):
                base = "" if name == "_rels/.rels" else name.rsplit("/_rels/", 1)[0]
                for relationship in element:
                    target = posixpath.normpath(posixpath.join(base, relationship.attrib["Target"]))
                    if target not in names:
                        raise ValueError(f"Missing package relationship target: {target}")
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        ids = [row.attrib["id"] for row in slide.findall(".//p:cNvPr", NS)]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate PowerPoint object IDs")
        if slide.findall(".//p:pic", NS) or any(name.startswith("ppt/media/") for name in names):
            raise ValueError("Native figure unexpectedly contains media")
        return {"groups": len(slide.findall(".//p:grpSp", NS)),
                "shapes": len(slide.findall(".//p:sp", NS)),
                "connectors": len(slide.findall(".//p:cxnSp", NS)),
                "text_runs": len(slide.findall(".//a:t", NS)), "embedded_images": 0}


def _cairo_source(context: Any, value: str, opacity: float = 1) -> None:
    rgb = _color(value)
    context.set_source_rgba(*(int(rgb[index:index + 2], 16) / 255 for index in (0, 2, 4)), opacity)


def _cairo_font(context: Any, style: dict, cairo: Any) -> None:
    context.select_font_face(style.get("font", "Arial"),
                             cairo.FONT_SLANT_ITALIC if style.get("italic") else cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_BOLD if style.get("bold") else cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(style.get("size", 20))


def render_preview(scene: dict, output: Path, scale: float = 1.5) -> None:
    """Render the scene for layout inspection; this is not an Office rendering.

    Args:
        scene: Same scene consumed by write_pptx.
        output: Destination PNG path.
        scale: Raster pixels per logical scene pixel.
    """
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, round(scene["width"] * scale), round(scene["height"] * scale))
    context = cairo.Context(surface)
    context.scale(scale, scale)
    context.set_source_rgb(1, 1, 1)
    context.paint()
    for group in scene["groups"]:
        for shape in group["shapes"]:
            context.save()
            kind = shape.get("kind", shape.get("type"))
            if kind == "text":
                lines = _text_lines(shape)
                sizes = [max(({**shape, **run}.get("size", 20) for run in line), default=shape.get("size", 20)) for line in lines]
                heights = [size * shape.get("line_spacing", 1.1) for size in sizes]
                total = sum(heights)
                top = shape["y"] + ((shape["h"] - total) / 2 if shape.get("valign", "center") == "center" else 0)
                for runs, size, height in zip(lines, sizes, heights):
                    widths = []
                    for run in runs:
                        merged = {**shape, **run}
                        _cairo_font(context, merged, cairo)
                        widths.append(context.text_extents(str(run.get("text", ""))).x_advance)
                    x = shape["x"]
                    if shape.get("align") == "center":
                        x += (shape["w"] - sum(widths)) / 2
                    elif shape.get("align") == "right":
                        x += shape["w"] - sum(widths)
                    for run, width in zip(runs, widths):
                        merged = {**shape, **run}
                        _cairo_font(context, merged, cairo)
                        _cairo_source(context, merged.get("color", "20242B"))
                        context.move_to(x, top + size * 0.84 - merged.get("baseline", 0) * merged.get("size", 20))
                        context.show_text(str(run.get("text", "")))
                        x += width
                    top += height
                context.restore()
                continue
            context.new_path()
            endpoint = previous = None
            if kind == "line":
                previous, endpoint = (shape["x1"], shape["y1"]), (shape["x2"], shape["y2"])
                context.move_to(*previous)
                context.line_to(*endpoint)
            elif kind == "path":
                for command in shape["commands"]:
                    if command[0] == "M":
                        context.move_to(*command[1:])
                        endpoint = tuple(command[1:])
                    elif command[0] == "L":
                        previous, endpoint = endpoint, tuple(command[1:])
                        context.line_to(*endpoint)
                    elif command[0] == "C":
                        previous, endpoint = tuple(command[3:5]), tuple(command[5:7])
                        context.curve_to(*command[1:])
                    elif command[0] == "Z":
                        context.close_path()
            else:
                x, y, w, h = _bounds(shape)
                if kind == "ellipse":
                    context.save()
                    context.translate(x + w / 2, y + h / 2)
                    context.scale(w / 2, h / 2)
                    context.arc(0, 0, 1, 0, 2 * pi)
                    context.restore()
                elif kind == "roundRect":
                    radius = max(0, min(shape.get("radius", 8), min(w, h) / 2))
                    for cx, cy, start in ((x + w - radius, y + radius, -pi / 2), (x + w - radius, y + h - radius, 0),
                                          (x + radius, y + h - radius, pi / 2), (x + radius, y + radius, pi)):
                        context.arc(cx, cy, radius, start, start + pi / 2)
                    context.close_path()
                else:
                    context.rectangle(x, y, w, h)
            if shape.get("fill") is not None:
                _cairo_source(context, shape["fill"], shape.get("opacity", 1))
                context.fill_preserve()
            if shape.get("stroke") is not None:
                _cairo_source(context, shape["stroke"])
                context.set_line_width(shape.get("stroke_width", 1))
                context.set_line_join(cairo.LINE_JOIN_ROUND)
                if shape.get("dash"):
                    context.set_dash([6, 4])
                context.stroke()
                if shape.get("arrow") and endpoint and previous:
                    angle = atan2(endpoint[1] - previous[1], endpoint[0] - previous[0])
                    length = max(7, shape.get("stroke_width", 1) * 3)
                    context.set_dash([])
                    context.move_to(*endpoint)
                    for delta in (-0.5, 0.5):
                        context.line_to(endpoint[0] - length * cos(angle + delta), endpoint[1] - length * sin(angle + delta))
                    context.close_path()
                    context.fill()
            else:
                context.new_path()
            context.restore()
    output.parent.mkdir(parents=True, exist_ok=True)
    surface.write_to_png(str(output))


def main() -> None:
    """Export a scene JSON file and optionally render an independent preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    evidence = write_pptx(scene, args.output)
    if args.preview:
        render_preview(scene, args.preview)
    logger.info("Exported %s: %s", args.output, evidence)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
