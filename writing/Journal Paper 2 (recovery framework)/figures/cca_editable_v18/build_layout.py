"""Author the editable CCA figure as a native PowerPoint scene."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PURPLE = "6522B8"
BLUE = "183EC6"
INK = "15191D"
PALE = "EFF7FE"
EDGE = "76A3DE"
GREEN = "188725"
RED = "D62626"
ORANGE = "E77A27"
PINK = "F335AE"
TEAL = "39C5C5"

scene = {"width": 1024, "height": 1536, "title": "Central Controller Agent — editable safety monitor figure", "groups": []}
active = None
logger = logging.getLogger(__name__)


def group(name: str) -> None:
    """Start a named native PowerPoint object group."""
    global active
    active = {"name": name, "shapes": []}
    scene["groups"].append(active)


def add(kind: str, **fields) -> None:
    """Append one native object to the active group."""
    assert active is not None
    active["shapes"].append({"kind": kind, **fields})


def rect(x, y, w, h, *, fill=None, stroke=PURPLE, width=1.0, rounded=True, opacity=1.0, radius=8) -> None:
    """Add an editable rectangle."""
    add("roundRect" if rounded else "rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, stroke_width=width, opacity=opacity, radius=radius)


def ellipse(x, y, w, h, *, fill=None, stroke=INK, width=1.5, opacity=1.0, dash=False) -> None:
    """Add an editable ellipse."""
    add("ellipse", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, stroke_width=width, opacity=opacity, dash=dash)


def line(x1, y1, x2, y2, *, color=INK, width=2.0, arrow=False, dash=False) -> None:
    """Add an editable line or directed connector."""
    add("line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width, arrow=arrow, dash=dash)


def path(commands, *, fill=None, color=INK, width=2.0, arrow=False, dash=False) -> None:
    """Add an editable DrawingML freeform."""
    add("path", commands=commands, fill=fill, stroke=color, stroke_width=width, arrow=arrow, dash=dash)


def text(x, y, w, h, value, *, size=20, color=BLUE, bold=False, italic=False, align="center", font="Arial", valign="center") -> None:
    """Add editable text without implicit wrapping."""
    fields = {"x": x, "y": y, "w": w, "h": h, "font": font, "size": size, "color": color, "bold": bold, "italic": italic, "align": align, "valign": valign}
    fields["text" if isinstance(value, str) else "lines"] = value
    add("text", **fields)


def math_runs(parts, size=26, color=BLUE):
    """Return runs with native subscript baselines."""
    runs = []
    for part in parts:
        if isinstance(part, tuple):
            value, baseline = part
            runs.append({"text": value, "font": "Cambria", "size": size * .68, "italic": True, "color": color, "baseline": baseline})
        else:
            runs.append({"text": part, "font": "Cambria", "size": size, "italic": True, "color": color})
    return [runs]


def tuple_label(x, y, w, h, pair, monitor, color=INK) -> None:
    """Render the AP projection and monitor state as editable runs."""
    text(x, y, w, h, math_runs([f"(({pair}), q", (monitor, -.27), ")"], 27, color), size=27, color=color)


def badge(number, title, y) -> None:
    """Draw a section badge and heading."""
    ellipse(28, y + 3, 45, 45, fill=PURPLE, stroke=PURPLE, width=0)
    text(29, y + 2, 43, 44, str(number), size=31, color="FFFFFF", bold=True)
    text(94, y + 1, 878, 47, title, size=32, color=PURPLE, bold=True, align="left")


def robot(points, *, blue=False) -> None:
    """Draw a robot using individually editable links and joints."""
    fill = "DCF1FC" if blue else "F1F2F2"
    bx, by = points[0]
    ellipse(bx - 22, by - 1, 45, 13, fill="D5D7D8", stroke=None, width=0)
    rect(bx - 15, by - 6, 31, 10, fill="D7DADD", stroke=INK, width=1.8, rounded=False)
    for first, second in zip(points[:-1], points[1:], strict=True):
        line(*first, *second, width=14)
        line(*first, *second, color=fill, width=9)
    for index, (cx, cy) in enumerate(points[:-1]):
        radius = 14 if index == 0 else 11
        ellipse(cx - radius, cy - radius, 2 * radius, 2 * radius, fill=fill, stroke=INK, width=2)
    gx, gy = points[-1]
    rect(gx - 6, gy - 4, 12, 11, fill=fill, stroke=INK, width=1.4, rounded=False)
    line(gx - 5, gy + 7, gx - 7, gy + 16, width=2)
    line(gx + 5, gy + 7, gx + 7, gy + 16, width=2)
    line(gx - 7, gy + 16, gx - 3, gy + 19, width=2)
    line(gx + 7, gy + 16, gx + 3, gy + 19, width=2)


def part(x, y, w=18, h=18) -> None:
    """Draw an editable cylindrical part."""
    rect(x, y, w, h, fill=TEAL, stroke=INK, width=1.3, rounded=False)
    ellipse(x, y + h - 4, w, 7, fill=TEAL, stroke=INK, width=1.3)
    ellipse(x, y - 4, w, 8, fill="8CE4DE", stroke=INK, width=1.3)


group("Page border")
rect(3, 3, 1018, 1530, fill="FFFFFF", stroke=PURPLE, width=5, radius=22)

group("Heading and editable icons")
text(299, 10, 447, 43, "Central Controller Agent", size=31, color=INK, bold=True)
rect(365, 53, 302, 28, fill="F8F3FD", stroke=PURPLE, width=1)
text(367, 54, 298, 25, "Mutual-exclusion example (SAFE_2)", size=16.5, color=BLUE)
rect(257, 27, 35, 27, fill=INK, stroke=INK, width=1)
ellipse(264, 34, 7, 7, fill="FFFFFF", stroke=None, width=0)
ellipse(278, 34, 7, 7, fill="FFFFFF", stroke=None, width=0)
line(267, 46, 282, 46, color="FFFFFF", width=1.5)
line(274, 27, 274, 21, width=2)
ellipse(270, 15, 8, 8, fill=INK, stroke=None, width=0)
rect(252, 32, 4, 16, fill=INK, stroke=None, width=0)
rect(293, 32, 4, 16, fill=INK, stroke=None, width=0)
path([["M", 266, 55], ["L", 283, 55], ["L", 287, 64], ["L", 262, 64], ["Z"]], fill=INK, color=None, width=0)
path([["M", 758, 22], ["L", 780, 14], ["L", 802, 22], ["L", 802, 40], ["C", 802, 54, 788, 63, 780, 68], ["C", 772, 63, 758, 54, 758, 40], ["Z"]], fill=INK, color=None, width=0)
path([["M", 769, 39], ["L", 777, 47], ["L", 791, 31]], color="FFFFFF", width=3)

group("Panel 1 background and heading")
rect(20, 87, 984, 671, fill=None, stroke=PURPLE, width=1.7)
badge(1, "Safety monitor construction", 89)

group("SAFE_2 and compiled DFA")
rect(33, 144, 958, 249, fill=PALE, stroke=EDGE, width=1)
text(50, 153, 466, 38, "Safety monitor (DFA for SAFE_2)", size=24, bold=True, align="left")
rect(53, 209, 280, 117, fill="E5F2FE", stroke=EDGE, width=1)
text(72, 217, 241, 35, "SAFE_2", size=29, bold=True)
text(65, 266, 256, 43, "G !(ap001 & ap002)", size=23, italic=True, font="Cambria")
text(352, 231, 96, 28, "Compile", size=20)
line(350, 276, 449, 276, color=PURPLE, width=8, arrow=True)
line(469, 271, 498, 271, width=2.5, arrow=True)
ellipse(500, 226, 90, 90, fill="F2FAEE", stroke=GREEN, width=3)
ellipse(507, 233, 76, 76, fill=None, stroke=GREEN, width=2)
ellipse(826, 230, 86, 86, fill="FFEEEE", stroke=RED, width=3)
text(511, 238, 68, 58, math_runs(["q", ("0", -.27)], 37, BLUE), size=37)
text(835, 238, 68, 58, math_runs(["q", ("d", -.27)], 37, BLUE), size=37)
line(591, 271, 826, 271, width=2.6, arrow=True)
text(604, 235, 217, 30, "ap001 & ap002", size=23, italic=True, font="Cambria")
path([["M", 531, 226], ["C", 501, 176, 587, 175, 557, 226]], color=INK, width=2.4, arrow=True)
text(576, 178, 237, 33, "!(ap001 & ap002)", size=23, italic=True, font="Cambria")
path([["M", 851, 230], ["C", 823, 191, 905, 183, 878, 230]], color=INK, width=2.4, arrow=True)
text(812, 166, 110, 31, "true", size=24, italic=True, font="Cambria")
text(460, 318, 170, 34, "No violation", size=25, color=GREEN, bold=True)
text(798, 318, 145, 34, "Violation", size=25, color=RED, bold=True)
text(302, 354, 470, 28, "Monitor state retained from accepted history.", size=18)

group("New recovery and bound program")
rect(33, 421, 285, 329, fill="FFF8EE", stroke=ORANGE, width=1)
text(44, 427, 263, 33, "New recovery", size=26, color=RED, bold=True)
rect(44, 463, 263, 190, fill="FFFBF5", stroke=ORANGE, width=.7)
text(99, 497, 42, 54, "e", size=40, color=RED, italic=True, font="Cambria")
text(132, 488, 30, 29, "R", size=24, color=RED, italic=True, font="Cambria")
text(132, 522, 26, 28, "k", size=23, color=RED, italic=True, font="Cambria")
text(157, 507, 19, 41, ",", size=33, color=RED, italic=True, font="Cambria")
text(181, 497, 42, 54, "x", size=40, color=RED, italic=True, font="Cambria")
text(215, 488, 30, 29, "R", size=24, color=RED, italic=True, font="Cambria")
text(215, 522, 26, 28, "k", size=23, color=RED, italic=True, font="Cambria")
text(57, 551, 152, 42, "RA program", size=25)
text(211, 549, 59, 43, "PC", size=31, italic=True, font="Cambria")
text(264, 574, 20, 25, "k", size=20, italic=True, font="Cambria")
text(54, 598, 244, 37, "... move_cartesian ...", size=21, color=INK)
text(45, 663, 260, 47, "Event and proposed\nsuccessor", size=18, color=INK)

group("Primitive models and observations")
rect(334, 421, 324, 329, fill="F4FAFE", stroke=EDGE, width=1)
text(345, 427, 301, 32, "Primitive models", size=25, bold=True)
ellipse(413, 545, 142, 59, fill="FCE4F1", stroke=PINK, width=1.5, dash=True)
ellipse(437, 533, 56, 57, fill="AAD5F5", stroke=None, width=0, opacity=.55)
ellipse(584, 567, 63, 29, fill="E6F5FE", stroke="3EA9E9", width=1.1, dash=True)
robot([(380, 584), (389, 544), (415, 494), (454, 516), (463, 538)])
robot([(617, 584), (611, 542), (578, 494), (552, 516), (551, 523)], blue=True)
part(475, 564, 19, 18)
part(615, 576, 16, 15)
text(348, 475, 43, 30, math_runs(["r", ("1", -.27)], 24), size=24)
text(605, 475, 42, 30, math_runs(["r", ("2", -.27)], 24), size=24)
text(346, 603, 300, 26, "Motion + state + geometry", size=18.5)
text(358, 632, 276, 25, "region_occupancy", size=20, bold=True)
rect(371, 656, 246, 32, fill="E9F5FC", stroke=EDGE, width=.7)
text(377, 657, 234, 30, "r1:  1      r2:  0", size=22, italic=True, bold=True)
text(345, 694, 302, 27, "Example intermediate observation", size=17)
text(350, 720, 293, 25, "Other resources’ behavior", size=17)

group("Predefined AP evaluation")
rect(673, 421, 318, 329, fill="FCFAFF", stroke=PURPLE, width=1)
text(681, 427, 302, 31, "Predefined AP evaluation", size=22.5, bold=True)
rect(683, 459, 297, 101, fill="FAF7FF", stroke="BA9BDC", width=.7)
text(692, 464, 278, 29, "ap001:  r1 occupancy", size=20, bold=True)
text(692, 493, 278, 29, "ap002:  r2 occupancy", size=20, bold=True)
text(689, 522, 283, 30, "region:  assembly station", size=19)
line(799, 564, 799, 610, color=PURPLE, width=7, arrow=True)
text(812, 572, 110, 26, "binding", size=18, align="left")
rect(748, 615, 86, 115, fill="F4F6FF", stroke=BLUE, width=1.5)
path([["M", 760, 627], ["L", 822, 627], ["L", 798, 653], ["L", 798, 666], ["L", 788, 660], ["L", 788, 653], ["Z"]], fill="E2EDFF", color=BLUE, width=1.8)
text(751, 672, 80, 52, "Match\nto APs", size=19)
rect(866, 617, 115, 48, fill="F2F7FF", stroke=BLUE, width=1)
rect(866, 682, 115, 48, fill="F7F7F7", stroke=INK, width=1)
text(870, 623, 107, 36, "ap001 = 1", size=19, bold=True)
text(870, 688, 107, 36, "ap002 = 0", size=19, color=INK, bold=True)
line(835, 641, 865, 641, width=1.8, arrow=True)
line(835, 706, 865, 706, width=1.8, arrow=True)

group("Panel 1 connections")
line(312, 576, 348, 576, color=PURPLE, width=8, arrow=True)
line(635, 666, 746, 666, color=PURPLE, width=8, arrow=True)
text(630, 630, 117, 28, "observations", size=17.5)
line(800, 420, 800, 381, color=PURPLE, width=7.5, arrow=True)
text(817, 387, 131, 30, "AP values", size=19, align="left")
line(512, 759, 512, 783, color=PURPLE, width=9, arrow=True)

group("Panel 2 background and heading")
rect(20, 781, 984, 288, fill=None, stroke=PURPLE, width=1.7)
badge(2, "Parallel composition", 783)
rect(33, 828, 958, 31, fill="EAF5FE", stroke=EDGE, width=.8)
text(44, 829, 936, 28, "Recovery + nominal + running tasks + safety monitors", size=20)

group("Local view of composed states")
rect(181, 872, 211, 56, fill="F4F7FC", stroke="202A54", width=1.5)
rect(642, 872, 205, 56, fill="FFF0EE", stroke=RED, width=1.5)
rect(181, 974, 211, 56, fill="F4F7FC", stroke="202A54", width=1.5)
rect(642, 974, 205, 56, fill="ECF8E8", stroke=GREEN, width=1.5)
tuple_label(185, 879, 203, 44, "0,1", "0")
tuple_label(646, 879, 197, 44, "1,1", "d", RED)
tuple_label(185, 981, 203, 44, "0,0", "0")
tuple_label(646, 981, 197, 44, "1,0", "0")
line(393, 900, 641, 900, width=2.5, arrow=True)
line(285, 929, 285, 973, width=2.5, arrow=True)
line(393, 1002, 641, 1002, width=2.5, arrow=True)
text(423, 870, 192, 27, "ap001: 0 → 1", size=21)
text(300, 937, 180, 29, "ap002: 1 → 0", size=21, align="left")
text(423, 972, 192, 27, "ap001: 0 → 1", size=21)
text(243, 1037, 590, 27, "Local view: ((ap001, ap002), monitor state)", size=18.5)

group("Panel 2 to panel 3")
line(512, 1070, 512, 1095, color=PURPLE, width=9, arrow=True)

group("Panel 3 background and heading")
rect(20, 1091, 984, 398, fill=None, stroke=PURPLE, width=1.7)
badge(3, "Timed safety validation", 1093)
rect(33, 1139, 958, 32, fill="EAF5FE", stroke=EDGE, width=.8)
text(44, 1140, 936, 29, "Projected execution", size=22, bold=True)


def timeline(*, conflict: bool) -> None:
    """Draw a fully editable occupancy timeline."""
    offset = 0 if conflict else 488
    col = RED if conflict else GREEN
    left = 33 + offset
    rect(left, 1178, 476 if conflict else 470, 170, fill="FFFBFB" if conflict else "FBFEFA", stroke=col, width=.8)
    rect(left + 9, 1184, 458 if conflict else 452, 30, fill="FFF0EF" if conflict else "EEF9E9", stroke=col, width=.5)
    text(left + 17, 1185, 442, 28, "Entry before exit" if conflict else "Exit before entry", size=23, color=col, bold=True)
    if conflict:
        rect(194, 1220, 104, 76, fill="F36A67", stroke=None, width=0, rounded=False, opacity=.13)
    for i, yy in enumerate([1237, 1273], start=1):
        text(56 + offset, yy - 20, 45, 33, math_runs(["r", (str(i), -.27)], 24), size=24)
        line(115 + offset, yy, 480 + offset, yy, color="7FACD4", width=1.2)
    rect(194 if conflict else 788, 1222, 175 if conflict else 127, 23, fill="B8DFFE", stroke=BLUE, width=1.2, rounded=False)
    rect(130 if conflict else 608, 1259, 168 if conflict else 99, 23, fill="CDCDCD", stroke=INK, width=1.0, rounded=False)
    line(115 + offset, 1296, 370 + offset, 1296, width=2.0, arrow=True)
    text(378 + offset, 1282, 113, 29, "Model time", size=18.5)
    text(left + 47, 1308, 388, 35, "Mutex violation" if conflict else "No mutex conflict", size=24, color=col, bold=True)


group("Entry before exit timeline")
timeline(conflict=True)
group("Exit before entry timeline")
timeline(conflict=False)

group("CCA decision and RA execution")
rect(33, 1360, 958, 119, fill="FBF8FF", stroke=PURPLE, width=1)
text(207, 1363, 611, 35, "CCA decision from composition", size=25, bold=True)
rect(113, 1404, 254, 56, fill="EBF8E4", stroke=GREEN, width=1.4)
rect(431, 1404, 261, 56, fill="F4F7FF", stroke=BLUE, width=1.0)
rect(712, 1404, 198, 56, fill="FFF0EE", stroke=RED, width=1.4)
text(124, 1411, 230, 42, "allow", size=30, color=INK, bold=True)
text(724, 1411, 174, 42, "block", size=30, color=RED, bold=True)
line(378, 1432, 430, 1432, width=2.8, arrow=True)
points = []
for i in range(32):
    a = i * math.pi / 16
    rad = 19 if i % 4 in (0, 1) else 15
    points.append(["M" if i == 0 else "L", 458 + rad * math.cos(a), 1432 + rad * math.sin(a)])
path([*points, ["Z"]], fill=INK, color=None, width=0)
ellipse(450, 1424, 16, 16, fill="FFFFFF", stroke=None, width=0)
text(483, 1413, 200, 37, "RA execution", size=24, bold=True)

if __name__ == "__main__":
    output = ROOT / "scene.json"
    output.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Saved %s (%s groups)", output, len(scene["groups"]))
