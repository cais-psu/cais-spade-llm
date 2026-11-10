"""Shared look of the result figures: Arial text sized for print and the results page's colors."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.figure import Figure

# Arial as Windows installs it, seen from WSL; matplotlib does not look there on its own. Liberation Sans has the
# same metrics and stands in where these files are missing.
WINDOWS_FONTS = Path("/mnt/c/Windows/Fonts")
ARIAL_FILES = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")

# Elsevier full text width (190 mm) and column width (90 mm).
FULL_WIDTH_IN = 190 / 25.4
COLUMN_WIDTH_IN = 90 / 25.4

INK = "#15181d"
INK_2 = "#4b515b"
MUTED = "#7c838d"
HAIR = "#e1e4e8"
AXIS = "#c3c7cd"
# The results page's color for each audit outcome, so a figure and the page read alike.
OUTCOME_COLORS = {
    "passed": "#2a78d6",
    "infeasible": "#eb6834",
    "physically_unsupported": "#1baf7a",
    "unsafe": "#eda100",
    "no_nominal_continuation": "#e87ba4",
    "unresolved": MUTED,
    "did_not_finish": "#c4c7cc",
    "not audited": "#f6f7f9",
}
# How the figures name each audit outcome, with the check that sets it. The records keep the audit's names.
OUTCOME_LABELS = {
    "passed": "Passed",
    "infeasible": "Infeasible: symbolic (RA)",
    "physically_unsupported": "Infeasible: physical (RA)",
    "unsafe": "Unsafe (CCA)",
    "no_nominal_continuation": "Blocking (CCA)",
    "not audited": "Not audited",
}
# The results page's model colors (light theme), by the variable the page data names for each model: red, blue and
# green at rgb(80% 0 0), rgb(0 20% 80%) and rgb(0 55% 0), which Fig. 11's outcomes do not use (user decision,
# 2026-11-09). And the mark shape the page data names for each model.
MODEL_COLORS = {"var(--m1)": "#cc0000", "var(--m2)": "#0033cc", "var(--m3)": "#008c00"}
MARKERS = {"square": "s", "circle": "o", "diamond": "D"}

# Setting indices as the results page groups them: what the LLM gets, then outline-level above event-level.
SETTING_GROUPS = (("Baseline", (0, 2)), ("Informed baseline", (1, 3)), ("Neuro-symbolic framework", (4, 5)))
# Row labels: the generation level of each setting, as the captions define them.
LEVEL_ACRONYMS = {"outline-level": "OL", "event-level": "EL"}
# Rows of a chart by setting, in data units: each group's header, then its two rows.
GROUP_STEP = 2.9
HEADER_OFFSET = 0.2
ROW_OFFSETS = (0.95, 1.95)
BAR_HEIGHT = 0.72


def level_label(setting_name: str) -> str:
    """Return the acronym of the generation level of a setting named "<method>, <level>"."""
    return LEVEL_ACRONYMS[setting_name.rsplit(", ", 1)[1]]


@cache
def _register_arial() -> None:
    for name in ARIAL_FILES:
        if (WINDOWS_FONTS / name).exists():
            font_manager.fontManager.addfont(str(WINDOWS_FONTS / name))


def apply_style() -> None:
    """Set print sizes and Arial text."""
    _register_arial()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "font.size": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.size": 0,
        "legend.frameon": False,
        # Keep text as text: SVG text stays editable in PowerPoint, PDF text embeds as TrueType.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: Figure, out_dir: Path, name: str) -> list[Path]:
    """Write a figure as SVG with editable text, for PowerPoint, and as a 300 dpi PNG.

    Returns:
        The written paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.svg", out_dir / f"{name}.png"]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return paths
