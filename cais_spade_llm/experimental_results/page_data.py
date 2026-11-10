"""Read the data block of the RCIM recovery results page.

The figures and tables use the numbers the page shows, so a figure never disagrees with the page. When the records
change, rebuild the page with ``results_page.py`` first.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULTS_PAGE = REPO / "writing" / "Journal Paper 2 (recovery framework)" / "figures" / "results_page" / "results.html"
# results_page.py writes the whole data block as one line of JSON.
_DATA_LINE = re.compile(r"^const DATA = (\{.*\});$", re.MULTILINE)


def read_page_data(page: Path = RESULTS_PAGE) -> dict[str, Any]:
    """Return the data block of a results page built from the batch records.

    Args:
        page: The rendered results page.

    Returns:
        The page's ``DATA`` object.

    Raises:
        ValueError: If the page has no data block or shows placeholder numbers.
    """
    match = _DATA_LINE.search(page.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{page} holds no data block; rebuild it with results_page.py")
    data = json.loads(match.group(1))
    if data["mode"] != "records":
        raise ValueError(f"{page} shows placeholder numbers; rebuild it from the batch records")
    return data


def pooled_cell(data: dict[str, Any], setting: int, scenario: str) -> dict[str, Any]:
    """Return the cell of one setting and scenario, pooled over every model."""
    return data["cells"]["all"][setting][scenario]


def model_cell(data: dict[str, Any], model_id: str, setting: int, scenario: str) -> dict[str, Any]:
    """Return one model's cell of one setting and scenario."""
    return data["cells"][model_id][setting][scenario]
